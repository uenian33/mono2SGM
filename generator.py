import random
import numpy as np
from PIL import Image  # using pillow-simd for increased speed
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import torch
import torch.utils.data as data
from torchvision import transforms
import torch.nn.functional as F
import skimage
from skimage.filters import gaussian, sobel
from skimage import io
from skimage.color import rgb2gray

from scipy.interpolate import griddata
import cv2
cv2.setNumThreads(0)

from utils.sgm.sgm import *
from utils.WB.classes import WBsRGB as wb_srgb

'''
void computeCorrespondingImage(const cv::Mat &img, const cv::Mat &disparity, cv::Mat &dest,
                           const bool leftInput, const int disparityScale)
{
    const int shiftDirection = leftInput ? -1 : 1;
    dest.create(img.rows, img.cols, img.type());

    for (int i(0) ; i < img.rows ; ++i) {
        for (int j(0) ; j < img.cols  ; ++j) {
            const uchar d(disparity.at<const uchar>(i, j));
            const int computedColumn(j + shiftDirection * (d / disparityScale));

            // No need to consider pixels who would be outside of the image's bounds
            if (d > 0 && computedColumn >= 0 && computedColumn < img.cols) {
                dest.at<cv::Vec3b>(i, computedColumn) = img.at<const cv::Vec3b>(i, j);
            }
        }
    }
}

def right_eye_view(left, bg, disparity, disparityScale):
    shiftDirection = -1
    rows, cols = left.shape(1), left.shape(0)

    for wi in range(rows):
        for hi in range(cols):
            d = disparity[wi, hi]
            computedColumn = j + shiftDirection * d / disparityScale
            # No need to consider pixels who would be outside of the image's bounds
            if (d > 0 and computedColumn >= 0 and computedColumn < cols):
                bg[i, computedColumn] = left[i, j]
'''

def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def normalise_image(img):
    """ Normalize image to [0, 1] range for visualization """
    # img_max = float(img.max().cpu().data)
    # img_min = float(img.min().cpu().data)

    img_max = float(img.max())
    img_min = float(img.min())
    denom = img_max - img_min if img_max != img_min else 1e5
    return (img - img_min) / denom


def transfer_color(target, source):
    target = target.astype(float) / 255
    source = source.astype(float) / 255

    target_means = target.mean(0).mean(0)
    target_stds = target.std(0).std(0)

    source_means = source.mean(0).mean(0)
    source_stds = source.std(0).std(0)

    target -= target_means
    target /= target_stds / source_stds
    target += source_means

    target = np.clip(target, 0, 1)
    target = (target * 255).astype(np.uint8)

    return target

class PreProcessor(object):
    """docstring for PreProcessor"""
    def __init__(self, wb_model_file_name='utils/WB/models/',
                denoise=False, 
                white_balance=False,
                blur_block=(1,1)):
        super(PreProcessor, self).__init__()
        # load a deep neural netowrk based white balance model
        self.wbModel = wb_srgb.WBsRGB(gamut_mapping=2,upgraded=0, weight_pth=wb_model_file_name)
        self.denoise = denoise
        self.white_balance = white_balance
        self.blur_block = blur_block


    def simple_preprocess(self,img,  reshape=False):
        """
        simple image preprocessing pipeline using denoise and white balance model
        """
        im_h, im_w, _ = img.shape
        #print(im_w, im_h)


        if reshape:
            r_w, r_h = int(im_w*0.4), int(im_h*0.4)
            img = cv2.resize(img, dsize=(r_w, r_h), interpolation=cv2.INTER_CUBIC)


        if self.denoise:
            img = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)

        if self.white_balance:
            img = self.wbModel.correctImage(img) * 255 # white balance it

            img= np.uint8(img)



        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img = cv2.GaussianBlur(img, self.blur_block, 0, 0)
        plt.imshow(img, interpolation='nearest')
        plt.show()


        return img
        
        

class StereoConverter():

    def __init__(self,
                 feed_height,
                 feed_width,
                 max_disparity,
                 is_train=True,
                 disable_normalisation=False,
                 keep_aspect_ratio=True,
                 disable_synthetic_augmentation=False,
                 disable_sharpening=False,
                 monodepth_model='midas',
                 disable_background=False,
                 fix_disparity_scale=True,
                 max_disparity_range=[40, 196],
                 crop_images=False,
                 **kwargs):

        super(StereoConverter, self).__init__()

        self.feed_height = feed_height
        self.feed_width = feed_width
        self.max_disparity = max_disparity
        self.disable_synthetic_augmentation = disable_synthetic_augmentation
        self.disable_sharpening = disable_sharpening
        self.monodepth_model = monodepth_model
        self.disable_background = disable_background
        self.keep_aspect_ratio = keep_aspect_ratio
        self.crop_images = crop_images

        # do image generation for a wider image so we can crop off missing pixels
        self.process_width = self.feed_width + self.max_disparity

        self.xs, self.ys = np.meshgrid(np.arange(self.process_width), np.arange(self.feed_height))

        # We need to specify augmentations differently in newer versions of torchvision.
        # We first try the newer tuple version; if this fails we fall back to scalars
        try:
            self.stereo_brightness = (0.8, 1.2)
            self.stereo_contrast = (0.8, 1.2)
            self.stereo_saturation = (0.8, 1.2)
            self.stereo_hue = (-0.01, 0.01)
            transforms.ColorJitter.get_params(
                self.stereo_brightness, self.stereo_contrast, self.stereo_saturation,
                self.stereo_hue)
        except TypeError:
            self.stereo_brightness = 0.2
            self.stereo_contrast = 0.2
            self.stereo_saturation = 0.2
            self.stereo_hue = 0.01

        self.silly_svsm = False
        self.fix_disparity_scale = fix_disparity_scale
        self.max_disparity_range = max_disparity_range

    
    def load_images(self, path, bg_pth, do_flip=False):
        """ Load an image to use as left and a random background image to fill in occlusion holes"""

        image = pil_loader(path)
        if do_flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        background = pil_loader(bg_pth)                                                     

        return image, background

    def load_disparity(self, path, do_flip=False):
        try:
            disparity = np.load(path)
        except:
            img = np.array(Image.open(path))
            disparity = (1 - img / 255) * 160
        if do_flip:
            disparity = disparity[:, ::-1]
        return disparity

    def process_disparity(self, disparity):
        """ Depth predictions have arbitrary scale - need to convert to a pixel disparity"""

        disparity = disparity.copy()

        # make disparities positive
        min_disp = disparity.min()
        if min_disp < 0:
            disparity += np.abs(min_disp)

        if random.random() < 0.01:
            # make max warped disparity bigger than network max -> will be clipped to max disparity,
            # but will mean network is robust to disparities which are too big
            max_disparity_range = (self.max_disparity * 1.05, self.max_disparity * 1.15)
        else:
            max_disparity_range = self.max_disparity_range

        disparity /= disparity.max()  # now 0-1

        if not self.fix_disparity_scale:
            scaling_factor = (max_disparity_range[0] + random.random() *
                              (max_disparity_range[1] - max_disparity_range[0]))
        else:
            scaling_factor = max_disparity_range[0]

        disparity *= scaling_factor

        if not self.disable_sharpening:
            # now find disparity gradients and set to nearest - stop flying pixels
            edges = sobel(disparity) > 3
            disparity[edges] = 0
            mask = disparity > 0

            try:
                disparity = griddata(np.stack([self.ys[mask].ravel(), self.xs[mask].ravel()], 1),
                                     disparity[mask].ravel(), np.stack([self.ys.ravel(),
                                                                        self.xs.ravel()], 1),
                                     method='nearest').reshape(self.feed_height, self.process_width)
            except (ValueError, IndexError) as e:
                pass  # just return disparity

        return disparity

    def prepare_sizes(self, inputs):

        height, width, _ = np.array(inputs['left_image']).shape

        if self.keep_aspect_ratio:
            if self.feed_height <= height and self.process_width <= width:
                # can simply crop the image
                target_height = height
                target_width = width

            else:
                # check the constraint
                current_ratio = height / width
                target_ratio = self.feed_height / self.process_width

                if current_ratio < target_ratio:
                    # height is the constraint
                    target_height = self.feed_height
                    target_width = int(self.feed_height / height * width)

                elif current_ratio > target_ratio:
                    # width is the constraint
                    target_height = int(self.process_width / width * height)
                    target_width = self.process_width

                else:
                    # ratio is the same - just resize
                    target_height = self.feed_height
                    target_width = self.process_width

        else:
            target_height = self.feed_height
            target_width = self.process_width

        inputs = self.resize_all(inputs, target_height, target_width)

        # now do cropping
        if target_height == self.feed_height and target_width == self.process_width:
            # we are already at the correct size - no cropping
            pass
        else:
            self.crop_all(inputs)

        return inputs

    def crop_all(self, inputs):

        # get crop parameters
        height, width, _ = np.array(inputs['left_image']).shape
        top = int(random.random() * (height - self.feed_height))
        left = int(random.random() * (width - self.process_width))
        right, bottom = left + self.process_width, top + self.feed_height

        for key in ['left_image', 'background']:
            inputs[key] = inputs[key].crop((left, top, right, bottom))
        inputs['loaded_disparity'] = inputs['loaded_disparity'][top:bottom, left:right]

        return inputs

    @staticmethod
    def resize_all(inputs, height, width):

        # images
        img_resizer = transforms.Resize(size=(height, width))
        for key in ['left_image', 'background']:
            inputs[key] = img_resizer(inputs[key])
        # disparity - needs rescaling
        disp = inputs['loaded_disparity']
        disp *= width / disp.shape[1]

        disp = cv2.resize(disp.astype(float), (width, height))  # ensure disp is float32 for cv2
        inputs['loaded_disparity'] = disp

        return inputs

    def get_occlusion_mask(self, shifted):

        mask_up = shifted > 0
        mask_down = shifted > 0

        shifted_up = np.ceil(shifted)
        shifted_down = np.floor(shifted)

        for col in range(self.process_width - 2):
            loc = shifted[:, col:col + 1]  # keepdims
            loc_up = np.ceil(loc)
            loc_down = np.floor(loc)

            _mask_down = ((shifted_down[:, col + 2:] != loc_down) * (
            (shifted_up[:, col + 2:] != loc_down))).min(-1)
            _mask_up = ((shifted_down[:, col + 2:] != loc_up) * (
            (shifted_up[:, col + 2:] != loc_up))).min(-1)

            mask_up[:, col] = mask_up[:, col] * _mask_up
            mask_down[:, col] = mask_down[:, col] * _mask_down

        mask = mask_up + mask_down
        return mask

   

    def project_image(self, image, disp_map, background_image):
        '''
        generating right eye view from esimated disparity and left eye view
        '''

        image = np.array(image)
        background_image = np.array(background_image)

        # set up for projection
        warped_image = np.zeros_like(image).astype(float)
        warped_image = np.stack([warped_image] * 2, 0)
        pix_locations = self.xs - disp_map

        # find where occlusions are, and remove from disparity map
        mask = self.get_occlusion_mask(pix_locations)
        masked_pix_locations = pix_locations * mask - self.process_width * (1 - mask)

        # do projection - linear interpolate up to 1 pixel away
        weights = np.ones((2, self.feed_height, self.process_width)) * 10000

        for col in range(self.process_width - 1, -1, -1):
            loc = masked_pix_locations[:, col]
            loc_up = np.ceil(loc).astype(int)
            loc_down = np.floor(loc).astype(int)
            weight_up = loc_up - loc
            weight_down = 1 - weight_up

            mask = loc_up >= 0
            mask[mask] = \
                weights[0, np.arange(self.feed_height)[mask], loc_up[mask]] > weight_up[mask]
            weights[0, np.arange(self.feed_height)[mask], loc_up[mask]] = \
                weight_up[mask]
            warped_image[0, np.arange(self.feed_height)[mask], loc_up[mask]] = \
                image[:, col][mask] / 255.

            mask = loc_down >= 0
            mask[mask] = \
                weights[1, np.arange(self.feed_height)[mask], loc_down[mask]] > weight_down[mask]
            weights[1, np.arange(self.feed_height)[mask], loc_down[mask]] = weight_down[mask]
            warped_image[1, np.arange(self.feed_height)[mask], loc_down[mask]] = \
                image[:, col][mask] / 255.

        weights /= weights.sum(0, keepdims=True) + 1e-7  # normalise
        weights = np.expand_dims(weights, -1)
        warped_image = warped_image[0] * weights[1] + warped_image[1] * weights[0]
        warped_image *= 255.

        # now fill occluded regions with random background
        if not self.disable_background:
            warped_image[warped_image.max(-1) == 0] = background_image[warped_image.max(-1) == 0]

        warped_image = warped_image.astype(np.uint8)

        return warped_image

    def augment_synthetic_image(self, image):

        if self.disable_synthetic_augmentation:

            return np.array(Image.fromarray(image.astype(np.uint8)))

        # add some noise to stereo image
        noise = np.random.randn(self.feed_height, self.process_width, 3) / 50
        image = np.clip(image / 255 + noise, 0, 1) * 255

        # add blurring
        if random.random() > 0.5:
            image = gaussian(image,
                             sigma=random.random(),
                             multichannel=True)

        image = np.clip(image, 0, 255)

        # color augmentation
        image = transforms.ColorJitter(
            self.stereo_brightness, self.stereo_contrast, self.stereo_saturation,
            self.stereo_hue)(Image.fromarray(image.astype(np.uint8)))
        
        return np.array(image)

    def deep_mono_disparity(self, imgL):
        mono_disp = None
        return mono_disp

    def simple_SGM_disparity(self, imgL, imgR, window_size=1, 
                            min_disp=40, 
                            new_max_disparity=50,
                            blockSize=13,
                            speckleWindowSize=0,
                            speckleRange=2):
        # disparity range is tuned for 'aloe' image pair
        window_size = 1
        min_disp = min_disp
        num_disp = new_max_disparity-min_disp
        stereo = cv2.StereoSGBM_create(minDisparity = min_disp,
            numDisparities = num_disp,
            blockSize = blockSize,
            P1 = 8*3*window_size**2,
            P2 = 32*3*window_size**2,
            disp12MaxDiff = 1,
            uniquenessRatio = 10,
            speckleWindowSize = speckleWindowSize,
            speckleRange = speckleRange
        )

        print('computing disparity...')
        disp = stereo.compute(imgL, imgR).astype(np.float32) / 16.0

        plt.imshow( (disp-min_disp)/num_disp,'gray')
        plt.show()

        # normalize the color map using customized disparity threshold, defined based on num_disp
        colored_disp = self.process_disparity(disp)
        plt.imshow(colored_disp)
        plt.show()

        print('Done')
        return disp

    def disparity_to_depth(self, disp):
        print('generating real (guessed) depth map and 3d point cloud...',)
        h, w = imgL.shape[:2]
        # guess for focal length can be changed
        f = 0.8*w                         
        Q = np.float32([[1, 0, 0, -0.5*w],
                        [0,-1, 0,  0.5*h], # turn points 180 deg around x-axis,
                        [0, 0, 0,     -f], # so that y-axis looks up
                        [0, 0, 1,      0]])
        points = cv2.reprojectImageTo3D(disp, Q)
        colors = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)
        mask = disp > disp.min()
        out_points = points[mask]
        out_colors = colors[mask]
        depth_map = imgL[mask]
        #out_fn = 'out.ply'
        #write_ply(out_fn, out_points, out_colors)
        #print('%s saved' % out_fn)

    def convert_stereo(self, left_pth, bg_pth, disp_pth, preprocessor=None, crop=False):

        inputs = {}

        left_image, background_image = self.load_images(left_pth, bg_pth)
        loaded_disparity = self.load_disparity(disp_pth)

        if not self.crop_images:
            self.feed_height = np.array(left_image).shape[0]
            self.feed_width = np.array(left_image).shape[1]
             # do image generation for a wider image so we can crop off missing pixels
            self.process_width = self.feed_width
            self.xs, self.ys = np.meshgrid(np.arange(self.process_width), np.arange(self.feed_height))

        if preprocessor is not None:
            # pre-process the L R images, to denoise and white-balance
            left_image =  preprocessor.simple_preprocess(np.array(left_image))
            left_image = Image.fromarray(left_image.astype('uint8'), 'RGB')



        inputs['left_image'] = left_image
        inputs['background'] = background_image
        inputs['loaded_disparity'] = loaded_disparity


        # resize and/or crop
        inputs = self.prepare_sizes(inputs)

        # match color in background image
        inputs['background'] = transfer_color(np.array(inputs['background']),
                                              np.array(inputs['left_image']))

        # convert scaleless disparity to pixel disparity
        inputs['disparity'] = \
            self.process_disparity(inputs['loaded_disparity'])

        # now generate synthetic stereo image
        projection_disparity = inputs['disparity']
        right_image = self.project_image(inputs['left_image'],
                                         projection_disparity, inputs['background'])

        # augmentation
        right_image = self.augment_synthetic_image(right_image)

        # only keep required keys and prepare for network
        inputs = {'image': inputs['left_image'],
                  'right_image': right_image,
                  'disparity': projection_disparity.astype(float),
                  'mono_disparity': inputs['loaded_disparity'].astype(float),
                  }

        # finally crop to feed width
        if crop:
            for key in ['image', 'right_image']:
                try:
                    inputs[key] = inputs[key].crop((0, 0, self.feed_width, self.feed_height))
                except:
                    inputs[key] = inputs[key][:, :self.feed_width]
            for key in ['disparity', 'mono_disparity']:
                inputs[key] = inputs[key][:, :self.feed_width]

        return inputs

