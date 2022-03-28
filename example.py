from generator import *

# define the left eye view, backdground, disparity map paths
left_pth = 'testsets/diode/00019_00183_indoors_000_010.png'
bg_pth = 'testsets/randoms/000002.jpg'
disp_pth = 'testsets/diode/00019_00183_indoors_000_010.npy'




# define a pre-processing pipeline, to denoise and white-balance and other filter
# not a necessary step, but just in case the image quality is bad, that SGM will not do feature matching well
preprocessor = PreProcessor(denoise=False,
                            white_balance=True,
                            blur_block=(1,1))


# init the class for stereo image generation
'''
 !!! if you wanna generate bad quality disparity map, change these parameters:
 1. increase the max_disparity_range lower bound value, when it is > 80, e.g.max_disparity_range=(80, 190), the quality gets bad (because the are more holes in generated right eye view)
 2. make fix_disparity_scale True (so that will always generate high disparity views)
 3. modify the hyper-paramters of SGM in function st_converter.convert_stereo(), different prefiltering and post-filtering will affect the quality of disparity estimation
 4. augmenting the synthesized right eye view, to add noise to the right eye imgae, then disparity estimation may perform bad: disable_synthetic_augmentation=False
 '''
st_converter = StereoConverter(feed_height=480,
                             feed_width=640,
                             max_disparity=100,
                             fix_disparity_scale=True,
                             max_disparity_range=(50, 100),
                             disable_synthetic_augmentation=True,
                             crop_images=False,
                             )

# returns a set of images, the code and algorithm are inspired by 
'''
Watson, Jamie, et al. "Learning stereo from single images." European Conference on Computer Vision. Springer, Cham, 2020.
'''
outs = st_converter.convert_stereo(left_pth, bg_pth, disp_pth, preprocessor=preprocessor)


# plot the images, 'right_image' is the synthesized right eye view based on projection model and disparity map
print('left eye view')
plt.imshow(outs['image'], interpolation='nearest')
plt.show()

print('right eye view')
plt.imshow(outs['right_image'], interpolation='nearest')
plt.show()

print('pre-defined disparity map')
plt.imshow(outs['disparity'], interpolation='nearest')
plt.show()

left_image, right_image = np.array(outs['image']), outs['right_image']

# use opencv implementation to estimate disparity
L = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
R = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)

# playing the paramters of SGM to check results
synthesized_disp = st_converter.simple_SGM_disparity(left_image, right_image,  
                                                    window_size=1, 
                                                    min_disp=0, 
                                                    new_max_disparity=40,
                                                    blockSize=3,
                                                    speckleWindowSize=50,
                                                    speckleRange=2)


# use a third party library to estimate disparity
# use original SGM method to calculate adaptive disparity, better but more time consuming
# better to transfer to grascale for SGBM algorithm
left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)
LD, RD = sgm(left_image, right_image, max_disp=192)

plt.imshow(LD, interpolation='nearest')
plt.show()

plt.imshow(RD, interpolation='nearest')
plt.show()
