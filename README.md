
# mono2SGM
 Generating SGM calculated disparity estimation from monocular RGB image.  This project provides a python based library which generates stereo disparity map from a single image input. 

This project is highly based on "Watson, Jamie, et al. "Learning stereo from single images." _European Conference on Computer Vision_. Springer, Cham, 2020". Refer to it for more mathmatical details. 


# Pipeline
![The synthetic stereo disparity generation pipeline](figs/pipeline.jpg?raw=true "Title")

## Inputs
The stereo generator assumes the following data are given:
-  1 RGB image as the left eye view (can be right eye as well, does not matter)
- 1 corresponding predicted disparity map (any monocular depth estimation is appliable). 
-  The depth estimation module to be added here...

## Usage
- Refer to example.py for more detailed examples. Please check the comments in example.py
```
python example.py
```
- Feel free to change the image path using the images in folder "testsets"
[a link](https://github.com/uenian33/mono2SGM/tree/main/testsets)

## Results
Using OpenCV provided SGBM (no post-processing), we can synthesize a rough disparity:
![The simple OpenCV syntheziation](figs/results.jpg?raw=true "Title")

Using a 3rd party  SGM (with post-processing), we can synthesize  relatively well disparity maps, but very time-consuming:
![The 3rd party SGM disparity calculation results](figs/3rd_reults.jpg?raw=true "Title")
## Parameters

If bad quality disparity map generation map is required, test these parameters:
 1. increase the max_disparity_range lower bound value, when it is > 80, e.g.max_disparity_range=(80, 190), the quality gets bad (because the are more holes in generated right eye view)
 2. make fix_disparity_scale True (so that will always generate high disparity views)
 3. modify the hyper-paramters of SGM in function st_converter.convert_stereo(), different prefiltering and post-filtering will affect the quality of disparity estimation
 4. augmenting the synthesized right eye view, to add noise to the right eye imgae, then disparity estimation may perform bad: disable_synthetic_augmentation=False
 
 # To-do
- [ ] Add monocular depth/disparity estimation modules


# References
- The right eye view generation was inspired by work https://github.com/nianticlabs/stereo-from-mono
- The right eye view image generation code is based on https://github.com/nianticlabs/stereo-from-mono
- The SGBM method is implemented using OpenCV library https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html
- The 3rd party SGBM method is based on https://github.com/beaupreda/semi-global-matching
- The image white balance model is based on  https://github.com/mahmoudnafifi/WB_sRGB


# Citatons
If you use this projection, please cite the references mentioned above, and this repository:
```
@misc{github,
  author={github},
  title={mono2SGM},
  year={2022},
  url={https://github.com/uenian33/mono2SGM},
}
```
