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

img = np.load('dataset/diode/midas_depths/val/indoors/scene_00019/scan_00183/00019_00183_indoors_000_010.npy', allow_pickle=True)

plt.imshow(img)
plt.show()