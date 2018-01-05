# retarded-PS 

by Chuzhe Tang

This is a course project, a utterly retarded digital image processing tool for SE342 Computer Vision @ SJTU.

The project includes two separate module, `image` algorithm implementation package and `App` a GUI application that provides *easy* access to algorithms.

Time is limited. Human mind is stupid. Project requirements are obscured. Let's appreciate what a student have done in just days before the deadline. 

## How to use

Copy image folder to your **`python 3`** project and add `from image import *`.

Or you can run GUI app with `python3 main.py` in command line.

## What's inside

`image` package provides following *naive* algorithm implementation:

1. Basic operations
- Color removal
- Channel extraction (RGB)
- Adjust hue / saturation / lightness (HSL)
- Rotation, Crop and resize
- Various ways to map grey values
2. Binarization
- Otus
- Arbitrary threshold
3. Filtering
- Gaussian / median / mean filter
- Convolution with user-specified core
4. Edge detection
- Sobel / Laplace operator
- Canny edge detection
5. Morphology
- Erosion, dilation, opening, closing
- Morphological reconstruction
- Watershed
- Thinning and thickening (binary image only)
- Skeleton and skeleton reconstruction (binary image only)
- Distance transformation (binary image only)
6. Others
- Algebra operations (addition, subtraction, multiplication)

---

To better relieve fellow SE students, patches and updates are always welcomed. 