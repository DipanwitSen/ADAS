About package :
cv2: OpenCV library for image processing.
numpy: Library for numerical operations.
os: Provides a way to use operating system-dependent functionality like file paths.
optimize: From SciPy, used for optimization algorithms.
pyplot, cm, colors: From Matplotlib for plotting and color management.

About variable
ym_per_pix: Represents meters per pixel in the y-dimension. It is assumed that the visible portion of the lane is about 30 meters long, and the image height is 720 pixels.
xm_per_pix: Represents meters per pixel in the x-dimension. It is assumed that the lane width is about 3.7 meters, and the image width is 720 pixels.

A pixel, short for "picture element," is the smallest unit of a digital image or display. Pixels are the building blocks of digital images, and each pixel represents a single point in the image. 
The resolution of an image is defined by the number of pixels along its width and height. For example, an image with a resolution of 1920x1080 pixels contains 1920 pixels in width and 1080 pixels in height.

hls = cv2.cvtColor(inpImage, cv2.COLOR_BGR2HLS)
HLS: Stands for Hue, Lightness, and Saturation. It is a color space that can make it easier to isolate certain colors (like lane lines) than the standard RGB color space.

lower_white and upper_white: Define the range of white colors in HLS space. These values can be adjusted based on lighting conditions and lane line color.
mask: A binary mask where the white lane lines are isolated.
hls_result: The result of applying the mask to the original image to highlight the white areas.

blur: Applies a Gaussian blur to smooth the image and reduce noise. The (3, 3) is the kernel size.
canny: Applies the Canny edge detector to find edges in the blurred image. The parameters 40 and 60 are the lower and upper thresholds for the hysteresis procedure in edge detection.

Why to use bird eye view ?
Perspective Removal,Parallel Lanes,Consistent Width etc are reason
img_size: The size of the image.
src: Source points in the original image for perspective transformation.
dst: Destination points in the bird's eye view.
matrix: The transformation matrix for converting the perspective.
minv: The inverse transformation matrix.
birdseye: The warped (transformed) image.
birdseyeLeft and birdseyeRight: The left and right halves of the bird's eye view image, useful for processing lanes separately.

inpImage: The input image for which the bird's eye view is to be calculated.
inpImage.shape[1]: The width of the input image.
inpImage.shape[0]: The height of the input image.
img_size: A tuple representing the size of the image in the form (width, height).



