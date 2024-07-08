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

1. get_perspective_transform: This function takes two sets of points as input: src_points and dst_points. It returns two matrices: M and Minv.
M is the perspective transformation matrix that maps the source points to the destination points.
Minv is the inverse of M, which maps the destination points back to the source points.

3. warp_perspective: This function takes an image, a perspective transformation matrix M, and an image size img_size as input. It applies the perspective transformation to the image using the M matrix and returns the warped image.

4. canny: This function takes an image as input and returns a binary image (i.e., an image with only 0s and 255s) that represents the edges in the original image. The Canny edge detection algorithm is used to detect edges
cv2.cvtColor(image, cv2.COLOR_RGB2GRAY): Convert the input image from RGB to grayscale.
cv2.GaussianBlur(gray, (5, 5), 0): Apply a Gaussian blur to the grayscale image to reduce noise.
cv2.Canny(blur, 50, 150): Apply the Canny edge detection algorithm to the blurred image. The two parameters 50 and 150 are the lower and upper thresholds for edge detection, respectively.

6. region_of_interest: This function takes an image as input and returns a masked image that only shows the region of interest (ROI).
height = image.shape[0]: Get the height of the input image.
polygons = np.array([(200, height), (1100, height), (550, 250)]): Define a polygon that represents the ROI. In this case, it's a triangle with vertices at (200, height), (1100, height), and (550, 250).
mask = np.zeros_like(image): Create a mask image with the same shape as the input image, filled with zeros.
cv2.fillPoly(mask, polygons, (0, 255, 0)): Fill the polygon with a green color (0, 255, 0) in the mask image.
masked_image = cv2.bitwise_and(image, mask): Apply the mask to the original image using a bitwise AND operation. This will set all pixels outside the ROI to zero.

Hough Lines: Hough lines are a way to detect lines in an image. The Hough transform is a feature extraction technique that can be used to detect lines, circles, and other shapes in an image.
In the context of lane detection, Hough lines are often used to detect the lines that make up the lane markings on the road. The Hough transform can be used to detect lines in the image, and then these lines can be used to infer the position and orientation of the lane.

Grey: Grey refers to a grayscale image, which is an image that only has shades of grey, ranging from pure black (0) to pure white (255). Grayscale images are often used as an intermediate step in image processing algorithms, as they can be easier to work with than color images.

Canny: Canny refers to the Canny edge detection algorithm, which is a popular edge detection technique used in image processing. The Canny algorithm is used to detect edges in an image by applying two thresholds to the gradient of the image. The resulting binary image shows the edges in the original image.



