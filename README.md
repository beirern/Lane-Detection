# This is the Annotation pipeline for detecting cars, people, and lanes.

## How to use image_detector.py:

### Required Packages:
* argparse (Default)
* cv2 (Install Required)
* numpy as np (Install Required)
* colorama (Might Require Install)

### Command Line Options
Running the script with no command line options will run all options that say 'Default True'. Outputs contour detection on the   
default image (lane-test.jpg) with canny edge detection and Otsu's Tresholding.

Usage:   
`image_detector.py [-h] [-image IMAGE PATH] [-blob] [-adaptive] 
                        [--show-thresh] [-contour] [--bounding-polygon]
                        [--convex-hull] [-color] [-canny] [-all]`

Options: 
* -h, --help
  * shows help message and exits
* -image IMAGE
  * Putting the name of an image directly after -image will change the image to run detection on
* -color
  * Runs the Detection With Color Thresholding for White and Yellow
* -canny
  * Runs the Detection With Canny Edge Detection for White and Yellow (Default True)
* -adaptive
  * Changes thresholding method to Adaptive (default is otsu)
* --show-thresh
  * Outputs the thresholded image using cv2.imshow
* -blob
  * Output the result of blob detection
* -contour
  * Output the result of contour detection (Default True)
* --bounding-polygon
  * Outputs the reuslt of the boulding polygon method
* --convex-hull
  * Output the result of the Convex Hull Method
* -all
  * Run Contour, Blob, Convex Hull, and Bounding Polygon Detection. Does not change any Thresholding Values.


Note:   
* Order of options does not matter.  
* Only one thresholding method can be used at a time (Adaptive or Otsu).  
* No duplicate images will be shown.

Examples:
* `python ./image_detector.py` will output the default image (lane-test.jpg) using the default  
method (Otsu's thresholding) and the default method (Contour Detection)
* `python ./image_detector.py -image lane-test.jpg` will output Contour Detection with Otsu's Thresholding  
from the image lane-test.jpg
* `python ./image_detector.py -blob --bouding-polygon --show-thresh` will output Blob Detection, Bouding Polygon  
method, and show all thresholding images using Otsu's
* `python ./image_detector.py -blob --bouding-polygon --show-thresh -adaptive` will output Blob Detection,   
Bouding Polygon method, and show all thresholding images using Adaptive Tresholding

Enjoy!