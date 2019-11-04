import argparse
import cv2
import numpy as np
from colorama import Fore

# Intro
print(Fore.BLUE + "Please check the README or pass in -h to see how to use options!")
print(Fore.RED + "Press ESC to Close all Windows")

# Variables to make colors simpler
green = (0, 255, 0)
red = (0, 0, 255)
# Masks for yellow and white
lower_yellow_mask = [20, 100, 100]
upper_yellow_mask = [30, 255, 255]
lower_white_mask = 200
upper_white_mask = 250
# Canny Thresholds
canny_low_thresh = 50
canny_high_thresh = 150
# Hough Values
rho = 6
theta = np.pi / 60
threshold_value = 160
minLineLength = 40
maxLineGap = 25

# Blob Detection, Not good. Even when messing with parameters could not detect lanes
def blob_detection(gray_img):
    # Creation and Detection
    detector = cv2.SimpleBlobDetector_create()

    keypoints = detector.detect(gray_img)

    im_with_keypoints = cv2.drawKeypoints(
        gray_img,
        keypoints,
        np.array([]),
        red,
        cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )

    cv2.imshow("Blob Detection", im_with_keypoints)


# Thresholds image to only show white and yellow objects
def color_thresholding(gray_img, hsv_img, show_thresh):
    # Creates lower and upper bounds for the yellow mask
    lower_yellow = np.array(lower_yellow_mask, dtype="uint8")
    upper_yellow = np.array(upper_yellow_mask, dtype="uint8")

    # Creates yellow and white masks over the image to isolate the colors
    mask_yellow = cv2.inRange(hsv_img, lower_yellow, upper_yellow)
    mask_white = cv2.inRange(gray_img, lower_white_mask, upper_white_mask)
    mask_yw = cv2.bitwise_or(mask_white, mask_yellow)
    mask_yw_image = cv2.bitwise_and(gray_img, mask_yw)
    if show_thresh:
        cv2.imshow("Color", mask_yw_image)
    return mask_yw_image


def canny(blur_img, show_thresh):
    edges = cv2.Canny(blur_img, canny_low_thresh, canny_high_thresh)
    if show_thresh:
        cv2.imshow("Canny Detection", edges)
    return edges


def get_image(original, color_img, canny_img, color_bool, canny_bool, show_thresh):
    image_canny, image_color = None, None
    if canny_bool:
        image_canny = canny(canny_img, show_thresh)
    if color_bool:
        image_color = color_thresholding(color_img[0], color_img[1], show_thresh)
    
    if canny_bool and color_bool:
        return cv2.bitwise_and(image_canny, image_color)
    elif canny_bool:
        return image_canny
    elif color_bool:
        return image_color
    else:
        return original


def threshold(method, gray_img, show_thresh):
    blur = cv2.GaussianBlur(gray_img, (5, 5), 0)
    if method == "Adaptive":
        # Adaptive Thresholding
        # Advanced Thresholding for Contours and Bounding Polygons
        threshed_img = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
    else:
        # Otsu's thresholding after Gaussian filtering
        # Different Thresholding Technique for Contours and Bounding Polygons
        ret, threshed_img = cv2.threshold(
            blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        if show_thresh:
            cv2.imshow(method + " Thresholding", threshed_img)
    # Further threshold by taking out the sky/upper half of image
    threshed_img = region_of_interest(threshed_img)
    if show_thresh:
        cv2.imshow(method + " Thresholding w/ Region of Interest", threshed_img)

    return [method, threshed_img]


# Region of Interest
def region_of_interest(image):
    # Get Verticies of Image
    rows, cols = image.shape[:2]
    bottom_left = [0, rows]
    top_left = [0, rows * 0.5]
    bottom_right = [cols, rows]
    top_right = [cols, rows * 0.5]
    verticies = np.array(
        [[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32
    )

    # Make a mask to not detect top half of image
    mask = np.zeros_like(image)

    if len(image.shape) == 2:
        cv2.fillPoly(mask, verticies, 255)
    else:
        cv2.fillPoly(mask, verticies, (255,) * mask.shape[2])

    return cv2.bitwise_and(image, mask)


def contours_detection(method, threshed_img):
    # Finds contours and get the external one
    # Basically Forms an outline of image
    contours, hier = cv2.findContours(
        threshed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    image = cv2.pyrDown(cv2.imread(filename, cv2.IMREAD_UNCHANGED))
    # Using only contours to fill in Objects
    cv2.drawContours(image, contours, -1, green, thickness=cv2.FILLED)

    return [image, contours]


def bouding_polygons(method, image, filename):
    bounding_polygon = cv2.pyrDown(cv2.imread(filename, cv2.IMREAD_UNCHANGED))
    contours = contours_detection(method, image)[1]

    for cnt in contours:
        # calculate epsilon base on contour's perimeter
        # contour's perimeter is returned by cv2.arcLength
        epsilon = 0.01 * cv2.arcLength(cnt, True)
        # get approx polygons
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        # draw approx polygons
        cv2.drawContours(bounding_polygon, [approx], -1, green, thickness=cv2.FILLED)

    return bounding_polygon


def convex_hull(method, image, filename):
    # Convex Hull ONLY Otsu's Thresholding
    # Hull is Filled
    bounding_polygon = cv2.pyrDown(cv2.imread(filename, cv2.IMREAD_UNCHANGED))
    contours = contours_detection(method, image)[1]
    for cnt in contours:
        # get convex hull
        hull = cv2.convexHull(cnt)
        # draw it in red color
        cv2.drawContours(bounding_polygon, [hull], -1, red, thickness=cv2.FILLED)

    return bounding_polygon


if __name__ == "__main__":
    # ArgParse
    print(Fore.WHITE)
    parser = argparse.ArgumentParser(description="Detecting Lanes in various methods")
    parser.add_argument(
        "-image",
        dest="Image Path",
        default="False",
        help="What image to run detection on",
    )
    parser.add_argument(
        "-blob",
        action="store_true",
        default=False,
        help="Outputs the Result of Blob Detection",
    )
    parser.add_argument(
        "-adaptive",
        action="store_true",
        default=False,
        help="Whether to Use Adaptive Thresholding instead of Otsu's",
    )
    parser.add_argument(
        "--show-thresh",
        action="store_true",
        default=False,
        help="Whether to Show Thresholding Images",
    )
    parser.add_argument(
        "-contour",
        action="store_false",
        default=True,
        help="Outputs the result of Contour Detection (Default True)",
    )
    parser.add_argument(
        "--bounding-polygon",
        action="store_true",
        default=False,
        help="Outputs the result of Using a Bounding Polygon",
    )
    parser.add_argument(
        "--convex-hull",
        action="store_true",
        default=False,
        help="Outputs the Result of the Convex Hull Method",
    )
    parser.add_argument(
        "-color",
        action="store_true",
        default=False,
        help="Runs the Detection With Color Thresholding for White and Yellow",
    )
    parser.add_argument(
        "-canny",
        action="store_false",
        default=True,
        help="Runs the Detection With Canny Edge Detection for White and Yellow (Default True)",
    )
    parser.add_argument(
        "-all",
        action="store_true",
        default=False,
        help="Run Contour, Blob, Convex Hull, and Bounding Polygon Detection. Does not change any Thresholding Values.",
    )
    args = vars(parser.parse_args())
    # read image using command line arguement
    if args["Image Path"] != "False":
        filename = args["Image"]
    else:
        filename = "lane-test.jpg"  # Default of no command line arguement
    # Show Threshold option
    show_thresh = args["show_thresh"]
    # Make images that are needed
    img = cv2.pyrDown(cv2.imread(filename, cv2.IMREAD_UNCHANGED))
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    cv2.imshow("Original Image", img)

    # Different Methods based off of command line arguements

    # Thresholding (Otsu Default)

    # Get the Threshold method passed in
    def get_thresh_method():
        if args["adaptive"]:
            method, image = threshold("Adaptive", gray_img, show_thresh)
        else:
            method, image = threshold("Otsu", gray_img, show_thresh)
        return [method, image]

    method, image = get_thresh_method()

    # Get image after Color and Canny Detection if Applicable
    image = cv2.bitwise_and(
        image,
        get_image(
            gray_img,
            [gray_img, hsv_img],
            cv2.GaussianBlur(gray_img, (5, 5), 0),
            args["color"],
            args["canny"],
            show_thresh,
        ),
    )

    if show_thresh:
        cv2.imshow(method + ' Thresholding with Color/Canny', image)
    
    check_mark = u"\u2713"
    cross_mark = u"\u2717"

    # Canny Edge Detection
    # if args["canny"]:
    #     canny_image = canny(cv2.GaussianBlur(gray_img, (5,5), 0))
    #     image = cv2.bitwise_and(image, canny_image)

    # Color Thresholding
    # if args["color"]:
    #     color_image = color_thresholding(gray_img, hsv_img)
    #     image = cv2.bitwise_and(color_image, image)

    # Message with what Images are loaded
    print(Fore.GREEN + "Showing Images...")
    print(Fore.GREEN + "Image: " + filename)
    print(Fore.GREEN + "Thresholding:")
    print(Fore.GREEN + "    Method: " + method)
    message = "    Show Thresholded Image? "
    if show_thresh:
        print(Fore.GREEN + message + Fore.LIGHTMAGENTA_EX + check_mark)
    else:
        print(Fore.GREEN + message + Fore.WHITE + cross_mark)
    message = "    Color Thresholding? "
    if args["color"]:
        print(Fore.GREEN + message + Fore.LIGHTMAGENTA_EX + check_mark)
    else:
        print(Fore.GREEN + message + Fore.WHITE + cross_mark)
    message = "    Canny Edge Detection? "
    if args["canny"]:
        print(Fore.GREEN + message + Fore.LIGHTMAGENTA_EX + check_mark)
    else:
        print(Fore.GREEN + message + Fore.WHITE + cross_mark)

    print(Fore.GREEN + "Methods:")

    # If '-all' is passed
    if args["all"]:
        args["blob"] = True
        args["contour"] = True
        args["bounding_polygon"] = True
        args["convex_hull"] = True

    # Blob Detection
    message = "    Blob Detection"
    if args["blob"]:
        blob_detection(gray_img)
        print(Fore.GREEN + message + " " + Fore.LIGHTMAGENTA_EX + check_mark)
    else:
        print(Fore.GREEN + message + " " + Fore.WHITE + cross_mark)

    # Detection Methods
    message = "    Contour Detection"
    if args["contour"]:
        image = contours_detection(method, image)[0]
        cv2.imshow("Contours w/ " + method + " Thresholding", image)
        method, thresh_image = get_thresh_method()
        image = get_image(
            gray_img,
            [gray_img, hsv_img],
            cv2.GaussianBlur(gray_img, (5, 5), 0),
            args["color"],
            args["canny"],
            show_thresh,
        )
        image = cv2.bitwise_and(thresh_image, image)
        print(Fore.GREEN + message + " " + Fore.LIGHTMAGENTA_EX + check_mark)
    else:
        print(Fore.GREEN + message + " " + Fore.WHITE + cross_mark)

    message = "    Bouding Polygon Method"
    if args["bounding_polygon"]:
        image = bouding_polygons(method, image, filename)
        cv2.imshow("Bounding Box " + method + " Thresholding", image)
        method, thresh_image = get_thresh_method()
        image = get_image(
            gray_img,
            [gray_img, hsv_img],
            cv2.GaussianBlur(gray_img, (5, 5), 0),
            args["color"],
            args["canny"],
            show_thresh,
        )
        image = cv2.bitwise_and(thresh_image, image)
        print(Fore.GREEN + message + " " + Fore.LIGHTMAGENTA_EX + check_mark)
    else:
        print(Fore.GREEN + message + " " + Fore.WHITE + cross_mark)

    message = "    Convex Hull Method"
    if args["convex_hull"]:
        image = convex_hull(method, image, filename)
        cv2.imshow("Convex Hull Otsu Thresholding", image)
        method, thresh_image = get_thresh_method()
        image = get_image(
            gray_img,
            [gray_img, hsv_img],
            cv2.GaussianBlur(gray_img, (5, 5), 0),
            args["color"],
            args["canny"],
            show_thresh,
        )
        image = cv2.bitwise_and(thresh_image, image)
        print(Fore.GREEN + message + " " + Fore.LIGHTMAGENTA_EX + check_mark)
    else:
        print(Fore.GREEN + message + " " + Fore.WHITE + cross_mark)

    # Press ESC to exit
    ESC = 27
    while True:
        keycode = cv2.waitKey()
        if keycode != -1:
            if keycode == ESC:
                break
    print(Fore.YELLOW + "Closing Windows...")
    cv2.destroyAllWindows()
