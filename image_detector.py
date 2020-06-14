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
kernel_size = (5, 5)
# Canny Thresholds
canny_low_thresh = 50
canny_high_thresh = 150

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
# DO NOT USE
def color_thresholding(gray_img, hsv_img, show_thresh):
    pass


def canny(blur_img, show_thresh):
    edges = cv2.Canny(blur_img, canny_low_thresh, canny_high_thresh)
    if show_thresh:
        cv2.imshow("Canny Detection", edges)

    img = region_of_interest(edges)
    if show_thresh:
        cv2.imshow("Canny Detection w/ Region of Interest", img)

    return img


def process_image(gray_img, options, imgs, show_thresh):
    processed_imgs = []

    threshed_img = threshold(options[0], gray_img, show_thresh)

    # Color thresholding is true
    if options[1]:
        processed_imgs.append(color_thresholding(imgs[0][0], imgs[0][1], show_thresh))
    # Canny Edge Detection is True
    if options[2]:
        processed_imgs.append(canny(imgs[1], show_thresh))

    # Bitwise And all Images
    for img in processed_imgs:
        threshed_img = cv2.bitwise_and(threshed_img, img)

    # Morphology is True
    if options[3]:
        threshed_img = morphology(threshed_img)

    return threshed_img


def threshold(method, gray_img, show_thresh):
    blur = cv2.GaussianBlur(gray_img, kernel_size, 0)
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

    return threshed_img


# Region of Interest
def region_of_interest(img):
    rows, cols = img.shape[:2]
    mask = np.zeros_like(img)

    left_bottom = [cols * 0.1, rows]
    right_bottom = [cols * 0.95, rows]
    left_top = [cols * 0.4, rows * 0.6]
    right_top = [cols * 0.6, rows * 0.6]

    vertices = np.array(
        [[left_bottom, left_top, right_top, right_bottom]], dtype=np.int32
    )

    if len(mask.shape) == 2:
        cv2.fillPoly(mask, vertices, 255)
    else:
        cv2.fillPoly(mask, vertices, (255,) * mask.shape[2])
    return cv2.bitwise_and(img, mask)


# Various morphing moethods to remove noise
def morphology(img):
    if show_thresh:
        cv2.imshow(method + " Thresholding with Color/Canny" + " Pre-Morphology", img)
    kernel = np.ones(kernel_size, np.uint8)
    morph_img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    if show_thresh:
        cv2.imshow(method + " Thresholding with Color/Canny" + " Morphology", morph_img)
    return morph_img


def contours_detection(processed_img):
    # Finds contours and get the external one
    # Basically Forms an outline of image
    contours, hier = cv2.findContours(
        processed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    image = cv2.pyrDown(cv2.imread(filename, cv2.IMREAD_UNCHANGED))
    # Using only contours to fill in Objects
    cv2.drawContours(image, contours, -1, green, thickness=cv2.FILLED)

    return [image, contours]


def bouding_polygons(img, filename):
    bounding_polygon = cv2.pyrDown(cv2.imread(filename, cv2.IMREAD_UNCHANGED))
    contours = contours_detection(img)[1]

    for cnt in contours:
        # calculate epsilon base on contour's perimeter
        # contour's perimeter is returned by cv2.arcLength
        epsilon = 0.01 * cv2.arcLength(cnt, True)
        # get approx polygons
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        # draw approx polygons
        perimeter = cv2.arcLength(cnt, True)
        x, y, w, h = cv2.boundingRect(cnt)
        if perimeter > 35 and w > 12 and h > 10:
            cv2.drawContours(
                bounding_polygon, [approx], -1, green, thickness=cv2.FILLED
            )

    return bounding_polygon


def convex_hull(img, filename):
    # Convex Hull ONLY Otsu's Thresholding
    # Hull is Filled
    bounding_polygon = cv2.pyrDown(cv2.imread(filename, cv2.IMREAD_UNCHANGED))
    contours = contours_detection(img)[1]
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
        "-otsu",
        action="store_true",
        default=False,
        help="Whether to Use Otsu Thresholding instead of Adaptive",
    )
    parser.add_argument(
        "--show-thresh",
        action="store_true",
        default=False,
        help="Whether to Show Thresholding Images",
    )
    parser.add_argument(
        "-contour",
        action="store_true",
        default=False,
        help="Outputs the result of Contour Detection (Default True)",
    )
    parser.add_argument(
        "--bounding-polygon",
        action="store_false",
        default=True,
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
        help="Runs the Detection With Color Thresholding for White and Yellow. (CURRENTLY UNIMPLEMENTED)",
    )
    parser.add_argument(
        "-morph",
        action="store_false",
        default=True,
        help="Run the Morphology Methods (Currently only Close) (Default True)",
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
        filename = args["Image Path"]
    else:
        # Default of no command line arguement
        filename = "dashcam_clear_conditions_picture.jpg"
    # Show Threshold option
    show_thresh = args["show_thresh"]
    if args["otsu"]:
        method = "Otsu"
    else:
        method = "Adaptive"
    # Make images that are needed
    original_image = cv2.pyrDown(cv2.imread(filename, cv2.IMREAD_UNCHANGED))
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    hsv_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)
    cv2.imshow("Original Image", original_image)

    # Get image after all thresholding and image detection
    image = process_image(
        gray_image,
        (method, args["color"], args["canny"], args["morph"]),
        ((gray_image, hsv_image), cv2.GaussianBlur(gray_image, kernel_size, 0)),
        show_thresh,
    )

    check_mark = u"\u2713"
    cross_mark = u"\u2717"

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
    message = "    Morphology? "
    if args["morph"]:
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
        blob_detection(gray_image)
        print(Fore.GREEN + message + " " + Fore.LIGHTMAGENTA_EX + check_mark)
    else:
        print(Fore.GREEN + message + " " + Fore.WHITE + cross_mark)

    # Detection Methods
    message = "    Contour Detection"
    if args["contour"]:
        image = contours_detection(image)[0]
        cv2.imshow("Contours w/ " + method + " Thresholding", image)
        image = process_image(
            gray_image,
            (method, args["color"], args["canny"], args["morph"]),
            ((gray_image, hsv_image), cv2.GaussianBlur(gray_image, kernel_size, 0)),
            show_thresh,
        )
        print(Fore.GREEN + message + " " + Fore.LIGHTMAGENTA_EX + check_mark)
    else:
        print(Fore.GREEN + message + " " + Fore.WHITE + cross_mark)

    message = "    Bouding Polygon Method"
    if args["bounding_polygon"]:
        image = bouding_polygons(image, filename)
        cv2.imshow("Bounding Box " + method + " Thresholding", image)
        image = process_image(
            gray_image,
            (method, args["color"], args["canny"], args["morph"]),
            ((gray_image, hsv_image), cv2.GaussianBlur(gray_image, kernel_size, 0)),
            show_thresh,
        )
        print(Fore.GREEN + message + " " + Fore.LIGHTMAGENTA_EX + check_mark)
    else:
        print(Fore.GREEN + message + " " + Fore.WHITE + cross_mark)

    message = "    Convex Hull Method"
    if args["convex_hull"]:
        image = convex_hull(image, filename)
        cv2.imshow("Convex Hull Otsu Thresholding", image)
        image = process_image(
            gray_image,
            (method, args["color"], args["canny"], args["morph"]),
            ((gray_image, hsv_image), cv2.GaussianBlur(gray_image, kernel_size, 0)),
            show_thresh,
        )
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
