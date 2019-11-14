import cv2
import numpy as np

filename = "darker_conditions.mp4"
video = cv2.VideoCapture(filename)

# Variables to make colors simpler
green = (0, 255, 0)
red = (0, 0, 255)
kernel_size = (5, 5)
# Canny Thresholds
canny_low_thresh = 50
canny_high_thresh = 150


def canny(blur_img):
    edges = cv2.Canny(blur_img, canny_low_thresh, canny_high_thresh)

    img = region_of_interest(edges)

    return img


def threshold(gray_img):
    blur = cv2.GaussianBlur(gray_img, kernel_size, 0)
    # Adaptive Thresholding
    # Advanced Thresholding for Contours and Bounding Polygons
    threshed_img = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    return threshed_img


# Region of Interest
def region_of_interest(canny_img):
    rows, cols = canny_img.shape[:2]
    mask = np.zeros_like(canny_img)

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
    return cv2.bitwise_and(canny_img, mask)


def morphology(img):
    kernel = np.ones(kernel_size, np.uint8)
    morph_img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    return morph_img
    

def bounding_polygons(processed_img, frame):
    bounding_polygon = frame
    contours, hier = cv2.findContours(
        processed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

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


while video.isOpened():
    ret, frame = video.read()
    frame = cv2.pyrDown(frame)

    # Make images that are needed
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    cv2.imshow("Original Video", frame)

    canny_img = canny(cv2.GaussianBlur(frame, kernel_size, 0))
    threshed_img = threshold(gray_image)

    final_img = cv2.bitwise_and(canny_img, threshed_img)
    final_img = morphology(final_img)
    final_img = bounding_polygons(final_img, frame)

    cv2.imshow("Final", final_img)

    if cv2.waitKey(35) == 27:
        break

video.release()
cv2.destroyAllWindows()
