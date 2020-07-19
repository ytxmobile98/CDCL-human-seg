import cv2
import imutils
import numpy as np
from imutils import contours
from skimage import measure


def circleInflamArea(input_image_path, output_image_path, lower_inflam_temp,
                     upper_inflam_temp, grey_threshold=50, area_threshold=500):
    """
    input_image is the path of the segmented lung thermal image
    Example: "folder/input_image.jpg"

    output_image is the path of the lung thermal image with
    inflammation area circled
    Example: "folder/output_image.jpg"

    lower_inflam_temp is the value BGR channel that represent the minimum
    temperature that is considered as inflammation
    Format:
        list of 3 integer corresponding to blue, green, red channel
        range of integer must be in [0, 255]
    Example: [0, 0, 128]

    upper_inflam_temp is the value BGR channel that represent the maximum
    temperature that is considered as inflammation
    Format:
        list of 3 integer corresponding to blue, green, red channel
        range of integer must be in [0, 255]
    Example: [190, 190, 255]

    grey_threshold is the threshold of grey channel value used to filter
    potential noise of the inflammation area selection
    Example: 50

    area_threshold is the threshold of pixel number required for an area
    to be considered as inflammation area
    Example: 300
    """

    inflam_lung = cv2.imread(input_image_path)

    lower = np.uint8(lower_inflam_temp)
    upper = np.uint8(upper_inflam_temp)
    inflam_mask = cv2.inRange(inflam_lung, lower, upper)
    cropped = cv2.bitwise_and(inflam_lung, inflam_lung, mask=inflam_mask)

    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)

    thresh = cv2.threshold(blurred, grey_threshold, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=4)

    labels = measure.label(thresh, connectivity=2, background=0)
    mask = np.zeros(thresh.shape, dtype="uint8")

    for label in np.unique(labels):
        if label == 0:
            continue

        labelMask = np.zeros(thresh.shape, dtype="uint8")
        labelMask[labels == label] = 255
        numPixels = cv2.countNonZero(labelMask)

        if numPixels > area_threshold:
            mask = cv2.add(mask, labelMask)

    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = contours.sort_contours(cnts)[0]

    for c in cnts:
        ((cX, cY), radius) = cv2.minEnclosingCircle(c)
        cv2.circle(inflam_lung, (int(cX), int(cY)), int(radius) + 10, (0, 0, 255), 2)

    out_name = input_image_path.split('/')[-1].split('.')[0] + '_circled'
    cv2.imwrite(output_image_path + out_name + '.png', inflam_lung)
