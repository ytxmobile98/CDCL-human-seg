import cv2
import numpy as np


def applyLungThermal(thermal_iamge_path, mask_image_path, output_image_path,
                     pts_thermal):
    """
    thermal_path is the path of the input thermal image
    Example: "folder/thermal.bmp"

    mask_path is the directory of the lung mask image
    Example: "folder/"

    pts_thermal is the key points in the order of left shoulder, throat,
    right shoulder
    Example: [[520, 370], [680, 320], [850, 370]]

    The output is a thermal image of the lung area
    """

    thermal = cv2.imread(thermal_iamge_path)
    mask = cv2.imread(mask_image_path)

    pts_mask = np.float32([[200, 400], [1400, 0], [2600, 400]])
    pts_thermal = np.float32(pts_thermal)

    h_mask, w_mask, _ = mask.shape
    h_thermal, w_thermal, _ = thermal.shape

    M = cv2.getAffineTransform(pts_mask, pts_thermal)
    trans_mask = cv2.warpAffine(mask, M, (w_mask, h_mask))
    cropped_mask = trans_mask[0:h_thermal, 0:w_thermal]

    thermal_lung = cv2.bitwise_and(thermal, cropped_mask)

    out_name = thermal_iamge_path.split('/')[-1].split('.')[0] + '_lung'
    cv2.imwrite(output_image_path + '/' + out_name + '.png', thermal_lung)
