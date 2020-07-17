import os
import glob
import argparse

import cv2
import numpy as np
from config_reader import config_reader
from scipy.ndimage.filters import gaussian_filter
from model_simulated_RGB101 import get_testing_model_resnet101
from human_seg.human_seg_gt import human_seg_combine_argmax

# parse arguments
parser = argparse.ArgumentParser(description='loading eval params')
parser.add_argument('--gpus', metavar='N', type=int, default=1)
parser.add_argument('--model', type=str, default='./weights/model_simulated_RGB_mgpu_scaling_append.0071.h5',
                    help='path to the weights file')
parser.add_argument('--input_folder', type=str, default='./input', help='path to the folder with test images')
parser.add_argument('--output_folder', type=str, default='./output', help='path to the output folder')
parser.add_argument('--max', type=bool, default=True)
parser.add_argument('--average', type=bool, default=False)
parser.add_argument('--scale', action='append', help='<Required> Set flag', required=True)

args = parser.parse_args()

right_part_idx = [2, 3, 4, 8, 9, 10, 14, 16]
left_part_idx = [5, 6, 7, 11, 12, 13, 15, 17]
human_part = [0, 1, 2, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13]
human_ori_part = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
seg_num = 15  # current model supports 15 parts only

# find connection in the specified sequence, center 29 is in the position 15
limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8],
           [2, 9], [9, 10], [10, 11], [2, 12], [12, 13],
           [13, 14], [2, 1], [1, 15], [15, 17], [1, 16],
           [16, 18], [3, 17], [6, 18]]

# the middle joints heatmap correpondence
mapIdx = [[31, 32], [39, 40], [33, 34], [35, 36], [41, 42],
          [43, 44], [19, 20], [21, 22], [23, 24], [25, 26],
          [27, 28], [29, 30], [47, 48], [49, 50], [53, 54],
          [51, 52], [55, 56], [37, 38], [45, 46]]

# visualize
colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
          [255, 255, 0], [170, 255, 0], [85, 255, 0],
          [0, 255, 0], [0, 255, 85], [0, 255, 170],
          [0, 255, 255], [0, 170, 255], [0, 85, 255],
          [0, 0, 255], [85, 0, 255], [170, 0, 255],
          [255, 0, 255], [255, 0, 170], [255, 0, 85]]

ori_paf_idx = [12, 13, 20, 21, 14, 15, 16, 17, 22, 23, 24, 25, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 28, 29, 30, 31, 34, 35, 32, 33, 36, 37, 18, 19, 26, 27]
flip_paf_idx = [20, 21, 12, 13, 22, 23, 24, 25, 14, 15, 16, 17, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 28, 29, 32, 33, 36, 37, 30, 31, 34, 35, 26, 27, 18, 19]
x_paf_idx = [20, 12, 22, 24, 14, 16, 6, 8, 10, 0, 2, 4, 28, 32, 36, 30, 34, 26, 18]


def recover_flipping_output(oriImg, heatmap_ori_size, paf_ori_size, part_ori_size):
    heatmap_ori_size = heatmap_ori_size[:, ::-1, :]
    heatmap_flip_size = np.zeros((oriImg.shape[0], oriImg.shape[1], 19))
    heatmap_flip_size[:, :, left_part_idx] = heatmap_ori_size[:, :, right_part_idx]
    heatmap_flip_size[:, :, right_part_idx] = heatmap_ori_size[:, :, left_part_idx]
    heatmap_flip_size[:, :, 0:2] = heatmap_ori_size[:, :, 0:2]

    paf_ori_size = paf_ori_size[:, ::-1, :]
    paf_flip_size = np.zeros((oriImg.shape[0], oriImg.shape[1], 38))
    paf_flip_size[:, :, ori_paf_idx] = paf_ori_size[:, :, flip_paf_idx]
    paf_flip_size[:, :, x_paf_idx] = paf_flip_size[:, :, x_paf_idx] * -1

    part_ori_size = part_ori_size[:, ::-1, :]
    part_flip_size = np.zeros((oriImg.shape[0], oriImg.shape[1], 15))
    part_flip_size[:, :, human_ori_part] = part_ori_size[:, :, human_part]
    return heatmap_flip_size, paf_flip_size, part_flip_size


def recover_flipping_output2(oriImg, part_ori_size):
    part_ori_size = part_ori_size[:, ::-1, :]
    part_flip_size = np.zeros((oriImg.shape[0], oriImg.shape[1], 15))
    part_flip_size[:, :, human_ori_part] = part_ori_size[:, :, human_part]
    return part_flip_size


def part_thresholding(seg_argmax):
    background = 0.6
    head = 0.5
    torso = 0.8

    rightfoot = 0.55
    leftfoot = 0.55
    leftthigh = 0.55
    rightthigh = 0.55
    leftshank = 0.55
    rightshank = 0.55
    rightupperarm = 0.55
    leftupperarm = 0.55
    rightforearm = 0.55
    leftforearm = 0.55
    lefthand = 0.55
    righthand = 0.55

    part_th = [background, head, torso, leftupperarm, rightupperarm, leftforearm, rightforearm, lefthand, righthand,
               leftthigh, rightthigh, leftshank, rightshank, leftfoot, rightfoot]
    th_mask = np.zeros(seg_argmax.shape)
    for indx in range(15):
        part_prediction = (seg_argmax == indx)
        part_prediction = part_prediction * part_th[indx]
        th_mask += part_prediction

    return th_mask


def process(input_image, params, model_params):
    input_scale = 1.0

    oriImg = cv2.imread(input_image)
    flipImg = cv2.flip(oriImg, 1)
    oriImg = (oriImg / 256.0) - 0.5
    flipImg = (flipImg / 256.0) - 0.5
    multiplier = [x * model_params['boxsize'] / oriImg.shape[0] for x in params['scale_search']]

    heatmap_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 19))
    paf_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 38))

    segmap_scale1 = np.zeros((oriImg.shape[0], oriImg.shape[1], seg_num))
    segmap_scale2 = np.zeros((oriImg.shape[0], oriImg.shape[1], seg_num))
    segmap_scale3 = np.zeros((oriImg.shape[0], oriImg.shape[1], seg_num))
    segmap_scale4 = np.zeros((oriImg.shape[0], oriImg.shape[1], seg_num))

    segmap_scale5 = np.zeros((oriImg.shape[0], oriImg.shape[1], seg_num))
    segmap_scale6 = np.zeros((oriImg.shape[0], oriImg.shape[1], seg_num))
    segmap_scale7 = np.zeros((oriImg.shape[0], oriImg.shape[1], seg_num))
    segmap_scale8 = np.zeros((oriImg.shape[0], oriImg.shape[1], seg_num))

    for m in range(len(multiplier)):
        scale = multiplier[m] * input_scale
        imageToTest = cv2.resize(oriImg, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        pad = [0, 0,
               (imageToTest.shape[0] - model_params['stride']) % model_params['stride'],
               (imageToTest.shape[1] - model_params['stride']) % model_params['stride']]

        imageToTest_padded = np.pad(imageToTest, ((0, pad[2]), (0, pad[3]), (0, 0)), mode='constant',
                                    constant_values=((0, 0), (0, 0), (0, 0)))

        input_img = imageToTest_padded[np.newaxis, ...]

        print("\tActual size fed into NN: ", input_img.shape)

        output_blobs = model.predict(input_img)

        seg = np.squeeze(output_blobs[2])
        seg = cv2.resize(seg, (0, 0), fx=model_params['stride'], fy=model_params['stride'],
                         interpolation=cv2.INTER_CUBIC)
        seg = seg[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
        seg = cv2.resize(seg, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)

        if m == 0:
            segmap_scale1 = seg
        elif m == 1:
            segmap_scale2 = seg
        elif m == 2:
            segmap_scale3 = seg
        elif m == 3:
            segmap_scale4 = seg

    # flipping
    for m in range(len(multiplier)):
        scale = multiplier[m]
        imageToTest = cv2.resize(flipImg, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        pad = [0,
               0,
               (imageToTest.shape[0] - model_params['stride']) % model_params['stride'],
               (imageToTest.shape[1] - model_params['stride']) % model_params['stride']
               ]

        imageToTest_padded = np.pad(imageToTest, ((0, pad[2]), (0, pad[3]), (0, 0)), mode='constant',
                                    constant_values=((0, 0), (0, 0), (0, 0)))
        input_img = imageToTest_padded[np.newaxis, ...]
        print("\tActual size fed into NN: ", input_img.shape)
        output_blobs = model.predict(input_img)

        # extract outputs, resize, and remove padding
        heatmap = np.squeeze(output_blobs[1])  # output 1 is heatmaps
        heatmap = cv2.resize(heatmap, (0, 0), fx=model_params['stride'], fy=model_params['stride'],
                             interpolation=cv2.INTER_CUBIC)
        heatmap = heatmap[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
        heatmap = cv2.resize(heatmap, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)

        paf = np.squeeze(output_blobs[0])  # output 0 is PAFs
        paf = cv2.resize(paf, (0, 0), fx=model_params['stride'], fy=model_params['stride'],
                         interpolation=cv2.INTER_CUBIC)
        paf = paf[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
        paf = cv2.resize(paf, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)

        seg = np.squeeze(output_blobs[2])
        seg = cv2.resize(seg, (0, 0), fx=model_params['stride'], fy=model_params['stride'],
                         interpolation=cv2.INTER_CUBIC)
        seg = seg[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
        seg = cv2.resize(seg, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)

        heatmap_recover, paf_recover, seg_recover = recover_flipping_output(oriImg, heatmap, paf, seg)

        heatmap_avg = heatmap_avg + heatmap_recover
        paf_avg = paf_avg + paf_recover

        if m == 0:
            segmap_scale5 = seg_recover
        elif m == 1:
            segmap_scale6 = seg_recover
        elif m == 2:
            segmap_scale7 = seg_recover
        elif m == 3:
            segmap_scale8 = seg_recover

    heatmap_avg = heatmap_avg / (len(multiplier) * 2)

    segmap_a = np.maximum(segmap_scale1, segmap_scale2)
    segmap_b = np.maximum(segmap_scale4, segmap_scale3)
    segmap_c = np.maximum(segmap_scale5, segmap_scale6)
    segmap_d = np.maximum(segmap_scale7, segmap_scale8)
    seg_ori = np.maximum(segmap_a, segmap_b)
    seg_flip = np.maximum(segmap_c, segmap_d)
    seg_avg = np.maximum(seg_ori, seg_flip)

    all_peaks = []
    peak_counter = 0

    for part in range(18):
        map_ori = heatmap_avg[:, :, part]
        map = gaussian_filter(map_ori, sigma=3)

        map_left = np.zeros(map.shape)
        map_left[1:, :] = map[:-1, :]
        map_right = np.zeros(map.shape)
        map_right[:-1, :] = map[1:, :]
        map_up = np.zeros(map.shape)
        map_up[:, 1:] = map[:, :-1]
        map_down = np.zeros(map.shape)
        map_down[:, :-1] = map[:, 1:]

        peaks_binary = np.logical_and.reduce(
            (map >= map_left, map >= map_right, map >= map_up, map >= map_down, map > params['thre1']))
        peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]))  # note reverse
        peaks_with_score = [x + (map_ori[x[1], x[0]],) for x in peaks]
        id = range(peak_counter, peak_counter + len(peaks))
        peaks_with_score_and_id = [peaks_with_score[i] + (id[i],) for i in range(len(id))]

        all_peaks.append(peaks_with_score_and_id)
        peak_counter += len(peaks)

    canvas = cv2.imread(input_image).copy()

    for i in range(18):
        for j in range(len(all_peaks[i])):
            cv2.circle(canvas, all_peaks[i][j][0:2], 4, colors[i], thickness=-1)

    return canvas, seg_avg


if __name__ == '__main__':

    args = parser.parse_args()
    keras_weights_file = args.model

    print('start processing...')
    # load model
    model = get_testing_model_resnet101()
    model.load_weights(keras_weights_file)
    params, model_params = config_reader()

    scale_list = []
    for item in args.scale:
        scale_list.append(float(item))

    params['scale_search'] = scale_list

    # generate image with body parts
    image_files = glob.glob(os.path.join(args.input_folder, '**'), recursive=True)
    image_exts = (".png", ".jpg", ".bmp")
    image_files = [os.path.abspath(file) for file in image_files if (os.path.splitext(file)[1] in image_exts)]

    for filename in image_files:
        print(filename)

        base_filename = os.path.basename(filename)
        file_category = os.path.basename(os.path.dirname(filename))
        output_folder = os.path.abspath(args.output_folder)

        # write sk
        canvas, seg = process(filename, params, model_params)
        sk_outfile = os.path.join(output_folder, file_category, 'sk_%s' % base_filename)
        cv2.imwrite(sk_outfile, canvas)
        print("Written to: %s" % sk_outfile)

        seg_argmax = np.argmax(seg, axis=-1)
        seg_max = np.max(seg, axis=-1)
        seg_max_thres = (seg_max > 0.1).astype(np.uint8)
        seg_argmax *= seg_max_thres

        # write seg
        seg_canvas = human_seg_combine_argmax(seg_argmax)
        seg_outfile = os.path.join(output_folder, file_category, 'seg_%s' % base_filename)
        cv2.imwrite(seg_outfile, seg_canvas)
        print("Written to: %s " % seg_outfile)
