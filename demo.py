import os
from utils.applyLungThermal import applyLungThermal
from utils.circleInflamArea import circleInflamArea
from segmentation import segmentation

if __name__ == '__main__':
    thermal_folder = './input/thermal'
    mask_path = './input/mask.png'
    input_path = './input/RGB'
    output_path = './output'
    weight_path = './model_simulated_RGB_mgpu_scaling_append.0071.h5'
    scale_list = [1]

    _, kpts_dict = segmentation(weight_path, input_path, output_path, scale_list)
    for filename, kpts in kpts_dict.items():
        kpts_list = []
        for _, kpt in kpts.items():
            kpts_list = kpts_list + [list(kpt)]
        thermal_path = thermal_folder + '/' + filename
        lung_path = output_path + '/' + os.path.splitext(filename)[0] + '_lung.png'
        lower_temp = [0, 0, 128]
        higher_temp = [255, 255, 255]

        applyLungThermal(thermal_path, mask_path, output_path, kpts_list)
        circleInflamArea(lung_path, output_path, lower_temp, higher_temp)
