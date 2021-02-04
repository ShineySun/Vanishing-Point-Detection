import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch

# for dynamic import
import importlib
import os

import vis_util

import monospline as mnsp
import interpolation

def main():
    # Select GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    print("Device ID : {}".format(format(torch.cuda.current_device())))
    print("torch.version : {}".format(torch.__version__))
    print("cuda.version : {}".format(torch.version.cuda))

    models = importlib.import_module('models.init')
    data_manager = importlib.import_module("data_manager.tusimple_manager")

    dataset = data_manager.Tusimple_Manager()
    dataset.tusimple_load_from_json()
    dataset.tusimple_split_instance()

    two_train_data, two_test_data = dataset.get_instance(option = 0)

    for idx, train_data in enumerate(two_train_data):
        # image clip directory
        clip_dir = dataset.train_path + train_data['raw_file']
        print("* clip dir : {}".format(clip_dir))
        # load image
        raw_image = cv2.imread(clip_dir)

        # get lanes
        lanes = dataset.get_lanes(train_data)

        # visualize annotatied data
        annot_image = raw_image.copy()
        annot_image = vis_util.draw_points(annot_image, lanes)

        # cubic spline interpolation
        cubic_spline_image = raw_image.copy()

        cubic_spline_sets = interpolation.cubic_spline(lanes)

        cubic_spline_image = vis_util.draw_points(cubic_spline_image, cubic_spline_sets)

        # mono spline interpolation
        mono_spline_image = raw_image.copy()

        mono_spline_sets = interpolation.mono_spline(lanes)

        mono_spline_image = vis_util.draw_points(mono_spline_image, mono_spline_sets)

        # linear interpolation
        linear_intrp_image = raw_image.copy()

        linear_intrp_sets = interpolation.linear_interpolate(lanes)

        linear_intrp_image = vis_util.draw_points(linear_intrp_image, linear_intrp_sets)

        # quadratic interpolation
        quadratic_intrp_image = raw_image.copy()

        quadratic_intrp_sets = interpolation.quadratic_interpolate(lanes)

        quadratic_intrp_image = vis_util.draw_points(quadratic_intrp_image, quadratic_intrp_sets)





        # cv2.imshow("raw_image", raw_image)
        # cv2.imshow("annot_image", annot_image)
        # cv2.imshow("cubic_spline_image", cubic_spline_image)
        # cv2.imshow("mono_spline_image", mono_spline_image)
        # cv2.imshow("linear_intrp_image", linear_intrp_image)
        cv2.imshow("quadratic_intrp_image", quadratic_intrp_image)
        cv2.waitKey(-1)


if __name__ == '__main__':
    main()
