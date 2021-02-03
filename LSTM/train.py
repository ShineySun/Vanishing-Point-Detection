import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch

# for dynamic import
import importlib
import os

import vis_util

import monospline as mnsp
from scipy.interpolate import CubicSpline

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

    two_train_data, two_test_data = dataset.get_instance(option = 3)

    for idx, train_data in enumerate(two_train_data):
        # image clip directory
        clip_dir = dataset.train_path + train_data['raw_file']
        # load image
        raw_image = cv2.imread(clip_dir)

        # get lanes
        lanes = dataset.get_lanes(train_data)

        # visualize annotatied data
        annot_image = raw_image.copy()
        annot_image = vis_util.draw_points(annot_image, lanes)

        # monospline
        intrp_image = raw_image.copy()

        intrp_point_set = []

        for lane in lanes:
            # print(lane)
            lane = np.flip(lane, axis=0)
            # print(lane)

            # y points -> x axis
            x = lane[:,1]
            # x points -> y axis
            y = lane[:,0]

            # intrp = mnsp.monospline(x,y)

            intrp = CubicSpline(x,y)

            x_intrp = np.linspace(720, 250, 100)
            # y_intrp = intrp.evaluate(x_intrp)
            y_intrp = intrp(x_intrp)

            points_intrp = np.array(list(zip(y_intrp, x_intrp)))

            intrp_point_set.append(points_intrp)


        intrp_image = vis_util.draw_points(intrp_image, intrp_point_set)





        cv2.imshow("raw_image", raw_image)
        cv2.imshow("annot_image", annot_image)
        cv2.imshow("intrp_image", intrp_image)
        cv2.waitKey(-1)










if __name__ == '__main__':
    main()
