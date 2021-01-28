import cv2
import numpy as np

import torch

# for dynamic import
import importlib
import os

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

    train_data, test_data = dataset.get_instance(option = 3)

    






if __name__ == '__main__':
    main()
