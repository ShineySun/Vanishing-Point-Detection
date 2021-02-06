import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable

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

    model_lstm = importlib.import_module('models.lstm')
    data_manager = importlib.import_module("data_manager.tusimple_manager")

    dataset = data_manager.Tusimple_Manager()
    dataset.tusimple_load_from_json()
    dataset.tusimple_split_instance()

    two_train_data, two_test_data = dataset.get_instance(option = 2)

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

        #----------------------------------------------------------------------------------------------------------------#

        seq_len_ = 60

        train_sets, test_sets = dataset.split_train_test(mono_spline_sets, test_size = 3, seq_len = seq_len_)

        # sequence train data
        train_x_s = []
        train_y_s = []

        # sequence test data
        test_x_s = []
        test_y_s = []

        for train_set in train_sets:
            train_x, train_y = dataset.make_sequence_data(train_set, seq_len=seq_len_)
            train_x_s.append(Variable(torch.Tensor(train_x)))
            train_y_s.append(Variable(torch.Tensor(train_y)))

        for test_set in test_sets:
            test_x, test_y = dataset.make_sequence_data(test_set, seq_len=seq_len_)
            test_x_s.append(Variable(torch.Tensor(test_x)))
            test_y_s.append(Variable(torch.Tensor(test_y)))


        # model
        models = []
        criterions = []
        optimizers = []

        for i in range(len(train_x_s)):
            model = model_lstm.LSTM()
            # criterion = torch.nn.SmoothL1Loss()
            criterion = torch.nn.MSELoss()
            #criterion = torch.nn.L1Loss()

            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            models.append(model)
            criterions.append(criterion)
            optimizers.append(optimizer)

        epochs = 500

        train_loss_lane = []
        test_loss_lane = []

        for epoch in range(epochs):
            # train_session
            for idx, (train_x_tensor, train_y_tensor) in enumerate(zip(train_x_s, train_y_s)):
                models[idx].train()
                optimizers[idx].zero_grad()

                predict = models[idx].forward(train_x_tensor.view(len(train_x_tensor), seq_len_, 1))

                loss = criterions[idx](predict, train_y_tensor)
                loss.backward()

                if epoch == 0:
                    lane_loss = [loss.item()]
                    train_loss_lane.append(lane_loss)
                else:
                    train_loss_lane[idx].append(loss.item())

                optimizers[idx].step()

            # test session
            with torch.no_grad():
                for idx, (test_x_tensor, test_y_tensor) in enumerate(zip(test_x_s, test_y_s)):
                    models[idx].eval()

                    # forward
                    inference = models[idx].forward(test_x_tensor.view(len(test_x_tensor), seq_len_, 1))

                    loss = criterions[idx](inference, test_y_tensor)

                    if epoch == 0:
                        lane_loss = [loss.item()]
                        test_loss_lane.append(lane_loss)
                    else:
                        test_loss_lane[idx].append(loss.item())

        for idx, (train_loss, test_loss) in enumerate(zip(train_loss_lane, test_loss_lane)):
            plt.plot(train_loss, label='train_loss_'+ str(idx))
            plt.plot(test_loss, label='test_loss_'+ str(idx))
            plt.legend()

        plt.show()

        inference_set = np.arange(720,0,-1)

        test_set = dataset.make_test_sequence_data(inference_set, seq_len_)/720.0

        test_sets = []

        for i in range(len(train_x_s)):
            test_sets.append(Variable(torch.Tensor(test_set)))

        test_image = raw_image.copy()

        with torch.no_grad():
            color_ = [(255,100,0), (100,255,0), (0,0,255), (255,100,100), (0,120,80)]

            for idx, (test_x_tensor, test_y_tensor) in enumerate(zip(test_sets, test_y_s)):

            #for idx, (test_x_tensor, test_y_tensor) in enumerate(zip(test_x_s, test_y_s)):
                models[idx].eval()

                # forward
                inference = models[idx].forward(test_x_tensor.view(len(test_x_tensor), seq_len_, 1)).data.numpy()

                test_x_numpy = test_x_tensor.data.numpy()
                test_y_numpy = test_y_tensor.data.numpy()*1280

                y_points = test_x_numpy[:, -1]*720 - 10

                inference = np.squeeze(inference, axis=1)*1280

                gt_pair = np.array(list(zip(test_y_numpy, y_points)))

                predict_pair = np.array(list(zip(inference, y_points)))
                print(predict_pair)

                for pt in predict_pair:
                    cv2.circle(test_image, (int(pt[0]), int(pt[1])), radius=2, color = color_[idx], thickness=2)
                # for pt in gt_pair:
                #     cv2.circle(test_image, (int(pt[0]), int(pt[1])), radius=2, color = color_[idx+1], thickness=2)




        cv2.imshow("raw_image", raw_image)
        '''
        cv2.imwrite("/home/sun/Desktop/Vanishing-Point-Detection/interpolation_image/annot_image.jpg", annot_image)
        cv2.imwrite("/home/sun/Desktop/Vanishing-Point-Detection/interpolation_image/cubic_spline_image.jpg", cubic_spline_image)
        cv2.imwrite("/home/sun/Desktop/Vanishing-Point-Detection/interpolation_image/mono_spline_image.jpg", mono_spline_image)
        cv2.imwrite("/home/sun/Desktop/Vanishing-Point-Detection/interpolation_image/linear_intrp_image.jpg", linear_intrp_image)
        cv2.imwrite("/home/sun/Desktop/Vanishing-Point-Detection/interpolation_image/quadratic_intrp_image.jpg", quadratic_intrp_image)
        '''

        cv2.imshow("test_image", test_image)
        cv2.waitKey(-1)


if __name__ == '__main__':
    main()
