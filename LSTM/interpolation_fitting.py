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

def curve_fit_mode(x_data,y_data, mode=1, result_img = None):
        def objective_linear(x, a, b):
	           return a * x + b

        def objective_quadratic(x,a,b,c):
            return a*x + b*x**2 + c

        def objective_cubic(x,a,b,c,d):
            return a*x + b*x**2 + c*x**3 + d

        def objective_quartic(x,a,b,c,d,e):
	        return a*x + b*x**2 + c*x**3 + d*x**4 + e

        return_list = []

        for batch in range(len(x_data)):
            new_x = []
            new_y = []
            batch_len = len(x_data[batch])
            batch_len = 0

            #print("class num : {}".format(len(x_data)))

            for idx in range(len(x_data[batch])):
                if idx >= batch_len:
                    new_x.append(x_data[batch][idx])
                    new_y.append(y_data[batch][idx])
                    # curve fit
            if mode == 1:
                try:
                    popt, _ = curve_fit(objective_linear,new_x, new_y)
                    a,b = popt
                except:
                    continue
            elif mode == 2:
                try:
                    popt, _ = curve_fit(objective_quadratic, new_x, new_y)
                    a,b,c = popt
                except:
                    continue
            elif mode == 3:
                try:
                    popt, _ = curve_fit(objective_cubic, new_x, new_y)
                    a,b,c,d = popt
                except:
                    continue
            elif mode == 4:
                try:
                    popt, _ = curve_fit(objective_quartic, new_x, new_y)
                    a,b,c,d,e = popt
                except:
                    continue
            # plot input vs output
            #pyplot.scatter(new_x, new_y)

            # define a sequence of inputs between the smallest and largest known inputs
            #x_line = arange(min(new_x), max(new_x), 1)
            x_line = np.arange(0, 720, 1)

            # calculate the output for the range
            if mode == 1:
                y_line = objective_linear(x_line, a, b)
            elif mode == 2:
                y_line = objective_quadratic(x_line, a, b, c)
            elif mode == 3:
                y_line = objective_cubic(x_line,a, b, c, d)
            elif mode == 4:
                y_line = objective_quartic(x_line,a, b, c, d, e)

            tmp_list = []

            if result_img is not None:
                for x,y in zip(x_line, y_line):
                    try:
                        return_list.append([int(y), int(x)])
                        cv2.circle(result_img, (int(y),int(x)), 1, (0,255,255))
                    except OverflowError:
                        print("Overflow")
                        continue
        # cv2.imshow("result_img", result_img)
        # cv2.waitKey(-1)
        return return_list

def vanish_point(fit_points, result_img):
        #print(fit_points)
        numpy_fit_points = np.array(fit_points)
        #print("len(fit_points) : {}".format(len(fit_points)))
        #print(numpy_fit_points.shape)

        score_matrix = np.zeros((len(numpy_fit_points),len(numpy_fit_points)))

        for i in range(len(numpy_fit_points)-1):
            for j in range(i+1, len(numpy_fit_points)):
                dist = np.linalg.norm(numpy_fit_points[i][:2] - numpy_fit_points[j][:2],2)
                #print(dist)

                if dist < 1:
                    dist = 1.0

                score_matrix[i][j] = 1/dist
                score_matrix[j][i] = score_matrix[i][j]

        sum_score_matrix = score_matrix.sum(axis=1)
        sum_score_vector = sum_score_matrix.sum()

        score_vector = sum_score_matrix/sum_score_vector
        sorted_score_vector = np.sort(score_vector)[::-1]

        score_idx = []

        for i in range(len(sorted_score_vector)):
            tmp_index = np.where(score_vector == sorted_score_vector[i])

            try:
                score_idx.append(tmp_index[0][0])
            except:
                return None

        num_point_rank = min(10, len(fit_points))

        topN_vp = []

        for i in range(num_point_rank):
            topN_vp.append(fit_points[score_idx[i]][:2])

        mean_topN_vp_x = np.mean(topN_vp[:], axis=0)
        cv2.circle(result_img, (int(mean_topN_vp_x[0]),int(mean_topN_vp_x[1])),6,(255,0,255),3)

        # cv2.imshow('result_img', result_img)
        # cv2.waitKey(-1)

        return mean_topN_vp_x



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

    two_train_data, two_test_data = dataset.get_instance(option = 0)

    for idx_, train_data in enumerate(two_train_data):
        if idx_ == 20: break
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
        seq_len_ = 2

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

        #plt.show()

        inference_set = np.arange(720,0,-1)

        test_set = dataset.make_test_sequence_data(inference_set, seq_len_)/720.0

        test_sets = []

        for i in range(len(train_x_s)):
            test_sets.append(Variable(torch.Tensor(test_set)))

        test_image = raw_image.copy()

        vp_points = []

        with torch.no_grad():
            color_ = [(255,100,0), (100,255,0), (100,120,255), (255,100,100), (0,120,80)]

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
                    cv2.circle(annot_image, (int(pt[0]), int(pt[1])), radius=2, color = color_[idx], thickness=2)
                    vp_points.append(pt)

        real_vp_1 = vanish_point(vp_points, annot_image)
                # for pt in gt_pair:
                #     cv2.circle(test_image, (int(pt[0]), int(pt[1])), radius=2, color = color_[idx+1], thickness=2)






        # cv2.imshow("raw_image", annot_image)
        '''
        cv2.imwrite("/home/sun/Desktop/Vanishing-Point-Detection/interpolation_image/annot_image.jpg", annot_image)
        cv2.imwrite("/home/sun/Desktop/Vanishing-Point-Detection/interpolation_image/cubic_spline_image.jpg", cubic_spline_image)
        cv2.imwrite("/home/sun/Desktop/Vanishing-Point-Detection/interpolation_image/mono_spline_image.jpg", mono_spline_image)
        cv2.imwrite("/home/sun/Desktop/Vanishing-Point-Detection/interpolation_image/linear_intrp_image.jpg", linear_intrp_image)
        cv2.imwrite("/home/sun/Desktop/Vanishing-Point-Detection/interpolation_image/quadratic_intrp_image.jpg", quadratic_intrp_image)
        '''

        cv2.imwrite("/home/sun/Desktop/Vanishing-Point-Detection/interpolation_image/vanishing_points_"+str(idx_) + "_" + str(seq_len_) +".jpg", annot_image)
        # cv2.waitKey(-1)


if __name__ == '__main__':
    main()
