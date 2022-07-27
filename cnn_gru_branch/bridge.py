from turtle import right
import time
from sklearn.utils import check_array
from scipy.interpolate import CubicSpline
from scipy.interpolate import interp1d
import torch.nn as nn
import torch
from util_hourglass import *
from copy import deepcopy
import numpy as np

from torch.autograd import Function as F
from parameters import Parameters
import util
import matplotlib.pyplot as plt
import cv2

def bridge(result, vp_gt):
    confidance, offset, feature = result
    # lane_keypoints = []
    ego_left_pts = []
    ego_right_pts = []
    vp_gt_used = []
    images = []
    vp_batch_idx = []
    label_l = []
    label_r = []

    for i in range(p.batch_size):
        try: confidance[i]
        except: break

        confidence_ = confidance[i].view(p.grid_y, p.grid_x).cpu().data.numpy()

        offset_ = offset[i].cpu().data.numpy()
        offset_ = np.rollaxis(offset_, axis=2, start=0)
        offset_ = np.rollaxis(offset_, axis=2, start=0)

        instance_ = feature[i].cpu().data.numpy()
        instance_ = np.rollaxis(instance_, axis=2, start=0)
        instance_ = np.rollaxis(instance_, axis=2, start=0)

        x, y, img = generate_result_heatmap(confidence_, offset_, instance_, p.threshold_point)
        x, y = eliminate_fewer_points(x, y)
        left_pt, right_pt, for_vis = util.ego_lane(x, y)
        # print("left----------------")
        left_pt = spline_lane(left_pt)
        # print("right----------------")
        right_pt = spline_lane(right_pt)
        # 점 40개.. (7+1) * 5
        # 모델에서 7개 학습, 1개 예측을 5번 반복
        # 총 5번의 loss 계산 추가... 
        # 이때, hidden=24인데, 이를 matmul로 2로 만들어줌 아니다.. 이럼 랜덤이니까
        # 그냥 뒤에 FC layer 로 2로 만들어주기
        # 그럼, 각 시각화도 가능할 것
        # 그냥 (3+1) * 10 먼저 실험해보자 


        if left_pt is not None and right_pt is not None:
            left_pt, label_left = sliding_window(left_pt)
            right_pt, label_right = sliding_window(right_pt)

            vp_gt_used.append([vp_gt[i][0]/p.x_size, vp_gt[i][1]/p.y_size])
            ego_left_pts.append(left_pt)
            ego_right_pts.append(right_pt)
            images.append(img.reshape(1, p.y_size, p.x_size))
            vp_batch_idx.append(i)
            label_l.append(label_left)
            label_r.append(label_right)
        

    # if ego_left_pts != []:
    #         vp_gt_used = torch.from_numpy(np.array(vp_gt_used, dtype='float32')).cuda()
    #         pred_vp = gru(torch.from_numpy(np.array(ego_left_pts, dtype='float32')).cuda(), torch.from_numpy(np.array(ego_right_pts, dtype='float32')).cuda())
    if ego_left_pts != []:
        label_l = torch.from_numpy(np.array(label_l, dtype='float32')).cuda()
        label_r = torch.from_numpy(np.array(label_r, dtype='float32')).cuda()
        return torch.from_numpy(np.array(ego_left_pts, dtype='float32')), torch.from_numpy(np.array(ego_right_pts, dtype='float32')), torch.from_numpy(np.array(vp_gt_used, dtype='float32')).cuda(), vp_batch_idx, torch.from_numpy(np.array(images, dtype='float32')).cuda(), label_l, label_r, for_vis
    else:
        return None, None, None, None, None, None, None, None

def sliding_window(pt):
    tmp = []
    label = []
    for i in range(0, 39, 4): # i == 0, 8, 16, 24, 32 # i = 0, 4, 8, 12 ,16 , ... , 32, 36
        tmp.append(pt[i:i+3,:].flatten('F')) # [x,x,x,x,x,x,x,y,y,y,y,y,y,y]
        label.append(pt[i+3,:])
    return np.array(tmp, dtype='float32'), np.array(label, dtype='float32')

def spline_lane(pt):
    if pt != None:
        ys = np.array(pt).T[0]
        xs = np.array(pt).T[1] 
        # xs = -xs ## vp에 가까운 부분먼저 spline에 넣기위함. spline이후 원래대로 돌려줌

        s = xs.argsort()
        xs = xs[s]
        ys = ys[s]
        xs, unique_idx = np.unique(xs, return_index=True)
        ys = ys[unique_idx]
        # interpolation scheme : Cubic Spline
        try:
            cs_intrp = interp1d(xs, ys)
            # cs_intrp2 = interp1d(xs, ys, kind='quadratic')
            # cs_intrp = CubicSpline(xs, ys)
        except: 
            # print('cubic spline error') 
            # print('xs:', xs)
            # print('ys:', ys)
            return None
        # x_intrp = np.linspace(int(xs.min()), int(xs.max()), int(xs.max())-int(xs.min())+1)
        x_intrp = np.linspace(int(xs.min()), int(xs.max()), 40)
        y_intrp = cs_intrp(x_intrp)


        x_intrp /= p.y_size
        y_intrp /= p.x_size
        intrp_lane = np.array(list(zip(y_intrp, x_intrp)), dtype='float32')
        return intrp_lane
    else: 
        # print("there is no lane")
        return None

def eliminate_fewer_points(x, y):
    # eliminate fewer points
    out_x = []
    out_y = []
    for i, j in zip(x, y):
        if len(i) > 2:
            out_x.append(i)
            out_y.append(j)
    return out_x, out_y

def generate_result_heatmap(confidance, offsets,instance, thresh):
    
    mask = confidance > thresh
    grid = p.grid_location[mask]
    offset = offsets[mask]
    feature = instance[mask]
    confidance = confidance[np.where(confidance >= thresh)]
    lane_feature = []
    x = []
    y = []
    x_ = []
    y_ = []
    for i in range(len(grid)):
        if (np.sum(feature[i]**2))>=0:
            point_x = int((offset[i][0]+grid[i][0])*p.resize_ratio)
            point_y = int((offset[i][1]+grid[i][1])*p.resize_ratio)
            x_.append(point_x)
            y_.append(point_y)
            if point_x > p.x_size or point_x < 0 or point_y > p.y_size or point_y < 0:
                continue
            if len(lane_feature) == 0:
                lane_feature.append(feature[i])
                x.append([point_x])
                y.append([point_y])
            else:
                flag = 0
                index = 0
                min_feature_index = -1
                min_feature_dis = 10000
                for feature_idx, j in enumerate(lane_feature):
                    dis = np.linalg.norm((feature[i] - j)**2)
                    if min_feature_dis > dis:
                        min_feature_dis = dis
                        min_feature_index = feature_idx
                if min_feature_dis <= p.threshold_instance:
                    lane_feature[min_feature_index] = (lane_feature[min_feature_index]*len(x[min_feature_index]) + feature[i])/(len(x[min_feature_index])+1)
                    x[min_feature_index].append(point_x)
                    y[min_feature_index].append(point_y)
                elif len(lane_feature) < 12:
                    lane_feature.append(feature[i])
                    x.append([point_x])
                    y.append([point_y])
    # 히트맵 만들기
    image = np.zeros((p.y_size, p.x_size))
    for i, j, c in zip(x_, y_, confidance):
        image[j][i] = c
    image = cv2.normalize(image, None, 0, 1,
                          cv2.NORM_MINMAX, dtype=cv2.CV_8U)  # uint8로변환
    for i, j, c in zip(x_, y_, confidance):
        confi = int(c*1)
        image = cv2.circle(image, (i, j), 8,
                           (confi, confi, confi), -1)
    image = cv2.GaussianBlur(image, (0, 0), 8, 4) # (512,256)이므로 x축 sigma를 2배로 설정
    # image = cv2.applyColorMap(image, cv2.COLORMAP_JET)
    # cv2.imshow("image", image) # image 보고싶으면 범위 0~256으로 지정
    # cv2.waitKey(0)
                
    return x, y, image
