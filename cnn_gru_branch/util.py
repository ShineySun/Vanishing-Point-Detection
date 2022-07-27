import torch.nn as nn
import cv2
import torch
from copy import deepcopy
import numpy as np
from torch.autograd import Variable
from torch.autograd import Function as F
from parameters import Parameters
import math
import matplotlib.pyplot as plt 

p = Parameters()

###############################################################
##
## visualize
## 
###############################################################

def visualize_points(image, x, y):
    image = image
    image =  np.rollaxis(image, axis=2, start=0)
    image =  np.rollaxis(image, axis=2, start=0)#*255.0
    image = image.astype(np.uint8).copy()

    for k in range(len(y)):
        for i, j in zip(x[k], y[k]):
            if i > 0:
                image = cv2.circle(image, (int(i), int(j)), 2, p.color[1], -1)

    cv2.imshow("test2", image)
    cv2.waitKey(0)  

def visualize_points_origin_size(x, y, test_image, ratio_w, ratio_h):
    color = 0
    image = deepcopy(test_image)
    image =  np.rollaxis(image, axis=2, start=0)
    image =  np.rollaxis(image, axis=2, start=0)#*255.0
    image = image.astype(np.uint8).copy()

    image = cv2.resize(image, (int(p.x_size/ratio_w), int(p.y_size/ratio_h)))

    for i, j in zip(x, y):
        color += 1
        for index in range(len(i)):
            cv2.circle(image, (int(i[index]), int(j[index])), 10, p.color[color], -1)
    cv2.imshow("test2", image)
    cv2.waitKey(0)  

    return test_image

def visualize_gt(gt_point, gt_instance, ground_angle, image):
    image =  np.rollaxis(image, axis=2, start=0)
    image =  np.rollaxis(image, axis=2, start=0)#*255.0
    image = image.astype(np.uint8).copy()

    for y in range(p.grid_y):
        for x in range(p.grid_x):
            if gt_point[0][y][x] > 0:
                xx = int(gt_point[1][y][x]*p.resize_ratio+p.resize_ratio*x)
                yy = int(gt_point[2][y][x]*p.resize_ratio+p.resize_ratio*y)
                image = cv2.circle(image, (xx, yy), 10, p.color[1], -1)

    cv2.imshow("image", image)
    cv2.waitKey(0)

def visualize_regression(image, gt):
    image =  np.rollaxis(image, axis=2, start=0)
    image =  np.rollaxis(image, axis=2, start=0)*255.0
    image = image.astype(np.uint8).copy()

    for i in gt:
        for j in range(p.regression_size):#gt
            y_value = p.y_size - (p.regression_size-j)*(220/p.regression_size)
            if i[j] >0:
                x_value = int(i[j]*p.x_size)
                image = cv2.circle(image, (x_value, y_value), 5, p.color[1], -1)
    cv2.imshow("image", image)
    cv2.waitKey(0)   


def draw_points(x, y, image):
    color_index = 0

    for i, j in zip(x, y):
        color_index += 1
        if color_index > 12:
            color_index = 12
        for index in range(len(i)):
            image = cv2.circle(image, (int(i[index]), int(j[index])), 5, p.color[color_index], -1)

    return image

def draw_points_vp(vp_pred, vp_gt, image):
    x_pred = vp_pred[0]*p.x_size
    x_gt = np.array(vp_gt)[0]*p.x_size
    y_pred = vp_pred[1]*p.y_size
    y_gt = np.array(vp_gt)[1]*p.y_size
    image = cv2.circle(image, (int(x_pred), int(y_pred)), 5, (0,0,255), thickness=-1) # 빨강
    image = cv2.circle(image, (int(x_gt), int(y_gt)), 5, (0,0,0), thickness=-1) #검정
    # cv2.imshow('img', image)
    # cv2.waitKey(0)

    return image

###############################################################
##
## calculate
## 
###############################################################
def convert_to_original_size(x, y, ratio_w, ratio_h):
    # convert results to original size
    out_x = []
    out_y = []

    for i, j in zip(x,y):
        out_x.append((np.array(i)/ratio_w).tolist())
        out_y.append((np.array(j)/ratio_h).tolist())

    return out_x, out_y

def get_closest_point_along_angle(x, y, point, angle):
    index = 0
    for i, j in zip(x, y): 
        a = get_angle_two_points(point, (i,j))
        if abs(a-angle) < 0.1:
            return (i, j), index
        index += 1
    return (-1, -1), -1


def get_num_along_point(x, y, point1, point2, image=None): # point1 : source
    x = np.array(x)
    y = np.array(y)

    x = x[y<point1[1]]
    y = y[y<point1[1]]

    dis = np.sqrt( (x - point1[0])**2 + (y - point1[1])**2 )

    count = 0
    shortest = 1000
    target_angle = get_angle_two_points(point1, point2)
    for i in range(len(dis)):
        angle = get_angle_two_points(point1, (x[i], y[i]))
        diff_angle = abs(angle-target_angle)
        distance = dis[i] * math.sin( diff_angle*math.pi*2 )
        if distance <= 12:
            count += 1
            if distance < shortest:
                shortest = distance

    return count, shortest

def get_closest_upper_point(x, y, point, n):
    x = np.array(x)
    y = np.array(y)

    x = x[y<point[1]]
    y = y[y<point[1]]

    dis = (x - point[0])**2 + (y - point[1])**2

    ind = np.argsort(dis, axis=0)
    x = np.take_along_axis(x, ind, axis=0).tolist()
    y = np.take_along_axis(y, ind, axis=0).tolist()

    points = []
    for i, j in zip(x[:n], y[:n]):
        points.append((i,j))

    return points

def sort_along_y(x, y):
    out_x = []
    out_y = []

    for i, j in zip(x, y):
        i = np.array(i)
        j = np.array(j)

        ind = np.argsort(j, axis=0)
        out_x.append(np.take_along_axis(i, ind[::-1], axis=0).tolist())
        out_y.append(np.take_along_axis(j, ind[::-1], axis=0).tolist())
    
    return out_x, out_y

def sort_along_x(x, y):
    out_x = []
    out_y = []

    for i, j in zip(x, y):
        i = np.array(i)
        j = np.array(j)

        ind = np.argsort(i, axis=0)
        out_x.append(np.take_along_axis(i, ind[::-1], axis=0).tolist())
        out_y.append(np.take_along_axis(j, ind[::-1], axis=0).tolist())
    
    return out_x, out_y

def sort_batch_along_y(target_lanes, target_h):
    out_x = []
    out_y = []

    for x_batch, y_batch in zip(target_lanes, target_h):
        temp_x = []
        temp_y = []
        for x, y, in zip(x_batch, y_batch):
            ind = np.argsort(y, axis=0)
            sorted_x = np.take_along_axis(x, ind[::-1], axis=0)
            sorted_y = np.take_along_axis(y, ind[::-1], axis=0)
            temp_x.append(sorted_x)
            temp_y.append(sorted_y)
        out_x.append(temp_x)
        out_y.append(temp_y)
    
    return out_x, out_y


def ego_lane(x, y):

    ego_lane = []
    left_lane = []
    right_lane = []

    center_x = 512/2
    center_y = 256/2

    for idx, cluster in enumerate(x):
        point_cluster = list(
            filter(lambda xyPair: xyPair[1] > center_y+50, zip(cluster, y[idx])))

        x_arr = []

        for i in range(len(point_cluster)):
            x_pt, y_pt = point_cluster[i]

            x_arr.append(x_pt)

        if len(x_arr) == 0:
            continue

        mean_val = np.mean(x_arr)

        diff = center_x - mean_val

        #print(idx," : ", diff)

        if diff < 0:
            left_lane.append((idx, diff))
        else:
            right_lane.append((idx, diff))

        ego_lane.append((idx, diff))

    ego_lane.sort(key=abs_diff)
    left_lane.sort(key=abs_diff)
    right_lane.sort(key=abs_diff)

    if len(right_lane) == 0 or len(left_lane) == 0:

        return None, None, None

    color_index = 0

    #print(x)

    left_lane_idx = left_lane[0][0]

    color_index += 1
    

    left_lane_pt = list(map(list, zip(x[left_lane_idx], y[left_lane_idx])))

    right_lane_idx = right_lane[0][0]
    # # visualize ego lanes
    # plt.axis([0, 512, 256, 0])
    # plt.scatter(x[left_lane_idx], y[left_lane_idx], color="red")
    # plt.scatter(x[right_lane_idx], y[right_lane_idx], color="blue")

    # plt.xlabel("X")
    # plt.ylabel("Y")
    # plt.title("Scatter Plot")
    # plt.show()

    color_index += 1
    right_lane_pt = list(map(list, zip(x[right_lane_idx], y[right_lane_idx])))

    return left_lane_pt, right_lane_pt, [x[left_lane_idx], y[left_lane_idx], x[right_lane_idx], y[right_lane_idx]]

def abs_diff(t):
    return abs(t[1])
