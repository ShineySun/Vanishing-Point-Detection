#############################################################################################################
##
##  Source code for testing
##
#############################################################################################################

import cv2
import json
import torch
import agent
import numpy as np
from copy import deepcopy
from data_loader_copy import Generator
import time
from parameters import Parameters
import util
import os
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from patsy import cr
import csaps
from scipy.interpolate import CubicSpline
from scipy.interpolate import interp1d
import pickle
import matplotlib.pyplot as plt

p = Parameters()

###############################################################
##
## Training
## 
###############################################################
def Testing():
    print('Testing')
    
    #########################################################################
    ## Get dataset
    #########################################################################
    print("Get dataset")
    loader = Generator()

    ##############################
    ## Get agent and model
    ##############################
    print('Get agent')
    if p.model_path == "":
        lane_agent = agent.Agent()
    else:
        lane_agent = agent.Agent()
        # lane_agent.load_weights(26, "tensor(1.0652)", "0.007536814548075199")
        lane_agent.load_weights(13, "tensor(1.4570)", "0.00014454746269620955") # best !!!! 
        # lane_agent.load_weights(6, "tensor(1.0949)", "0.00032658438431099057") #지금까지 best
	
    ##############################
    ## Check GPU
    ##############################
    print('Setup GPU mode')
    if torch.cuda.is_available():
        lane_agent.cuda()

    ##############################
    ## testing
    ##############################
    print('Testing loop')
    lane_agent.evaluate_mode()

    if p.mode == 0 : # check model with test data 
        for _, _, _, test_image, vp_gt in loader.Generate(): 
            _, _, ti = test(lane_agent, np.array([test_image]))
            cv2.imshow("test", ti[0])
            cv2.waitKey(0) 

    elif p.mode == 1: # check model with video
        cap = cv2.VideoCapture("/home/kym/research/autonomous_car_vision/lane_detection/code/Tusimple/git_version/LocalDataset_Day.mp4")
        while(cap.isOpened()):
            ret, frame = cap.read()
            torch.cuda.synchronize()
            prevTime = time.time()
            frame = cv2.resize(frame, (512,256))/255.0
            frame = np.rollaxis(frame, axis=2, start=0)
            _, _, ti = test(lane_agent, np.array([frame])) 
            curTime = time.time()
            sec = curTime - prevTime
            fps = 1/(sec)
            s = "FPS : "+ str(fps)
            ti[0] = cv2.resize(ti[0], (1280,800))
            cv2.putText(ti[0], s, (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
            cv2.imshow('frame',ti[0])
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    elif p.mode == 2: # check model with a picture
        #test_image = cv2.imread(p.test_root_url+"clips/0530/1492720840345996040_0/20.jpg")
        test_image = cv2.imread("./aa.png")
        test_image = cv2.resize(test_image, (512,256))/255.0
        test_image = np.rollaxis(test_image, axis=2, start=0)
        _, _, ti = test(lane_agent, np.array([test_image]))
        cv2.imshow("test", ti[0])
        cv2.waitKey(0)   

    elif p.mode == 3: #evaluation
        test_loss = [0, 0, 0]
        norm_dist = []
        print("evaluate")
        evaluation_test(test_loss, norm_dist, loader, lane_agent)
        print(test_loss[0]/test_loss[2], test_loss[1]/test_loss[2])
        save_norm_dist(norm_dist)


############################################################################
## evaluate on the test dataset
############################################################################
def evaluation(test_loss, loader, lane_agent, thresh = p.threshold_point, index= -1, name = None):
    progressbar = tqdm(range(loader.size_val//p.batch_size))
    for test_image, ratio_w, ratio_h, path, target_h, target_lanes, vp_gt in loader.Generate_Val():
        x, y, _ = test(test_loss, [], vp_gt, lane_agent, test_image, thresh, index= index)
        x_ = []
        y_ = []
        for i, j in zip(x, y):
            temp_x, temp_y = util.convert_to_original_size(i, j, ratio_w, ratio_h)
            x_.append(temp_x)
            y_.append(temp_y)
        #x_, y_ = find_target(x_, y_, ratio_w, ratio_h)
        x_, y_ = fitting(x_, y_, ratio_w, ratio_h)

        #util.visualize_points_origin_size(x_[0], y_[0], test_image[0]*255, ratio_w, ratio_h)
        #print(target_lanes)
        #util.visualize_points_origin_size(target_lanes[0], target_h[0], test_image[0]*255, ratio_w, ratio_h)

        result_data = write_result(x_, y_, path)
        progressbar.update(1)
    progressbar.close()

def evaluation_test(test_loss, norm_dist, loader, lane_agent, thresh = p.threshold_point, index= -1, name = None):
    progressbar = tqdm(range(loader.size_test//p.batch_size))
    count = 0
    for test_image, ratio_w, ratio_h, path, target_h, target_lanes, vp_gt in loader.Generate_Test():
        x, y, out_images, for_vis = test(test_loss, norm_dist, vp_gt, lane_agent, test_image, thresh, index=index)
        x_ = []
        y_ = []
        for i, j in zip(x, y):
            temp_x, temp_y = util.convert_to_original_size(i, j, ratio_w, ratio_h)
            x_.append(temp_x)
            y_.append(temp_y)
        #x_, y_ = find_target(x_, y_, ratio_w, ratio_h)
        x_, y_ = fitting(x_, y_, ratio_w, ratio_h)

        # visualize ego lanes
        if for_vis is not None:
            plt.scatter(for_vis[0], for_vis[1], color="red")
            plt.scatter(for_vis[2], for_vis[3], color="blue")

            

            image = test_image[0]*255
            image = np.rollaxis(image, axis=2, start=0)
            image = np.rollaxis(image, axis=2, start=0)  # *255.0
            image = image.astype(np.uint8).copy()
            tmp_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            plt.imshow(tmp_image)
            # plt.show()

        # for i in range(len(for_vis[0])): ]), int(for_vis[1][i])), p.color[1], 1)

        # for k in range(len(y)):
        #     for i, j in zip(x[k], y[k]):
        #         if i > 0:
        #             image = cv2.circle(image, (int(i), int(j)), 2, p.color[1], -1)


        # util.visualize_points(test_image[0]*255, x_[0], y_[0])
        #util.visualize_points_origin_size(x_[0], y_[0], test_image[0]*255, ratio_w, ratio_h)
        #print(target_lanes)
        #util.visualize_points_origin_size(target_lanes[0], target_h[0], test_image[0]*255, ratio_w, ratio_h)
        if count % 30 == 0 and out_images != []:
            cv2.imwrite('./output_img/result_'+str(count)+'.png', out_images[0])

        result_data = write_result(x_, y_, path)
        progressbar.update(1)
        count += 1
    progressbar.close()

############################################################################
## linear interpolation for fixed y value on the test dataset
############################################################################
def find_target(x, y, ratio_w, ratio_h):
    # find exact points on target_h
    out_x = []
    out_y = []
    x_size = p.x_size/ratio_w
    y_size = p.y_size/ratio_h
    for x_batch, y_batch in zip(x,y):
        predict_x_batch = []
        predict_y_batch = []
        for i, j in zip(x_batch, y_batch):
            min_y = min(j)
            max_y = max(j)
            temp_x = []
            temp_y = []
            for h in range(100, 590, 10):
                temp_y.append(h)
                if h < min_y:
                    temp_x.append(-2)
                elif min_y <= h and h <= max_y:
                    for k in range(len(j)-1):
                        if j[k] >= h and h >= j[k+1]:
                            #linear regression
                            if i[k] < i[k+1]:
                                temp_x.append(int(i[k+1] - float(abs(j[k+1] - h))*abs(i[k+1]-i[k])/abs(j[k+1]+0.0001 - j[k])))
                            else:
                                temp_x.append(int(i[k+1] + float(abs(j[k+1] - h))*abs(i[k+1]-i[k])/abs(j[k+1]+0.0001 - j[k])))
                            break
                else:
                    temp_x.append(-2)
            predict_x_batch.append(temp_x)
            predict_y_batch.append(temp_y)
        out_x.append(predict_x_batch)
        out_y.append(predict_y_batch)            
    
    return out_x, out_y

def fitting(x, y, ratio_w, ratio_h):
    out_x = []
    out_y = []
    x_size = p.x_size/ratio_w
    y_size = p.y_size/ratio_h

    for x_batch, y_batch in zip(x,y):
        predict_x_batch = []
        predict_y_batch = []
        for i, j in zip(x_batch, y_batch):
            min_y = min(j)
            max_y = max(j)
            temp_x = []
            temp_y = []

            jj = []
            pre = -100
            for temp in j[::-1]:
                if temp > pre:
                    jj.append(temp)
                    pre = temp
                else:
                    jj.append(pre+0.00001)
                    pre = pre+0.00001
            sp = csaps.CubicSmoothingSpline(jj, i[::-1], smooth=0.0001)

            last = 0
            last_second = 0
            last_y = 0
            last_second_y = 0
            for pts in range(62, -1, -1):
                h = 590 - pts*5 - 1
                temp_y.append(h)
                if h < min_y:
                    temp_x.append(-2)
                elif min_y <= h and h <= max_y:
                    temp_x.append( sp([h])[0] )
                    last = temp_x[-1]
                    last_y = temp_y[-1]
                    if len(temp_x)<2:
                        last_second = temp_x[-1]
                        last_second_y = temp_y[-1]
                    else:
                        last_second = temp_x[-2]
                        last_second_y = temp_y[-2]
                else:
                    if last < last_second:
                        l = int(last_second - float(-last_second_y + h)*abs(last_second-last)/abs(last_second_y+0.0001 - last_y))
                        if l > x_size or l < 0 :
                            temp_x.append(-2)
                        else:
                            temp_x.append(l)
                    else:
                        l = int(last_second + float(-last_second_y + h)*abs(last_second-last)/abs(last_second_y+0.0001 - last_y))
                        if l > x_size or l < 0 :
                            temp_x.append(-2)
                        else:
                            temp_x.append(l)
            predict_x_batch.append(temp_x[::-1])
            predict_y_batch.append(temp_y[::-1])
        out_x.append(predict_x_batch)
        out_y.append(predict_y_batch) 


    return out_x, out_y

############################################################################
## write result
############################################################################
def write_result(x, y, path):
    
    batch_size = len(path)
    save_path = "./output"
    for i in range(batch_size):
        path_detail = path[i].split("/")
        first_folder = path_detail[0]
        second_folder = path_detail[1]
        file_name = path_detail[2].split(".")[0]+".lines.txt"
        if not os.path.exists(save_path+"/"+first_folder):
            os.makedirs(save_path+"/"+first_folder)
        if not os.path.exists(save_path+"/"+first_folder+"/"+second_folder):
            os.makedirs(save_path+"/"+first_folder+"/"+second_folder)      
        with open(save_path+"/"+first_folder+"/"+second_folder+"/"+file_name, "w") as f:  
            for x_values, y_values in zip(x[i], y[i]):
                count = 0
                if np.sum(np.array(x_values)>=0) > 1 : ######################################################
                    for x_value, y_value in zip(x_values, y_values):
                        if x_value >= 0:
                            f.write(str(x_value) + " " + str(y_value) + " ")
                            count += 1
                    if count>1:
                        f.write("\n")


############################################################################
## save result by json form
############################################################################
def save_result(result_data, fname):
    with open(fname, 'w') as make_file:
        for i in result_data:
            json.dump(i, make_file, separators=(',', ': '))
            make_file.write("\n")

############################################################################
## test on the input test image
############################################################################
def test(test_loss, norm_dist, vp_gt, lane_agent, test_images, thresh = p.threshold_point, index= -1):

    result, vp_info, for_vis = lane_agent.predict_lanes_test(test_images, vp_gt)
    torch.cuda.synchronize()
    confidences, offsets, instances = result[index]
    
    num_batch = len(test_images)

    out_x = []
    out_y = []
    out_images = []
    
    # vp detect
    ego_left_pts = []
    ego_right_pts = []
    vp_gt_used = []
    used_idx_lst = []
    
    for i in range(num_batch):
        # test on test data set
        image = deepcopy(test_images[i])
        image =  np.rollaxis(image, axis=2, start=0)
        image =  np.rollaxis(image, axis=2, start=0)*255.0
        image = image.astype(np.uint8).copy()

        confidence = confidences[i].view(p.grid_y, p.grid_x).cpu().data.numpy()

        offset = offsets[i].cpu().data.numpy()
        offset = np.rollaxis(offset, axis=2, start=0)
        offset = np.rollaxis(offset, axis=2, start=0)
        
        instance = instances[i].cpu().data.numpy()
        instance = np.rollaxis(instance, axis=2, start=0)
        instance = np.rollaxis(instance, axis=2, start=0)

        # generate point and cluster
        raw_x, raw_y = generate_result(confidence, offset, instance, thresh)

        # eliminate fewer points
        in_x, in_y = eliminate_fewer_points(raw_x, raw_y)

        # sort points along y 
        in_x, in_y = util.sort_along_y(in_x, in_y)  

        # vp detect까지 성공한 경우만 이미지 저장
        if vp_info[2] is not None and i in vp_info[2]:  # vp detect에 사용한 batch라면
            result_image = util.draw_points(in_x, in_y,  deepcopy(image))
            idx = vp_info[2].index(i)
            result_image = util.draw_points_vp(vp_info[0][idx].cpu().detach().numpy(), vp_info[1][idx].cpu().detach().numpy(), result_image)
            out_images.append(result_image)

        out_x.append(in_x)
        out_y.append(in_y)
        
    if vp_info[0] is not None:
        vp_pred = vp_info[0].cpu().detach().numpy()
        vp_gt_used = vp_info[1].cpu().detach()

        x_pred = vp_pred[:, 0]*p.x_size
        x_gt = np.array(vp_gt_used)[:, 0]*p.x_size
        y_pred = vp_pred[:, 1]*p.y_size
        y_gt = np.array(vp_gt_used)[:, 1]*p.y_size
        test_loss[0] += np.sum(np.abs(x_pred-x_gt))
        test_loss[1] += np.sum(np.abs(y_pred-y_gt))
        test_loss[2] += len(vp_gt_used)
        norm_dist_tmp = np.sqrt((x_pred-x_gt)**2 + (y_pred-y_gt)**2) / p.img_diag
        norm_dist.extend(norm_dist_tmp.tolist())
        # print(x_pred)
        # print(y_pred)
        

    return out_x, out_y,  out_images, for_vis

############################################################################
## eliminate result that has fewer points than threshold
############################################################################
def eliminate_fewer_points(x, y):
    # eliminate fewer points
    out_x = []
    out_y = []
    for i, j in zip(x, y):
        if len(i)>5:
            out_x.append(i)
            out_y.append(j)     
    return out_x, out_y   

############################################################################
## generate raw output
############################################################################
def generate_result(confidance, offsets,instance, thresh):

    mask = confidance > thresh

    grid = p.grid_location[mask]
    offset = offsets[mask]
    feature = instance[mask]

    lane_feature = []
    x = []
    y = []
    for i in range(len(grid)):
        if (np.sum(feature[i]**2))>=0:
            point_x = int((offset[i][0]+grid[i][0])*p.resize_ratio)
            point_y = int((offset[i][1]+grid[i][1])*p.resize_ratio)
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
                
    return x, y

#####################################################
## spline lane
#####################################################
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
            print('cubic spline error_valid')
            # print('xs:', xs)
            # print('ys:', ys)
            return None
        # x_intrp = np.linspace(int(xs.min()), int(xs.max()), int(xs.max())-int(xs.min())+1)
        x_intrp = np.linspace(int(xs.min()), int(xs.max()), 40)
        y_intrp = cs_intrp(x_intrp)
        # x_intrp = -x_intrp
        # x_intrp = x_intrp[::-1]
        # y_intrp = y_intrp[::-1]

        #### interpolation 시각화 ####
        # x_intrp1 = np.linspace(int(xs.min()), int(xs.max()), 40)
        # y_intrp1 = cs_intrp1(x_intrp1)
        # x_intrp1 = -x_intrp1
        # x_intrp1 = x_intrp1[::-1]
        # y_intrp1 = y_intrp1[::-1]

        # x_intrp2 = np.linspace(int(xs.min()), int(xs.max()), 40)
        # y_intrp2 = cs_intrp2(x_intrp2)
        # x_intrp2 = -x_intrp2
        # x_intrp2 = x_intrp2[::-1]
        # y_intrp2 = y_intrp2[::-1]

        # plt.subplot(221).axis([0, 512, 0, 256])
        # plt.scatter(ys[::-1], -xs[::-1])

        # plt.subplot(222).axis([0, 512, 0, 256])
        # plt.scatter(y_intrp1, x_intrp1)

        # plt.subplot(223).axis([0, 512, 0, 256])
        # plt.scatter(y_intrp2, x_intrp2)

        # plt.subplot(224).axis([0, 512, 0, 256])
        # plt.scatter(y_intrp, x_intrp)
        # plt.show()
        ##################################

        x_intrp /= p.y_size
        y_intrp /= p.x_size
        intrp_lane = np.array(list(zip(y_intrp, x_intrp)), dtype='float32')
        return intrp_lane
    else:
        # print("there is no lane valid")
        return None


def save_norm_dist(norm_dist):
    with open('./norm_dist/norm_dist.pkl', 'wb') as f:
        pickle.dump(norm_dist, f)

if __name__ == '__main__':
    Testing()
