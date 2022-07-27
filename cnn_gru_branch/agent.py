#########################################################################
##
## train agent that has some utility for training and saving.
##
#########################################################################

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
from torch.autograd import Variable
from hourglass_network import lane_detection_network

from gru_cnn_vp import GRU_CNN
from torch.autograd import Function as F
from parameters import Parameters
import math
import util
import hard_sampling
import matplotlib.pyplot as plt

from flopco import FlopCo
from torchinfo import summary

############################################################
##
## agent for lane detection
##
############################################################


class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()

    def forward(self, x, y):
        eps = 1e-8
        criterion = nn.MSELoss()
        loss = torch.sqrt(criterion(x, y) + eps)
        return loss

class Agent(nn.Module):

    #####################################################
    ## Initialize
    #####################################################
    def __init__(self):
        super(Agent, self).__init__()
        
        self.gru_loss_former = 0

        self.p = Parameters()

        self.lane_detection_network = lane_detection_network()

        # self.gru = GRU(input_size=2, hidden_size=24, num_layers=2, output_size=2).cuda()

        self.setup_optimizer()

        self.current_epoch = 0

        self.hard_sampling = hard_sampling.hard_sampling()

        self.criterion_gru = RMSELoss()
        self.criterion_gru_l = RMSELoss()
        self.criterion_gru_r = RMSELoss()
        # self.criterion_gru = nn.MSELoss()
        # self.optimizer_branch = torch.optim.SGD(
        #     self.lane_detection_network.parameters(), lr=0.00005, momentum=0.9)

        print("model parameters: ")
        print(self.count_parameters(self.lane_detection_network))
    
    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def setup_optimizer(self):
        self.lane_detection_optim = torch.optim.Adam(self.lane_detection_network.parameters(),
                                                    lr=self.p.l_rate,
                                                    weight_decay=self.p.weight_decay)
        self.lane_detection_optim_branch = torch.optim.Adam(self.lane_detection_network.parameters(),
                                                     lr=self.p.l_rate,
                                                     weight_decay=self.p.weight_decay)

    #####################################################
    ## Make ground truth for key point estimation
    #####################################################
    def make_ground_truth_point(self, target_lanes, target_h):

        target_lanes, target_h = util.sort_batch_along_y(target_lanes, target_h)

        ground = np.zeros((len(target_lanes), 3, self.p.grid_y, self.p.grid_x))
        ground_binary = np.zeros((len(target_lanes), 1, self.p.grid_y, self.p.grid_x))

        for batch_index, batch in enumerate(target_lanes):
            for lane_index, lane in enumerate(batch):
                for point_index, point in enumerate(lane):
                    if point > 0:
                        x_index = int(point/self.p.resize_ratio)
                        y_index = int(target_h[batch_index][lane_index][point_index]/self.p.resize_ratio)
                        ground[batch_index][0][y_index][x_index] = 1.0
                        ground[batch_index][1][y_index][x_index]= (point*1.0/self.p.resize_ratio) - x_index
                        ground[batch_index][2][y_index][x_index] = (target_h[batch_index][lane_index][point_index]*1.0/self.p.resize_ratio) - y_index
                        ground_binary[batch_index][0][y_index][x_index] = 1

        return ground, ground_binary


    #####################################################
    ## Make ground truth for instance feature
    #####################################################
    def make_ground_truth_instance(self, target_lanes, target_h):

        ground = np.zeros((len(target_lanes), 1, self.p.grid_y*self.p.grid_x, self.p.grid_y*self.p.grid_x))

        for batch_index, batch in enumerate(target_lanes):
            temp = np.zeros((1, self.p.grid_y, self.p.grid_x))
            lane_cluster = 1
            for lane_index, lane in enumerate(batch):
                previous_x_index = 0
                previous_y_index = 0
                for point_index, point in enumerate(lane):
                    if point > 0:
                        x_index = int(point/self.p.resize_ratio)
                        y_index = int(target_h[batch_index][lane_index][point_index]/self.p.resize_ratio)
                        temp[0][y_index][x_index] = lane_cluster
                    if previous_x_index != 0 or previous_y_index != 0: #interpolation make more dense data
                        temp_x = previous_x_index
                        temp_y = previous_y_index
                        while False:      ###############################################false
                            delta_x = 0
                            delta_y = 0
                            temp[0][temp_y][temp_x] = lane_cluster
                            if temp_x < x_index:
                                temp[0][temp_y][temp_x+1] = lane_cluster
                                delta_x = 1
                            elif temp_x > x_index:
                                temp[0][temp_y][temp_x-1] = lane_cluster
                                delta_x = -1
                            if temp_y < y_index:
                                temp[0][temp_y+1][temp_x] = lane_cluster
                                delta_y = 1
                            elif temp_y > y_index:
                                temp[0][temp_y-1][temp_x] = lane_cluster
                                delta_y = -1
                            temp_x += delta_x
                            temp_y += delta_y
                            if temp_x == x_index and temp_y == y_index:
                                break
                    if point > 0:
                        previous_x_index = x_index
                        previous_y_index = y_index
                lane_cluster += 1

            for i in range(self.p.grid_y*self.p.grid_x): #make gt
                temp = temp[temp>-1]
                gt_one = deepcopy(temp)
                if temp[i]>0:
                    gt_one[temp==temp[i]] = 1   #same instance
                    if temp[i] == 0:
                        gt_one[temp!=temp[i]] = 3 #different instance, different class
                    else:
                        gt_one[temp!=temp[i]] = 2 #different instance, same class
                        gt_one[temp==0] = 3 #different instance, different class
                    ground[batch_index][0][i] += gt_one

        return ground

    #####################################################
    ## train
    #####################################################
    def train(self, inputs, target_lanes, target_h, epoch, agent, data_list, vp_gt, step):
        point_loss, gru_loss, for_vis = self.train_point(inputs, target_lanes, target_h, epoch, data_list, vp_gt, step)
        return point_loss, gru_loss, for_vis

    #####################################################
    ## compute loss function and optimize
    #####################################################
    def train_point(self, inputs, target_lanes, target_h, epoch, data_list, vp_gt, step):
        real_batch_size = len(target_lanes)
        torch.autograd.set_detect_anomaly(True)
        # print(vp_gt)
        #generate ground truth
        ground_truth_point, ground_binary = self.make_ground_truth_point(target_lanes, target_h)
        ground_truth_instance = self.make_ground_truth_instance(target_lanes, target_h)
        #util.visualize_gt(ground_truth_point[0], ground_truth_instance[0], 0, inputs[0])

        # convert numpy array to torch tensor
        ground_truth_point = torch.from_numpy(ground_truth_point).float()
        ground_truth_point = Variable(ground_truth_point).cuda()
        ground_truth_point.requires_grad=False

        ground_binary = torch.LongTensor(ground_binary.tolist()).cuda()
        ground_binary.requires_grad=False

        ground_truth_instance = torch.from_numpy(ground_truth_instance).float()
        ground_truth_instance = Variable(ground_truth_instance).cuda()
        ground_truth_instance.requires_grad=False

        #util.visualize_gt(ground_truth_point[0], ground_truth_instance[0], inputs[0])

        # update lane_detection_network
        result, attentions, vp_info, for_vis = self.predict_lanes(inputs, vp_gt)
        lane_detection_loss = 0
        exist_condidence_loss = 0
        nonexist_confidence_loss = 0
        offset_loss = 0
        x_offset_loss = 0
        y_offset_loss = 0
        sisc_loss = 0
        disc_loss = 0
        gru_loss = 0
        gru_loss_kp_left = 0
        gru_loss_kp_right = 0
        gru_loss_kp = 0
        
        # hard sampling ##################################################################
        confidance, offset, feature = result[-1]
        pred_vp = vp_info[0]
        vp_gt_used = vp_info[1]
        hard_loss = 0
        if pred_vp is not None:
            gru_loss = self.criterion_gru(pred_vp, vp_gt_used) / len(vp_gt_used)
            # if p.batch_size/3 > len(vp_gt_used):
            #     gru_loss *= 1.5
            # elif p.batch_size*2/3 > len(vp_gt_used):
            #     gru_loss *= 1.2
            # self.gru_loss_former = gru_loss
        else: gru_loss = self.gru_loss_former * 1.8


        for i in range(real_batch_size):
            
            # lane key point 생성
            confidence_ = confidance[i].view(p.grid_y, p.grid_x).cpu().data.numpy()
            # confidence.shape <- [32, 64]
            # confidence map을 시각화한것이 히트맵

            offset_ = offset[i].cpu().data.numpy()
            offset_ = np.rollaxis(offset_, axis=2, start=0)
            offset_ = np.rollaxis(offset_, axis=2, start=0)
            # offset.shape < - (32, 64, 2)

            instance_ = feature[i].cpu().data.numpy()
            instance_ = np.rollaxis(instance_, axis=2, start=0)
            instance_ = np.rollaxis(instance_, axis=2, start=0)
            # instance.shape <- (32, 64, 4)

            


            confidance_gt = ground_truth_point[i, 0, :, :]
            confidance_gt = confidance_gt.view(1, self.p.grid_y, self.p.grid_x)
            hard_loss =  hard_loss +\
                torch.sum( (1-confidance[i][confidance_gt==1])**2 )/\
                (torch.sum(confidance_gt==1)+1)

            target = confidance[i][confidance_gt==0]
            hard_loss =  hard_loss +\
				torch.sum( ( target[target>0.01] )**2 )/\
				(torch.sum(target>0.01)+1)

            node = hard_sampling.sampling_node(loss = hard_loss.cpu().data, data = data_list[i], previous_node = None, next_node = None)
            self.hard_sampling.insert(node)

        for (confidance, offset, feature) in result:

            #exist confidance loss##########################
            confidance_gt = ground_truth_point[:, 0, :, :]
            confidance_gt = confidance_gt.view(real_batch_size, 1, self.p.grid_y, self.p.grid_x)
            a = confidance_gt[0][confidance_gt[0]==1] - confidance[0][confidance_gt[0]==1]
            exist_condidence_loss =  exist_condidence_loss +\
				torch.sum( (1-confidance[confidance_gt==1])**2 )/\
				(torch.sum(confidance_gt==1)+1)

            #non exist confidance loss##########################
            target = confidance[confidance_gt==0]
            nonexist_confidence_loss =  nonexist_confidence_loss +\
				torch.sum( ( target[target>0.01] )**2 )/\
				(torch.sum(target>0.01)+1)

            #offset loss ##################################
            offset_x_gt = ground_truth_point[:, 1:2, :, :]
            offset_y_gt = ground_truth_point[:, 2:3, :, :]

            predict_x = offset[:, 0:1, :, :]
            predict_y = offset[:, 1:2, :, :]

            offset_loss = offset_loss + \
			            torch.sum( (offset_x_gt[confidance_gt==1] - predict_x[confidance_gt==1])**2 )/\
				        (torch.sum(confidance_gt==1)+1) + \
			            torch.sum( (offset_y_gt[confidance_gt==1] - predict_y[confidance_gt==1])**2 )/\
				        (torch.sum(confidance_gt==1)+1)

            #compute loss for similarity #################
            feature_map = feature.view(real_batch_size, self.p.feature_size, 1, self.p.grid_y*self.p.grid_x)
            feature_map = feature_map.expand(real_batch_size, self.p.feature_size, self.p.grid_y*self.p.grid_x, self.p.grid_y*self.p.grid_x)#.detach()

            point_feature = feature.view(real_batch_size, self.p.feature_size, self.p.grid_y*self.p.grid_x,1)
            point_feature = point_feature.expand(real_batch_size, self.p.feature_size, self.p.grid_y*self.p.grid_x, self.p.grid_y*self.p.grid_x)#.detach()

            distance_map = (feature_map-point_feature)**2 
            distance_map = torch.sum( distance_map, dim=1 ).view(real_batch_size, 1, self.p.grid_y*self.p.grid_x, self.p.grid_y*self.p.grid_x)
            
            # same instance
            sisc_loss = sisc_loss+\
				torch.sum(distance_map[ground_truth_instance==1])/\
				torch.sum(ground_truth_instance==1)

            # different instance, same class
            disc_loss = disc_loss + \
				torch.sum((self.p.K1-distance_map[ground_truth_instance==2])[(self.p.K1-distance_map[ground_truth_instance==2]) > 0])/\
				torch.sum(ground_truth_instance==2)

        #attention loss
        attention_loss = 0
        source = attentions[:-1]
        m = nn.Softmax(dim=0)
        
        for i in range(real_batch_size):
            target = torch.sum((attentions[-1][i].data)**2, dim=0).view(-1) 
            #target = target/torch.max(target)
            # print(len(target))
            target = m(target)
            for j in source:
                s = torch.sum(j[i]**2, dim=0).view(-1)
                attention_loss = attention_loss + torch.sum( (m(s) - target)**2 )/(len(target)*real_batch_size)

        lane_detection_loss = lane_detection_loss + self.p.constant_exist*exist_condidence_loss
        lane_detection_loss = lane_detection_loss + self.p.constant_nonexist*nonexist_confidence_loss
        lane_detection_loss = lane_detection_loss + self.p.constant_offset*offset_loss
        lane_detection_loss = lane_detection_loss + self.p.constant_alpha*sisc_loss
        lane_detection_loss = lane_detection_loss + self.p.constant_beta*disc_loss + 0.00001*torch.sum(feature**2)
        lane_detection_loss = lane_detection_loss + self.p.constant_attention*attention_loss
        lane_detection_loss = lane_detection_loss + self.p.constant_gru_loss*gru_loss

        # print("######################################################################")
        # print("seg loss")
        # print("same instance loss: ", sisc_loss.data)
        # print("different instance loss: ", disc_loss.data)

        # print("point loss")
        # print("exist loss: ", exist_condidence_loss.data)
        # print("non-exit loss: ", nonexist_confidence_loss.data)
        # print("offset loss: ", offset_loss.data)

        # print("attention loss")
        # print("attention loss: ", attention_loss.data)

        # print("--------------------------------------------------------------------")
        # print("total loss: ", lane_detection_loss.data)

        # with torch.autograd.set_detect_anomaly(True):
        if step % 3 == 0 and step != 0 and pred_vp is not None:
            try:

                gru_loss_kp_left = self.criterion_gru_l(vp_info[5][0], vp_info[3][:,-3:,:]) # left
                gru_loss_kp_right = self.criterion_gru_r(vp_info[5][1], vp_info[4][:,-3:,:]) # right
                gru_loss_kp += gru_loss_kp_left
                gru_loss_kp += gru_loss_kp_right

                self.lane_detection_optim_branch.load_state_dict(self.lane_detection_optim.state_dict())
                self.lane_detection_optim_branch.zero_grad()
                gru_loss_kp.backward()
                self.lane_detection_optim_branch.step()
                self.lane_detection_optim.load_state_dict(self.lane_detection_optim_branch.state_dict())
                print(gru_loss_kp)
            except Exception as e:
                print("now training the branch")
                print(e)
        else:
            try:
                self.lane_detection_optim.zero_grad()
                lane_detection_loss.backward()   #divide by batch size
                self.lane_detection_optim.step()
            except Exception as e:
                print("now training main")
                print(e)

        del confidance, offset, feature
        del ground_truth_point, ground_binary, ground_truth_instance
        del feature_map, point_feature, distance_map
        del exist_condidence_loss, nonexist_confidence_loss, offset_loss, sisc_loss, disc_loss
        del gru_loss_kp_right, gru_loss_kp_left

        trim = 180
        if epoch>0 and self.current_epoch != epoch:
            self.current_epoch = epoch
            if epoch == 1-trim:
                self.p.l_rate = 0.0005
                self.setup_optimizer()
            elif epoch == 2-trim:
                self.p.l_rate = 0.0002
                self.setup_optimizer()
            elif epoch == 3-trim:
                self.p.l_rate = 0.0001
                self.setup_optimizer()
            elif epoch == 5-trim:
                self.p.l_rate = 0.00005
                self.setup_optimizer()
            elif epoch == 7-trim:
                self.p.l_rate = 0.00002
                self.setup_optimizer()
            elif epoch == 9-trim:
                self.p.l_rate = 0.00001
                self.setup_optimizer()
            elif epoch == 11-trim:
                self.p.l_rate = 0.000005
                self.setup_optimizer()
            elif epoch == 13-trim:
                self.p.l_rate = 0.000002
                self.setup_optimizer()
            elif epoch == 15-trim:
                self.p.l_rate = 0.000001
                self.setup_optimizer()
            elif epoch == 21-trim:  
                self.p.l_rate = 0.0000001
                self.setup_optimizer()
        if pred_vp is None:
            return lane_detection_loss, 0, None
        else:
            return lane_detection_loss, gru_loss.item(), for_vis
    #####################################################
    ## spline lane
    #####################################################
    def spline_lane(self, pt):
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
                print('cubic spline error') 
                print('xs:', xs)
                print('ys:', ys)
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
            # print("there is no lane")
            return None

    #####################################################
    ## predict lanes
    #####################################################
    def predict_lanes(self, inputs, vp_gt):
        inputs = torch.from_numpy(inputs).float() 
        inputs = Variable(inputs).cuda()

        return self.lane_detection_network(inputs, vp_gt)

    #####################################################
    ## predict lanes in test
    #####################################################
    def predict_lanes_test(self, inputs, vp_gt):
        inputs = torch.from_numpy(inputs).float() 
        inputs = Variable(inputs).cuda()
        outputs, features, vp_info, for_vis = self.lane_detection_network(inputs, vp_gt)
        return outputs, vp_info, for_vis

    #####################################################
    ## Training mode
    #####################################################                                                
    def training_mode(self):
        self.lane_detection_network.train()

    #####################################################
    ## evaluate(test mode)
    #####################################################                                                
    def evaluate_mode(self):
        self.lane_detection_network.eval()

    #####################################################
    ## Setup GPU computation
    #####################################################                                                
    def cuda(self):
        GPU_NUM = 0
        device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
        torch.cuda.set_device(device)
        self.lane_detection_network.cuda()

    #####################################################
    ## Load save file
    #####################################################
    def load_weights(self, epoch, loss, gru_loss):
        self.lane_detection_network.load_state_dict(
            torch.load(self.p.model_path+str(epoch)+'_'+str(loss)+'_'+ str(gru_loss) +'_'+'lane_detection_network.pkl', map_location='cuda:0'),False
        )


        # stats = FlopCo(self.lane_detection_network, img_size=(1, 3, 256, 512),
        #                instances=[nn.Conv2d, nn.Linear,
        #                           nn.BatchNorm2d, nn.ReLU, nn.Conv1d]
        #                )
        # print(stats)

        # summary(self.lane_detection_network, input_size=(1,3,256,512), verbose=1)


    def load_weights_default(self, epoch, loss):
        self.lane_detection_network.load_state_dict(
            torch.load(self.p.model_path+str(epoch)+'_'+str(loss)+'_'+'lane_detection_network.pkl', map_location='cuda:0'),False
        )
        # stats = FlopCo(self.lane_detection_network, img_size=(1, 3, 256, 512), device='cuda:0')


        # print(stats.total_macs, stats.relative_flops)


    #####################################################
    ## Save model
    #####################################################
    def save_model(self, epoch, loss, loss_gru):
        torch.save(
            self.lane_detection_network.state_dict(),
            self.p.save_path+str(epoch)+'_'+str(loss)+'_'+str(loss_gru)+'_'+'lane_detection_network.pkl'
        )

    def get_data_list(self):
        return self.hard_sampling.get_list()

    def sample_reset(self):
        self.hard_sampling = hard_sampling.hard_sampling()

    def eliminate_fewer_points(self, x, y):
        # eliminate fewer points
        out_x = []
        out_y = []
        for i, j in zip(x, y):
            if len(i)>2:
                out_x.append(i)
                out_y.append(j)
        return out_x, out_y

    def generate_result(self, confidance, offsets,instance, thresh):
    
        mask = confidance > thresh
        # print(confidance.shape)
        # print(offsets.shape)
        # print(instance.shape)
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
