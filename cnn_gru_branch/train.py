#############################################################################################################
##
##  Source code for training. In this source code, there are initialize part, training part, ...
##
#############################################################################################################

import cv2
import torch
import visdom
#import sys
#sys.path.append('/home/kym/research/autonomous_car_vision/lanedection/code/')
import agent
import numpy as np
from data_loader_copy import Generator
from parameters import Parameters
import test
import evaluation
import util
import os
import copy
import csv

p = Parameters()

###############################################################
##
## Training
## 
###############################################################
def Training():
    print('Training')

    ####################################################################
    ## Hyper parameter
    ####################################################################
    print('Initializing hyper parameter')

    # vis = visdom.Visdom()
    # loss_window = vis.line(X=torch.zeros((1,)).cpu(),
    #                        Y=torch.zeros((1)).cpu(),
    #                        opts=dict(xlabel='epoch',
    #                                  ylabel='Loss',
    #                                  title='Training Loss',
    #                                  legend=['Loss']))
    
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
        lane_agent.load_weights_default(296, "tensor(1.6947)")
        # lane_agent.load_weights(24, "tensor(1.6059)", "0.01302009355276823")

    ##############################
    ## Check GPU
    ##############################
    print('Setup GPU mode')
    if torch.cuda.is_available():
        lane_agent.cuda()
        #torch.backends.cudnn.benchmark=True

    ##############################
    ## Loop for training
    ##############################
    print('Training loop')
    step = 0
    sampling_list = None
    for epoch in range(p.n_epoch):
        lane_agent.training_mode()
        for inputs, target_lanes, target_h, test_image, data_list, vp_gt in loader.Generate(sampling_list):

            #util.visualize_points(inputs[0], target_lanes[0], target_h[0])
            #training
            loss_p, loss_gru = lane_agent.train(inputs, target_lanes, target_h, epoch, lane_agent, data_list, vp_gt, step)
            torch.cuda.synchronize()
            loss_p = loss_p.cpu().data

            # if step%500 == 0:
                # print("epoch : " + str(epoch))
                # print("step : " + str(step))
                # print("loss_gru :", loss_gru)
                # vis.line(
                #     X=torch.ones((1, 1)).cpu() * int(step/500),
                #     Y=torch.Tensor([loss_p]).unsqueeze(0).cpu(),
                #     win=loss_window,
                #     update='append')
                
            # if step%100 == 0:
            #     print("epoch : " + str(epoch))
            #     print("step : " + str(step))
            #     print("loss_gru :", loss_gru)

            if step % 1000 == 0:
                print("epoch : " + str(epoch))
                print("step : " + str(step))
                print("loss_gru :", loss_gru)
                lane_agent.save_model(int(step/1000), loss_p, loss_gru)
                # testing(vp_gt, lane_agent, test_image, step, loss_p, loss_gru)
            step += 1

        sampling_list = copy.deepcopy(lane_agent.get_data_list())
        lane_agent.sample_reset()

        #evaluation
        if epoch%1 == 0:
            print("evaluation")
            lane_agent.evaluate_mode()
            # lane_agent.save_model(int(step/100), loss_p, loss_gru)
            val_loss = [0,0,0]
            test.evaluation(val_loss, loader, lane_agent, name="test_result_"+str(epoch)+".json")
            with open('./output_gru/val_loss_512_256.csv', 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch, val_loss[0]/val_loss[2], val_loss[1]/val_loss[2]])
        # #evaluation
        # if epoch%1 == 0:
        #     print("evaluation")
        #     lane_agent.evaluate_mode()
        #     th_list = [0.9]
        #     index = [3]
        #     lane_agent.save_model(int(step/100), loss_p, loss_gru)

        #     for idx in index:
        #         print("generate result")
        #         test.evaluation(loader, lane_agent, index = idx, name="test_result_"+str(epoch)+"_"+str(idx)+".json")
        #         name = "epoch_idx_"+str(epoch) + str(idx) + str(step/100)
        #         # os.system("sh /home/kym/research/autonomous_car_vision/lane_detection/code/ITS/CuLane/evaluation_code/SCNN_Pytorch/utils/lane_evaluation/CULane/Run.sh " + name)

        if int(step)>700000:
            break


def testing(vp_gt, lane_agent, test_image, step, loss, loss_gru):
    lane_agent.evaluate_mode()

    _, _, ti = test.test([0,0,0], vp_gt, lane_agent, np.array([test_image]))

    cv2.imwrite('test_result/result_'+str(step)+'_'+str(loss)+'_'+str(loss_gru)+'.png', ti[0])

    lane_agent.training_mode()

    
if __name__ == '__main__':
    Training()

