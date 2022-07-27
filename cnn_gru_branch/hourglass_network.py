#########################################################################
##
## Structure of network.
##
#########################################################################
import torch
import torch.nn as nn
from util_hourglass import *
from bridge import bridge
from gru_cnn_vp import GRU_CNN

####################################################################
##
## lane_detection_network
##
####################################################################
class lane_detection_network(nn.Module):
    def __init__(self):
        super(lane_detection_network, self).__init__()

        self.resizing = resize_layer(3, 128)

        #feature extraction
        self.layer1 = hourglass_block(128, 128)
        self.layer2 = hourglass_block(128, 128)
        self.layer3 = hourglass_block(128, 128)
        self.layer4 = hourglass_block(128, 128)
        self.gru_cnn = GRU_CNN(input_size=6, hidden_size=24, num_layers=2, output_size=2).cuda()


    def forward(self, inputs, vp_gt):
    # def forward(self, inputs):
    #     vp_gt = [221.01753737, 129.22843748]
        # print(vp_gt)
        #feature extraction
        out = self.resizing(inputs)
        result1, out, feature1 = self.layer1(out)
        result2, out, feature2 = self.layer2(out)   
        result3, out, feature3 = self.layer3(out)
        result4, out, feature4 = self.layer4(out)
        
        # vp detect
        left, right, vp_gt_used, vp_batch_idx, img, label_left, label_right, for_vis = bridge(result4, vp_gt)

        # print(img.shape)
        if left is not None:
            pred_vp, branch = self.gru_cnn(left.cuda(), right.cuda(), img.cuda())
        else:
            # left = torch.tensor([[[0.5039, 0.5078, 0.5118, 0.5703, 0.5750, 0.5797],
            #                       [0.5196, 0.5235, 0.5274, 0.5891, 0.5939, 0.5986],
            #                       [0.5353, 0.5392, 0.5431, 0.6080, 0.6127, 0.6174],
            #                       [0.5510, 0.5566, 0.5637, 0.6268, 0.6315, 0.6362],
            #                       [0.5719, 0.5758, 0.5797, 0.6456, 0.6503, 0.6550],
            #                       [0.5883, 0.5930, 0.5972, 0.6645, 0.6692, 0.6739],
            #                       [0.6043, 0.6078, 0.6113, 0.6833, 0.6880, 0.6927],
            #                       [0.6177, 0.6207, 0.6237, 0.7021, 0.7068, 0.7115],
            #                       [0.6298, 0.6354, 0.6424, 0.7210, 0.7257, 0.7304],
            #                       [0.6496, 0.6531, 0.6566, 0.7398, 0.7445, 0.7492]]]).type(torch.FloatTensor)
            # right = torch.tensor([[[0.3086, 0.3011, 0.2936, 0.6328, 0.6384, 0.6440],
            #                        [0.2809, 0.2746, 0.2680, 0.6552, 0.6609, 0.6665],
            #                        [0.2501, 0.2434, 0.2385, 0.6777, 0.6833, 0.6889],
            #                        [0.2241, 0.2166, 0.2100, 0.7001, 0.7057, 0.7113],
            #                        [0.1974, 0.1876, 0.1798, 0.7226, 0.7282, 0.7338],
            #                        [0.1670, 0.1606, 0.1542, 0.7450, 0.7506, 0.7562],
            #                        [0.1408, 0.1341, 0.1273, 0.7674, 0.7730, 0.7786],
            #                        [0.1139, 0.1072, 0.1005, 0.7899, 0.7955, 0.8011],
            #                        [0.0873, 0.0807, 0.0741, 0.8123, 0.8179, 0.8235],
            #                        [0.0608, 0.0542, 0.0476, 0.8347, 0.8403, 0.8460]]]).type(torch.FloatTensor)
            # img = torch.zeros(1,1,256,512).type(torch.FloatTensor)
            # self.gru_cnn(left.cuda(), right.cuda(), img.cuda())
            pred_vp = None
            branch = None



        return [result1, result2, result3, result4], [feature1, feature2, feature3, feature4], [pred_vp, vp_gt_used, vp_batch_idx, label_left, label_right, branch], for_vis
        #return [result2], [feature2]
