import cv2
import numpy as np
import os
from operator import itemgetter

# data folder
path_dir = "../v_data"

# raw name
raw_list = os.listdir(path_dir)

# remove .png
file_list = [x[:-4] for x in raw_list]

rgb_list = []
line_list = []

# add path + name + format
for x in file_list:
    if "line" in x:
        line_list.append(path_dir+"/{}.png".format(x))
    elif "rgb" in x:
        rgb_list.append(path_dir+"/{}.png".format(x))

line_list.sort()
rgb_list.sort()

num_list = len(rgb_list)

# HoughLinesP parameters
threshold = 50
# minLineLength -> height / 2
minLineLength = 50
maxLineGap = 50

# Distance Minimum
dist_threshold = 70

# Distance Calculator
def dist_calc(line):
    x1, y1, x2, y2 = line[0]

    dist = (x1-x2)**2+(y1-y2)**2

    return np.sqrt(dist)

# Normalization
# def line_norm(real_lines):
#     for real_line in real_lines:
#         print(real_line)

# Angle Calculator
#def ang_calc(line)

# 일단 귀찮으니 함수화 x
# Main Loop

for x in range(num_list):
    rgb_img = cv2.imread(rgb_list[x],cv2.IMREAD_COLOR)
    line_img = cv2.imread(line_list[x],cv2.IMREAD_GRAYSCALE)

    (img_size_x, img_size_y) = rgb_img.shape[:2]

    print("* size_x : ", img_size_x, end = '  ')
    print("* size_y : ", img_size_y)

    real_lines = []
    #edges = cv2.Canny(line_img, 70, 150, apertureSize = 7)

    #lines = cv2.HoughLinesP(line_img, 1, np.pi/180, threshold, minLineLength)
    lines = cv2.HoughLinesP(line_img, 1, np.pi/180, threshold, minLineLength)
    print("Number of Line : ", len(lines))

    # exclude vertical, horizon lines and short lines
    for line in lines:

        x1, y1, x2, y2 = line[0]

        angle = 0.0
        angle = np.arctan2(y2-y1,x2-x1) * 180 / np.pi
        dist = dist_calc(line)

        #print(angle)
        #print(dist)

        if x1 == x2 or y1 == y2 or (angle < 3 and angle > -3):
            continue
            #cv2.line(rgb_img, (x1,y1), (x2,y2), (255,0,0),2)
        #Filtering out using Distance
        elif dist > dist_threshold:
            tmp = np.array([[dist]])
            tmp_line = np.concatenate((line[0],tmp[0]))
            #new_line = np.array([tmp_line])
            # new_line = list(tmp_line)
            new_line = tmp_line
            #print(new_line)

            real_lines.append(new_line)
            #print(new_line)
            cv2.line(rgb_img, (x1,y1), (x2,y2), (0,255,0),2)
        else:
            continue

    print("Number of Filtered Line : ", len(real_lines))

    # set camera vector
    camera_vec = np.array([[(img_size_x+img_size_y)/2, 0, img_size_x/2],[0, (img_size_x+img_size_y)/2,img_size_y/2],[0, 0, 1]])

    # print(type(real_lines[0]))
    # print(np.shape(real_lines))
    # print(real_lines)

    # sort line distance
    sorted_lines = sorted(real_lines, key=itemgetter(4), reverse = True)
    #sorted_lines = real_lines.sort(key=itemgetter(4))

    #print(sorted_lines)

    interpretation_plane = []
    # make interpretation planes between lines
    for lines in sorted_lines:
        x1,y1,x2,y2,dist = lines

        point_1 = np.array([[x1], [y1], [1]])
        line_point_1 = np.linalg.inv(camera_vec)
        line_point_1 = np.matmul(line_point_1, point_1)

        #print(line_point_1)

        point_2 = np.array([[x2], [y2], [1]])
        line_point_2 = np.linalg.inv(camera_vec)
        line_point_2 = np.matmul(line_point_2, point_2)

        #print(line_point_2)

        cross_two_line = np.cross(line_point_1.transpose(), line_point_2.transpose())

        #print(cross_two_line)

        interpretation_plane.append(cross_two_line/np.sqrt(cross_two_line[0][0]**2+cross_two_line[0][1]**2+cross_two_line[0][2]**2))

        #print(interpretation_plane)


    VD3D = []

    # make (candidate) vanishing directions between interpretation planes
    tmp_count = 0
    for i in range(0,len(interpretation_plane)):
        for j in range(i+1,len(interpretation_plane)):
            tmp_count += 1
            tmp_product = np.cross(interpretation_plane[i],interpretation_plane[j])
            #print(tmp_product.shape)
            #print(tmp_product)

            #print(tmp_product[0][1])
            tmp_product_norm = tmp_product/np.sqrt(tmp_product[0][0]**2+tmp_product[0][1]**2+tmp_product[0][2]**2);

            if tmp_product_norm[0][2] < 0:
                tmp_product_norm[0] = -tmp_product_norm[0]

            VD3D.append(tmp_product_norm[0])

    #print(len(VD3D))


    VP = []

    # chane (candidate) vanishing direction 3-d to (candidate) vanishing points (2-d)
    for i in range(len(VD3D)):
        tmp_vp = np.matmul(camera_vec,VD3D[i].transpose())
        tmp_vp = tmp_vp/VD3D[i][2]

        if tmp_vp[0] > img_size_x or tmp_vp[1] > img_size_y:
            continue

        #print(i , " : ", tmp_vp)
        VP.append(tmp_vp)

        cv2.circle(rgb_img, (int(tmp_vp[0]),int(tmp_vp[1])),3,(0,0,255),-1)


    # make score for each candidate vanishing points
    score_matrix = np.zeros((len(VD3D),len(VD3D)))

    print(len(VD3D))
    print(score_matrix)

    # for i in range(len(VD3D)):
    #     for j in range(i,len(VD3D)):
    #         dist =





                #print(tmp_product_norm[0])


            #print(tmp_product_norm)
            #VD3D.append()




        #cv2.line(rgb_img, (x1,y1), (x2,y2), (0,0,255),2)



        #cv2.imshow("RGB",rgb_img)
        #cv2.imshow("CANNY",edges)

        #cv2.waitKey(300000)


    cv2.imshow("RGB",rgb_img)
    cv2.imshow("LINE",line_img)
    #cv2.imshow("CANNY",edges)

    cv2.waitKey(300000)
