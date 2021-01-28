import cv2
import numpy as np
import os
from operator import itemgetter
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from clustering import clustering

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
numPointRank = 100

# Distance Minimum
dist_threshold = 70

def plot_circle(VD3D):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #ax.set_aspect('equal')

    u = np.linspace(0,2*np.pi, 100)
    # u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)

    x = 1 * np.outer(np.cos(u), np.sin(v))
    y = 1 * np.outer(np.sin(u), np.sin(v))
    z = 1 * np.outer(np.ones(np.size(u)), np.cos(v))
    #for i in range(2):
    #    ax.plot_surface(x+random.randint(-5,5), y+random.randint(-5,5), z+random.randint(-5,5),  rstride=4, cstride=4, color='b', linewidth=0, alpha=0.5)
    elev = 1.0
    rot = 1.0 / 180 * np.pi
    ax.plot_surface(x, y, z,  rstride=4, cstride=4, color='y', linewidth=0, alpha=0.5)
    #calculate vectors for "vertical" circle
    a = np.array([-np.sin(elev / 180 * np.pi), 0, np.cos(elev / 180 * np.pi)])
    # a = np.array([-VD3D[0][0], -VD3D[0][1], -VD3D[0][2]])
    # b = np.array([0, 1, 0])
    b = np.array([VD3D[0][0], VD3D[0][1], VD3D[0][2]])
    b = b * np.cos(rot) + np.cross(a, b) * np.sin(rot) + a * np.dot(a, b) * (1 - np.cos(rot))
    #ax.plot(np.sin(u),np.cos(u),0,color='k', linestyle = 'dashed')
    horiz_front = np.linspace(0, np.pi, 100)
    #ax.plot(np.sin(VD3D[0][0]),np.cos(VD3D[0][1]),VD3D[0][2],color='k')
    vert_front = np.linspace(np.pi / 2, 3 * np.pi / 2, 100)
    #ax.plot(a[0] * np.sin(u) + b[0] * np.cos(u), b[1] * np.cos(u), a[2] * np.sin(u) + b[2] * np.cos(u),color='k', linestyle = 'dashed')
    # ax.plot(b[0] * np.sin(u), b[1] * np.cos(u), b[2] * np.cos(u),color='k', linestyle = 'dashed')
    #ax.plot(a[0] * np.sin(vert_front) + b[0] * np.cos(vert_front), b[1] * np.cos(vert_front), a[2] * np.sin(vert_front) + b[2] * np.cos(vert_front),color='k')

    for VD3D_tmp in VD3D:
        ax.scatter(VD3D_tmp[0], VD3D_tmp[1], VD3D_tmp[2], color="r", s=20)

    ax.view_init(elev = elev, azim = 0)


    plt.show()

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

    # change (candidate) vanishing direction 3-d to (candidate) vanishing points (2-d)
    for i in range(len(VD3D)):
        tmp_vp = np.matmul(camera_vec,VD3D[i].transpose())
        tmp_vp = tmp_vp/VD3D[i][2]

        if (tmp_vp[0] >= 0 and tmp_vp[0] <= img_size_x) or (tmp_vp[1] >= img_size_y and tmp_vp[1] <= img_size_y):
           cv2.circle(rgb_img, (int(tmp_vp[0]),int(tmp_vp[1])),3,(0,0,255),-1)

        #print(i , " : ", tmp_vp)
        VP.append(tmp_vp)

        #cv2.circle(rgb_img, (int(tmp_vp[0]),int(tmp_vp[1])),3,(0,0,255),-1)


    #print(VD3D)

    # make score for each candidate vanishing points
    score_matrix = np.zeros((len(VD3D),len(VD3D)))

    #print(score_matrix.shape)
    #print(VP)

    #[103.88763198, 146.3627451 ,   1.        ]
    #[-262.90243902,  -74.68292683,    1.        ]
    # 428.2478


    for i in range(len(VD3D)):
        for j in range(i+1,len(VD3D)):

            # print(i, " : ", VP[i][:2])
            # print(j, " : ", VP[j][:2])
            dist = np.linalg.norm(VP[i][:2]-VP[j][:2],2)
            # print(dist)

            if dist < 1:
                dist = 1

            score_matrix[i][j] = 1/dist
            score_matrix[j][i] = score_matrix[i][j]

    # score_vector = sum(score_matrix)/sum(sum(sum(score_matrix)))
    sum_score_matrix = score_matrix.sum(axis = 1)
    sum_score_vector = sum_score_matrix.sum()

    score_vector = sum_score_matrix/sum_score_vector

    #print(score_vector.shape)

    #print(score_vector)

    sorted_score_vector = np.sort(score_vector)[::-1]

    score_idx = []

    for i in range(len(sorted_score_vector)):
        tmp_index = np.where(score_vector == sorted_score_vector[i])
        #print(tmp_index[0][0])
        score_idx.append(tmp_index[0][0])
    #print(score_idx)
    # print(sorted_score_vector)
    #print(score_matrix)

    # extract top N candidate vanishing points
    if len(VP) < numPointRank:
        numPointRank = len(VP)

    topN_VP = []

    for i in range(numPointRank):
        topN_VP.append(VP[score_idx[i]][0:2])
        print(topN_VP[i])
    # print(topN_VP)



    a = clustering(topN_VP)

    print("a", a)
    #






                #print(tmp_product_norm[0])


            #print(tmp_product_norm)
            #VD3D.append()




        #cv2.line(rgb_img, (x1,y1), (x2,y2), (0,0,255),2)



        #cv2.imshow("RGB",rgb_img)
        #cv2.imshow("CANNY",edges)

        #cv2.waitKey(300000)

    cv2.imshow("CANNY", rgb_img)
    cv2.imshow("LINE",line_img)
    #plot_circle(VD3D)


    cv2.waitKey(300000)
