import numpy as np
import cv2

color_ = [(255,0,0), (0,255,0), (0,0,255), (255,100,100), (200,120,80)]
radius_ = 2
thickness_ = 2

def draw_points(img, lanes):
    for lane_idx, lane in enumerate(lanes):
        for xy in lane:
            cv2.circle(img, (int(xy[0]), int(xy[1])), radius = radius_, color = color_[lane_idx], thickness = thickness_)

    return img
