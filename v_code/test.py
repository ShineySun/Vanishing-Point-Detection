import numpy as np

img_size_x = 320
img_size_y = 320

camera = np.array([[(img_size_x+img_size_y)/2, 0, img_size_x/2],[0, (img_size_x+img_size_y)/2,img_size_y/2],[0, 0, 1]])

print(camera)

tmp1 = np.array([[189],[210],[1]])
line_point1 = np.linalg.inv(camera)
line_point1 = np.matmul(line_point1, tmp1)

print(line_point1)
print(line_point1.shape)

tmp2 = np.array([[160],[172],[1]])
line_point2 = np.linalg.inv(camera)
line_point2 = np.matmul(line_point2,tmp2)

print(line_point2)

#cross_two_line1 = np.outer(line_point1, line_point2)
cross_two_line2 = np.cross(line_point1.transpose(),line_point2.transpose())

# print(cross_two_line1)
print(cross_two_line2)

interpretation_plane = cross_two_line2/np.sqrt(cross_two_line2[0][0]**2+cross_two_line2[0][1]**2+cross_two_line2[0][2]**2)

print(interpretation_plane)
