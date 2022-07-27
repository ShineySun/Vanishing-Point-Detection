import pickle
import matplotlib.pyplot as plt
import numpy as np

with open('./norm_dist/norm_dist_branch.pkl', 'rb') as f:
    norm_dist = pickle.load(f)
# print(norm_dist)

others = 0

lst = [[0, 0], [0.01, 0], [0.02, 0], [0.03, 0], [0.04, 0],
       [0.05, 0], [0.06, 0], [0.07, 0], [0.08, 0], [0.09, 0]]


for i in norm_dist:
    if i >=0 and i <0.01: lst[0][1] += 1
    elif i < 0.02: lst[1][1] += 1
    elif i < 0.03: lst[2][1] += 1
    elif i < 0.04: lst[3][1] += 1
    elif i < 0.05: lst[4][1] += 1
    elif i < 0.06: lst[5][1] += 1
    elif i < 0.07: lst[6][1] += 1
    elif i < 0.08: lst[7][1] += 1
    elif i < 0.09: lst[8][1] += 1
    elif i < 0.1: lst[9][1] += 1
    else: others += 1

for i in range(len(lst)):
    lst[i][1] /= len(norm_dist)
    if i>0:
        lst[i][1] += lst[i-1][1]

lst = np.array(lst).T.tolist()

plt.scatter(lst[0], lst[1])
plt.plot(lst[0], lst[1], label = 'x_axis')
plt.title("NormDist")
plt.xlabel("normdist error")
plt.ylabel("percentage of the whole dataset(%)")
plt.xticks(np.arange(0, 0.1, 0.01))
plt.legend()
plt.show()
