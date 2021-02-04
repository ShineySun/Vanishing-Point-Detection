import numpy as np

from scipy.interpolate import CubicSpline
from scipy.interpolate import interp1d

class HermiteCubic:
    def __init__(self, x, y):
        self.x = np.array(x)
        self.y = np.array(y)
        self.n = self.y.size
        self.h = self.x[1:] - self.x[:-1]
        self.m = (self.y[1:] - self.y[:-1]) / self.h
        self.a = self.y[:]
        self.b = self.compute_b(self.x, self.y)
        self.c = (3 * self.m - self.b[1:] - 2 * self.b[:-1]) / self.h
        self.d = (self.b[1:] + self.b[:-1] - 2 * self.m) / (self.h * self.h)

    def compute_b(self, t, r):
        b = np.empty(self.n)
        for i in range(1, self.n - 1):
            is_mono = self.m[i - 1] * self.m[i] > 0
            if is_mono:
                b[i] = 3 * self.m[i - 1] * self.m[i] / (max(self.m[i - 1], self.m[i]) + 2 * min(self.m[i - 1], self.m[i]))
            else:
                b[i] = 0
            if is_mono and self.m[i] > 0:
                b[i] = min(max(0, b[i]), 3 * min(self.m[i - 1], self.m[i]))
            elif is_mono and self.m[i] < 0:
                b[i] = max(min(0, b[i]), 3 * max(self.m[i - 1], self.m[i]))

        b[0] = ((2 * self.h[0] + self.h[1]) * self.m[0] - self.h[0] * self.m[1]) / (self.h[0] + self.h[1])
        b[self.n - 1] = ((2 * self.h[self.n - 2] + self.h[self.n - 3]) * self.m[self.n - 2]
                         - self.h[self.n - 2] * self.m[self.n - 3]) / (self.h[self.n - 2] + self.h[self.n - 3])
        return b

    def evaluate(self, t_intrp):
        ans = []
        for tau in t_intrp:
            i = np.where(tau >= self.x)[0]
            if i.size == 0:
                i = 0
            else:
                i = i[-1]
            i = min(i, self.n-2)
            res = self.a[i] + self.b[i] * (tau - self.x[i]) + self.c[i] * np.power(tau - self.x[i], 2.0) + self.d[i] * np.power(tau - self.x[i], 3.0) #original curve r(t)
            ans.append(res)
        return ans

    def evaluate_derivative(self, t_intrp):
        ans = []
        if not hasattr(t_intrp, "__len__"):
            t_intrp = np.array([t_intrp])
        for tau in t_intrp:
            i = np.where(tau >= self.x)[0]
            if i.size == 0:
                i = 0
            else:
                i = i[-1]
            i = min(i, self.n-2)
            res = self.b[i] + 2 * self.c[i] * (tau - self.x[i]) + 3 * self.d[i] * np.power(tau - self.x[i], 2.0)
            ans.append(res)
        if len(ans) == 1:
            return ans[0]
        else:
            return ans

    def evaluate_forward(self, t_intrp):
        ans = []
        for tau in t_intrp:
            i = np.where(tau >= self.x)[0]
            if i.size == 0:
                i = 0
            else:
                i = i[-1]
            i = min(i, self.n-2)
            res = self.a[i] + self.b[i] * (2 * tau - self.x[i]) + self.c[i] * (tau - self.x[i]) * (3*tau - self.x[i]) \
                  + self.d[i] * np.power(tau - self.x[i], 2.0) * (4 * tau - self.x[i]) # d(xy)/dx
            ans.append(res)
        return ans

def mono_spline(lanes):
    mono_spline_sets = []

    for lane in lanes:
        lane = np.flip(lane, axis=0)

        # y points -> x axis
        x = lane[:, 1]
        # x points -> y axis
        y = lane[:, 0]

        # interpolation scheme : Mono spline
        mn_intrp = HermiteCubic(x,y)

        x_intrp = np.linspace(int(x.min()), int(x.max()), int(x.max())-int(x.min())+1)
        y_intrp = mn_intrp.evaluate(x_intrp)

        intrp_lane = np.array(list(zip(y_intrp, x_intrp)))

        mono_spline_sets.append(np.flip(intrp_lane, axis=0))

    return mono_spline_sets

def cubic_spline(lanes):
    cubic_intrp_sets = []

    for lane in lanes:
        lane = np.flip(lane, axis=0)

        # y points -> x axis
        x = lane[:, 1]
        # x points -> y axis
        y = lane[:, 0]

        # interpolation scheme : Cubic Spline
        cs_intrp = CubicSpline(x,y)

        x_intrp = np.linspace(int(x.min()), int(x.max()), int(x.max())-int(x.min())+1)
        y_intrp = cs_intrp(x_intrp)

        intrp_lane = np.array(list(zip(y_intrp, x_intrp)))

        cubic_intrp_sets.append(np.flip(intrp_lane, axis=0))

    return cubic_intrp_sets


def linear_interpolate(lanes):
    linear_intrp_sets = []

    for lane in lanes:
        lane = np.flip(lane, axis=0)

        # y points -> x axis
        x = lane[:, 1]
        # x points -> y axis
        y = lane[:, 0]

        # interpolation scheme : linear
        lin_intrp = interp1d(x,y, kind='linear')

        x_intrp = np.linspace(int(x.min()), int(x.max()), int(x.max())-int(x.min())+1)
        y_intrp = lin_intrp(x_intrp)

        intrp_lane = np.array(list(zip(y_intrp, x_intrp)))

        linear_intrp_sets.append(np.flip(intrp_lane, axis=0))

    return linear_intrp_sets

def quadratic_interpolate(lanes):
    quadratic_intrp_sets = []

    for lane in lanes:
        lane = np.flip(lane, axis=0)

        # y points -> x axis
        x = lane[:, 1]
        # x points -> y axis
        y = lane[:, 0]

        # interpolation scheme : linear
        quadratic_intrp = interp1d(x,y, kind='quadratic')

        x_intrp = np.linspace(int(x.min()), int(x.max()), int(x.max())-int(x.min())+1)
        y_intrp = quadratic_intrp(x_intrp)

        intrp_lane = np.array(list(zip(y_intrp, x_intrp)))

        quadratic_intrp_sets.append(np.flip(intrp_lane,axis=0))

    return quadratic_intrp_sets
