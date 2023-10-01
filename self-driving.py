import pygame
import math
import numpy as np
import PySimpleGUI as sg
import threading
from time import time


# parameters initiation
max_c = 0.006  # max curvature
STEP_SIZE = 1
MAX_LENGTH = 10.0
PI = math.pi
FIRST_POINT_CF = 0.75
STEERING_CF = 2.5
NEXT_POINT_OFFSET = 20


# class for PATH element
class PATH:
    def __init__(self, lengths, ctypes, L, x, y, yaw, directions):
        self.lengths = lengths  # lengths of each part of path (+: forward, -: backward) [float]
        self.ctypes = ctypes  # type of each part of the path [string]
        self.L = L  # total path length [float]
        self.x = x  # final x positions [m]
        self.y = y  # final y positions [m]
        self.yaw = yaw  # final yaw angles [rad]
        self.directions = directions  # forward: 1, backward:-1


def calc_optimal_path(sx, sy, syaw, gx, gy, gyaw, maxc, step_size=STEP_SIZE):
    paths = calc_all_paths(sx, sy, syaw, gx, gy, gyaw, maxc, step_size=step_size)

    minL = paths[0].L
    mini = 0

    for i in range(len(paths)):
        if paths[i].L <= minL:
            minL, mini = paths[i].L, i

    return paths[mini]


def calc_all_paths(sx, sy, syaw, gx, gy, gyaw, maxc, step_size=STEP_SIZE):
    q0 = [sx, sy, syaw]
    q1 = [gx, gy, gyaw]

    paths = generate_path(q0, q1, maxc)

    for path in paths:
        x, y, yaw, directions = \
            generate_local_course(path.L, path.lengths,
                                  path.ctypes, maxc, step_size * maxc)

        # convert global coordinate
        path.x = [math.cos(-q0[2]) * ix + math.sin(-q0[2]) * iy + q0[0] for (ix, iy) in zip(x, y)]
        path.y = [-math.sin(-q0[2]) * ix + math.cos(-q0[2]) * iy + q0[1] for (ix, iy) in zip(x, y)]
        path.yaw = [pi_2_pi(iyaw + q0[2]) for iyaw in yaw]
        path.directions = directions
        path.lengths = [l / maxc for l in path.lengths]
        path.L = path.L / maxc

    return paths


def set_path(paths, lengths, ctypes):
    path = PATH([], [], 0.0, [], [], [], [])
    path.ctypes = ctypes
    path.lengths = lengths

    # check same path exist
    for path_e in paths:
        if path_e.ctypes == path.ctypes:
            if sum([x - y for x, y in zip(path_e.lengths, path.lengths)]) <= 0.01:
                return paths  # not insert path

    path.L = sum([abs(i) for i in lengths])

    if path.L >= MAX_LENGTH:
        return paths

    assert path.L >= 0.01
    paths.append(path)

    return paths


def LSL(x, y, phi):
    u, t = R(x - math.sin(phi), y - 1.0 + math.cos(phi))

    if t >= 0.0:
        v = M(phi - t)
        if v >= 0.0:
            return True, t, u, v

    return False, 0.0, 0.0, 0.0


def LSR(x, y, phi):
    u1, t1 = R(x + math.sin(phi), y - 1.0 - math.cos(phi))
    u1 = u1 ** 2

    if u1 >= 4.0:
        u = math.sqrt(u1 - 4.0)
        theta = math.atan2(2.0, u)
        t = M(t1 + theta)
        v = M(t - phi)

        if t >= 0.0 and v >= 0.0:
            return True, t, u, v

    return False, 0.0, 0.0, 0.0


def LRL(x, y, phi):
    u1, t1 = R(x - math.sin(phi), y - 1.0 + math.cos(phi))

    if u1 <= 4.0:
        u = -2.0 * math.asin(0.25 * u1)
        t = M(t1 + 0.5 * u + PI)
        v = M(phi - t + u)

        if t >= 0.0 and u <= 0.0:
            return True, t, u, v

    return False, 0.0, 0.0, 0.0


def SCS(x, y, phi, paths):
    flag, t, u, v = SLS(x, y, phi)

    if flag:
        paths = set_path(paths, [t, u, v], ["S", "L", "S"])

    flag, t, u, v = SLS(x, -y, -phi)
    if flag:
        paths = set_path(paths, [t, u, v], ["S", "R", "S"])

    return paths


def SLS(x, y, phi):
    phi = M(phi)

    if y > 0.0 and 0.0 < phi < PI * 0.99:
        xd = -y / math.tan(phi) + x
        t = xd - math.tan(phi / 2.0)
        u = phi
        v = math.sqrt((x - xd) ** 2 + y ** 2) - math.tan(phi / 2.0)
        return True, t, u, v
    elif y < 0.0 and 0.0 < phi < PI * 0.99:
        xd = -y / math.tan(phi) + x
        t = xd - math.tan(phi / 2.0)
        u = phi
        v = -math.sqrt((x - xd) ** 2 + y ** 2) - math.tan(phi / 2.0)
        return True, t, u, v

    return False, 0.0, 0.0, 0.0


def CSC(x, y, phi, paths):
    flag, t, u, v = LSL(x, y, phi)
    if flag:
        paths = set_path(paths, [t, u, v], ["L", "S", "L"])

    flag, t, u, v = LSL(-x, y, -phi)
    if flag:
        paths = set_path(paths, [-t, -u, -v], ["L", "S", "L"])

    flag, t, u, v = LSL(x, -y, -phi)
    if flag:
        paths = set_path(paths, [t, u, v], ["R", "S", "R"])

    flag, t, u, v = LSL(-x, -y, phi)
    if flag:
        paths = set_path(paths, [-t, -u, -v], ["R", "S", "R"])

    flag, t, u, v = LSR(x, y, phi)
    if flag:
        paths = set_path(paths, [t, u, v], ["L", "S", "R"])

    flag, t, u, v = LSR(-x, y, -phi)
    if flag:
        paths = set_path(paths, [-t, -u, -v], ["L", "S", "R"])

    flag, t, u, v = LSR(x, -y, -phi)
    if flag:
        paths = set_path(paths, [t, u, v], ["R", "S", "L"])

    flag, t, u, v = LSR(-x, -y, phi)
    if flag:
        paths = set_path(paths, [-t, -u, -v], ["R", "S", "L"])

    return paths


def CCC(x, y, phi, paths):
    flag, t, u, v = LRL(x, y, phi)
    if flag:
        paths = set_path(paths, [t, u, v], ["L", "R", "L"])

    flag, t, u, v = LRL(-x, y, -phi)
    if flag:
        paths = set_path(paths, [-t, -u, -v], ["L", "R", "L"])

    flag, t, u, v = LRL(x, -y, -phi)
    if flag:
        paths = set_path(paths, [t, u, v], ["R", "L", "R"])

    flag, t, u, v = LRL(-x, -y, phi)
    if flag:
        paths = set_path(paths, [-t, -u, -v], ["R", "L", "R"])

    # backwards
    xb = x * math.cos(phi) + y * math.sin(phi)
    yb = x * math.sin(phi) - y * math.cos(phi)

    flag, t, u, v = LRL(xb, yb, phi)
    if flag:
        paths = set_path(paths, [v, u, t], ["L", "R", "L"])

    flag, t, u, v = LRL(-xb, yb, -phi)
    if flag:
        paths = set_path(paths, [-v, -u, -t], ["L", "R", "L"])

    flag, t, u, v = LRL(xb, -yb, -phi)
    if flag:
        paths = set_path(paths, [v, u, t], ["R", "L", "R"])

    flag, t, u, v = LRL(-xb, -yb, phi)
    if flag:
        paths = set_path(paths, [-v, -u, -t], ["R", "L", "R"])

    return paths


def calc_tauOmega(u, v, xi, eta, phi):
    delta = M(u - v)
    A = math.sin(u) - math.sin(delta)
    B = math.cos(u) - math.cos(delta) - 1.0

    t1 = math.atan2(eta * A - xi * B, xi * A + eta * B)
    t2 = 2.0 * (math.cos(delta) - math.cos(v) - math.cos(u)) + 3.0

    if t2 < 0:
        tau = M(t1 + PI)
    else:
        tau = M(t1)

    omega = M(tau - u + v - phi)

    return tau, omega


def LRLRn(x, y, phi):
    xi = x + math.sin(phi)
    eta = y - 1.0 - math.cos(phi)
    rho = 0.25 * (2.0 + math.sqrt(xi * xi + eta * eta))

    if rho <= 1.0:
        u = math.acos(rho)
        t, v = calc_tauOmega(u, -u, xi, eta, phi)
        if t >= 0.0 and v <= 0.0:
            return True, t, u, v

    return False, 0.0, 0.0, 0.0


def LRLRp(x, y, phi):
    xi = x + math.sin(phi)
    eta = y - 1.0 - math.cos(phi)
    rho = (20.0 - xi * xi - eta * eta) / 16.0

    if 0.0 <= rho <= 1.0:
        u = -math.acos(rho)
        if u >= -0.5 * PI:
            t, v = calc_tauOmega(u, u, xi, eta, phi)
            if t >= 0.0 and v >= 0.0:
                return True, t, u, v

    return False, 0.0, 0.0, 0.0


def CCCC(x, y, phi, paths):
    flag, t, u, v = LRLRn(x, y, phi)
    if flag:
        paths = set_path(paths, [t, u, -u, v], ["L", "R", "L", "R"])

    flag, t, u, v = LRLRn(-x, y, -phi)
    if flag:
        paths = set_path(paths, [-t, -u, u, -v], ["L", "R", "L", "R"])

    flag, t, u, v = LRLRn(x, -y, -phi)
    if flag:
        paths = set_path(paths, [t, u, -u, v], ["R", "L", "R", "L"])

    flag, t, u, v = LRLRn(-x, -y, phi)
    if flag:
        paths = set_path(paths, [-t, -u, u, -v], ["R", "L", "R", "L"])

    flag, t, u, v = LRLRp(x, y, phi)
    if flag:
        paths = set_path(paths, [t, u, u, v], ["L", "R", "L", "R"])

    flag, t, u, v = LRLRp(-x, y, -phi)
    if flag:
        paths = set_path(paths, [-t, -u, -u, -v], ["L", "R", "L", "R"])

    flag, t, u, v = LRLRp(x, -y, -phi)
    if flag:
        paths = set_path(paths, [t, u, u, v], ["R", "L", "R", "L"])

    flag, t, u, v = LRLRp(-x, -y, phi)
    if flag:
        paths = set_path(paths, [-t, -u, -u, -v], ["R", "L", "R", "L"])

    return paths


def LRSR(x, y, phi):
    xi = x + math.sin(phi)
    eta = y - 1.0 - math.cos(phi)
    rho, theta = R(-eta, xi)

    if rho >= 2.0:
        t = theta
        u = 2.0 - rho
        v = M(t + 0.5 * PI - phi)
        if t >= 0.0 and u <= 0.0 and v <= 0.0:
            return True, t, u, v

    return False, 0.0, 0.0, 0.0


def LRSL(x, y, phi):
    xi = x - math.sin(phi)
    eta = y - 1.0 + math.cos(phi)
    rho, theta = R(xi, eta)

    if rho >= 2.0:
        r = math.sqrt(rho * rho - 4.0)
        u = 2.0 - r
        t = M(theta + math.atan2(r, -2.0))
        v = M(phi - 0.5 * PI - t)
        if t >= 0.0 and u <= 0.0 and v <= 0.0:
            return True, t, u, v

    return False, 0.0, 0.0, 0.0


def CCSC(x, y, phi, paths):
    flag, t, u, v = LRSL(x, y, phi)
    if flag:
        paths = set_path(paths, [t, -0.5 * PI, u, v], ["L", "R", "S", "L"])

    flag, t, u, v = LRSL(-x, y, -phi)
    if flag:
        paths = set_path(paths, [-t, 0.5 * PI, -u, -v], ["L", "R", "S", "L"])

    flag, t, u, v = LRSL(x, -y, -phi)
    if flag:
        paths = set_path(paths, [t, -0.5 * PI, u, v], ["R", "L", "S", "R"])

    flag, t, u, v = LRSL(-x, -y, phi)
    if flag:
        paths = set_path(paths, [-t, 0.5 * PI, -u, -v], ["R", "L", "S", "R"])

    flag, t, u, v = LRSR(x, y, phi)
    if flag:
        paths = set_path(paths, [t, -0.5 * PI, u, v], ["L", "R", "S", "R"])

    flag, t, u, v = LRSR(-x, y, -phi)
    if flag:
        paths = set_path(paths, [-t, 0.5 * PI, -u, -v], ["L", "R", "S", "R"])

    flag, t, u, v = LRSR(x, -y, -phi)
    if flag:
        paths = set_path(paths, [t, -0.5 * PI, u, v], ["R", "L", "S", "L"])

    flag, t, u, v = LRSR(-x, -y, phi)
    if flag:
        paths = set_path(paths, [-t, 0.5 * PI, -u, -v], ["R", "L", "S", "L"])

    # backwards
    xb = x * math.cos(phi) + y * math.sin(phi)
    yb = x * math.sin(phi) - y * math.cos(phi)

    flag, t, u, v = LRSL(xb, yb, phi)
    if flag:
        paths = set_path(paths, [v, u, -0.5 * PI, t], ["L", "S", "R", "L"])

    flag, t, u, v = LRSL(-xb, yb, -phi)
    if flag:
        paths = set_path(paths, [-v, -u, 0.5 * PI, -t], ["L", "S", "R", "L"])

    flag, t, u, v = LRSL(xb, -yb, -phi)
    if flag:
        paths = set_path(paths, [v, u, -0.5 * PI, t], ["R", "S", "L", "R"])

    flag, t, u, v = LRSL(-xb, -yb, phi)
    if flag:
        paths = set_path(paths, [-v, -u, 0.5 * PI, -t], ["R", "S", "L", "R"])

    flag, t, u, v = LRSR(xb, yb, phi)
    if flag:
        paths = set_path(paths, [v, u, -0.5 * PI, t], ["R", "S", "R", "L"])

    flag, t, u, v = LRSR(-xb, yb, -phi)
    if flag:
        paths = set_path(paths, [-v, -u, 0.5 * PI, -t], ["R", "S", "R", "L"])

    flag, t, u, v = LRSR(xb, -yb, -phi)
    if flag:
        paths = set_path(paths, [v, u, -0.5 * PI, t], ["L", "S", "L", "R"])

    flag, t, u, v = LRSR(-xb, -yb, phi)
    if flag:
        paths = set_path(paths, [-v, -u, 0.5 * PI, -t], ["L", "S", "L", "R"])

    return paths


def LRSLR(x, y, phi):
    # formula 8.11 *** TYPO IN PAPER ***
    xi = x + math.sin(phi)
    eta = y - 1.0 - math.cos(phi)
    rho, theta = R(xi, eta)

    if rho >= 2.0:
        u = 4.0 - math.sqrt(rho * rho - 4.0)
        if u <= 0.0:
            t = M(math.atan2((4.0 - u) * xi - 2.0 * eta, -2.0 * xi + (u - 4.0) * eta))
            v = M(t - phi)

            if t >= 0.0 and v >= 0.0:
                return True, t, u, v

    return False, 0.0, 0.0, 0.0


def CCSCC(x, y, phi, paths):
    flag, t, u, v = LRSLR(x, y, phi)
    if flag:
        paths = set_path(paths, [t, -0.5 * PI, u, -0.5 * PI, v], ["L", "R", "S", "L", "R"])

    flag, t, u, v = LRSLR(-x, y, -phi)
    if flag:
        paths = set_path(paths, [-t, 0.5 * PI, -u, 0.5 * PI, -v], ["L", "R", "S", "L", "R"])

    flag, t, u, v = LRSLR(x, -y, -phi)
    if flag:
        paths = set_path(paths, [t, -0.5 * PI, u, -0.5 * PI, v], ["R", "L", "S", "R", "L"])

    flag, t, u, v = LRSLR(-x, -y, phi)
    if flag:
        paths = set_path(paths, [-t, 0.5 * PI, -u, 0.5 * PI, -v], ["R", "L", "S", "R", "L"])

    return paths


def generate_local_course(L, lengths, mode, maxc, step_size):
    point_num = int(L / step_size) + len(lengths) + 3

    px = [0.0 for _ in range(point_num)]
    py = [0.0 for _ in range(point_num)]
    pyaw = [0.0 for _ in range(point_num)]
    directions = [0 for _ in range(point_num)]
    ind = 1

    if lengths[0] > 0.0:
        directions[0] = 1
    else:
        directions[0] = -1

    if lengths[0] > 0.0:
        d = step_size
    else:
        d = -step_size

    pd = d
    ll = 0.0

    for m, l, i in zip(mode, lengths, range(len(mode))):
        if l > 0.0:
            d = step_size
        else:
            d = -step_size

        ox, oy, oyaw = px[ind], py[ind], pyaw[ind]

        ind -= 1
        if i >= 1 and (lengths[i - 1] * lengths[i]) > 0:
            pd = -d - ll
        else:
            pd = d - ll

        while abs(pd) <= abs(l):
            ind += 1
            px, py, pyaw, directions = \
                interpolate(ind, pd, m, maxc, ox, oy, oyaw, px, py, pyaw, directions)
            pd += d

        ll = l - pd - d  # calc remain length

        ind += 1
        px, py, pyaw, directions = \
            interpolate(ind, l, m, maxc, ox, oy, oyaw, px, py, pyaw, directions)

    # remove unused data
    while px[-1] == 0.0:
        px.pop()
        py.pop()
        pyaw.pop()
        directions.pop()

    return px, py, pyaw, directions


def interpolate(ind, l, m, maxc, ox, oy, oyaw, px, py, pyaw, directions):
    if m == "S":
        px[ind] = ox + l / maxc * math.cos(oyaw)
        py[ind] = oy + l / maxc * math.sin(oyaw)
        pyaw[ind] = oyaw
    else:
        ldx = math.sin(l) / maxc
        if m == "L":
            ldy = (1.0 - math.cos(l)) / maxc
        elif m == "R":
            ldy = (1.0 - math.cos(l)) / (-maxc)

        gdx = math.cos(-oyaw) * ldx + math.sin(-oyaw) * ldy
        gdy = -math.sin(-oyaw) * ldx + math.cos(-oyaw) * ldy
        px[ind] = ox + gdx
        py[ind] = oy + gdy

    if m == "L":
        pyaw[ind] = oyaw + l
    elif m == "R":
        pyaw[ind] = oyaw - l

    if l > 0.0:
        directions[ind] = 1
    else:
        directions[ind] = -1

    return px, py, pyaw, directions


def generate_path(q0, q1, maxc):
    dx = q1[0] - q0[0]
    dy = q1[1] - q0[1]
    dth = q1[2] - q0[2]
    c = math.cos(q0[2])
    s = math.sin(q0[2])
    x = (c * dx + s * dy) * maxc
    y = (-s * dx + c * dy) * maxc

    paths = []
    paths = SCS(x, y, dth, paths)
    paths = CSC(x, y, dth, paths)
    paths = CCC(x, y, dth, paths)
    paths = CCCC(x, y, dth, paths)
    paths = CCSC(x, y, dth, paths)
    paths = CCSCC(x, y, dth, paths)

    return paths


# utils
def pi_2_pi(theta):
    while theta > PI:
        theta -= 2.0 * PI

    while theta < -PI:
        theta += 2.0 * PI

    return theta


def R(x, y):
    """
    Return the polar coordinates (r, theta) of the point (x, y)
    """
    r = math.hypot(x, y)
    theta = math.atan2(y, x)

    return r, theta


def M(theta):
    """
    Regulate theta to -pi <= theta < pi
    """
    phi = theta % (2.0 * PI)

    if phi < -PI:
        phi += 2.0 * PI
    if phi > PI:
        phi -= 2.0 * PI

    return phi


def get_label(path):
    label = ""

    for m, l in zip(path.ctypes, path.lengths):
        label = label + m
        if l > 0.0:
            label = label + "+"
        else:
            label = label + "-"

    return label


def calc_curvature(x, y, yaw, directions):
    c, ds = [], []

    for i in range(1, len(x) - 1):
        dxn = x[i] - x[i - 1]
        dxp = x[i + 1] - x[i]
        dyn = y[i] - y[i - 1]
        dyp = y[i + 1] - y[i]
        dn = math.hypot(dxn, dyn)
        dp = math.hypot(dxp, dyp)
        dx = 1.0 / (dn + dp) * (dp / dn * dxn + dn / dp * dxp)
        ddx = 2.0 / (dn + dp) * (dxp / dp - dxn / dn)
        dy = 1.0 / (dn + dp) * (dp / dn * dyn + dn / dp * dyp)
        ddy = 2.0 / (dn + dp) * (dyp / dp - dyn / dn)
        curvature = (ddy * dx - ddx * dy) / (dx ** 2 + dy ** 2)
        d = (dn + dp) / 2.0

        if np.isnan(curvature):
            curvature = 0.0

        if directions[i] <= 0.0:
            curvature = -curvature

        if len(c) == 0:
            ds.append(d)
            c.append(curvature)

        ds.append(d)
        c.append(curvature)

    ds.append(ds[-1])
    c.append(c[-1])

    return c, ds


def check_path(sx, sy, syaw, gx, gy, gyaw, maxc):
    paths = calc_all_paths(sx, sy, syaw, gx, gy, gyaw, maxc)

    assert len(paths) >= 1

    for path in paths:
        assert abs(path.x[0] - sx) <= 0.01
        assert abs(path.y[0] - sy) <= 0.01
        assert abs(path.yaw[0] - syaw) <= 0.01
        assert abs(path.x[-1] - gx) <= 0.01
        assert abs(path.y[-1] - gy) <= 0.01
        assert abs(path.yaw[-1] - gyaw) <= 0.01

        # course distance check
        d = [math.hypot(dx, dy)
             for dx, dy in zip(np.diff(path.x[0:len(path.x) - 1]),
                               np.diff(path.y[0:len(path.y) - 1]))]

        for i in range(len(d)):
            assert abs(d[i] - STEP_SIZE) <= 0.001


def det(arr):
    return (arr[0][0] * arr[1][1]) - (arr[0][1] * arr[1][0])


def vector_angle(vector):
    return math.atan2(vector.y, vector.x)


class POINT:

    def __init__(self, x, y, angle=0, direction = 1):
        self.x = x
        self.y = y
        self.angle = angle
        self.direction = direction


def distance(A, B):
    return math.sqrt((A.x - B.x) ** 2 + (A.y - B.y) ** 2)


def is_between(A, B, C):
    a = distance(B, C)
    b = distance(C, A)
    c = distance(A, B)
    return a ** 2 + b ** 2 >= c ** 2 and a ** 2 + c ** 2 >= b ** 2


class STRAIGHT:

    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def intersection_bool(self, x3, y3, x4, y4):
        try:
            if abs((self.x1 - self.x2) * (y3 - y4) - (self.y1 - self.y2) * (x3 - x4)) > 100:
                Px = ((self.x1 * self.y2 - self.y1 * self.x2) * (x3 - x4) - (self.x1 - self.x2) * (
                            x3 * y4 - y3 * x4)) / ((self.x1 - self.x2) * (y3 - y4) - (self.y1 - self.y2) * (x3 - x4))
            else:
                Px = ((self.x1 * self.y2 - self.y1 * self.x2) * (x3 - x4) - (self.x1 - self.x2) * (
                            x3 * y4 - y3 * x4)) / 100

            if abs((self.x1 - self.x2) * (y3 - y4) - (self.y1 - self.y2) * (x3 - x4)) > 100:
                Py = ((self.x1 * self.y2 - self.y1 * self.x2) * (y3 - y4) - (self.y1 - self.y2) * (
                            x3 * y4 - y3 * x4)) / ((self.x1 - self.x2) * (y3 - y4) - (self.y1 - self.y2) * (x3 - x4))
            else:
                Py = ((self.x1 * self.y2 - self.y1 * self.x2) * (x3 - x4) - (self.x1 - self.x2) * (
                            x3 * y4 - y3 * x4)) / 100
        except:
            return False

        if is_between(POINT(Px, Py), POINT(x3, y3), POINT(x4, y4)) and is_between(POINT(Px, Py),
                                                                                  POINT(self.x1, self.y1),
                                                                                  POINT(self.x2, self.y2)):
            return True
        else:
            return False

    def intersection(self, x3, y3, x4, y4):
        if abs((self.x1 - self.x2) * (y3 - y4) - (self.y1 - self.y2) * (x3 - x4)) > 100:
            Px = ((self.x1 * self.y2 - self.y1 * self.x2) * (x3 - x4) - (self.x1 - self.x2) * (x3 * y4 - y3 * x4)) / (
                        (self.x1 - self.x2) * (y3 - y4) - (self.y1 - self.y2) * (x3 - x4))
        else:
            Px = ((self.x1 * self.y2 - self.y1 * self.x2) * (x3 - x4) - (self.x1 - self.x2) * (x3 * y4 - y3 * x4)) / 100

        if abs((self.x1 - self.x2) * (y3 - y4) - (self.y1 - self.y2) * (x3 - x4)) > 100:
            Py = ((self.x1 * self.y2 - self.y1 * self.x2) * (y3 - y4) - (self.y1 - self.y2) * (x3 * y4 - y3 * x4)) / (
                        (self.x1 - self.x2) * (y3 - y4) - (self.y1 - self.y2) * (x3 - x4))
        else:
            Py = ((self.x1 * self.y2 - self.y1 * self.x2) * (x3 - x4) - (self.x1 - self.x2) * (x3 * y4 - y3 * x4)) / 100

        if is_between(POINT(Px, Py), POINT(x3, y3), POINT(x4, y4)) and is_between(POINT(Px, Py),
                                                                                  POINT(self.x1, self.y1),
                                                                                  POINT(self.x2, self.y2)):
            return POINT(Px, Py, int(distance(POINT(x3, y3), POINT(Px, Py))))
        else:
            return POINT(x4, y4, int(distance(POINT(x3, y3), POINT(x4, y4))))


class VECTOR2:

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def median(self, x, y, k):
        self.x = (self.x - x) * k + x
        self.y = (self.y - y) * k + y
        return VECTOR2(self.x, self.y)

    def get_length(self):
        return math.sqrt(self.x ** 2 + self.y ** 2)

    # reference = VECTOR2(1, 0)

    def AngleOfReference(self, v):
        return self.NormalizeAngle(math.atan2(v.y, v.x) / math.pi * 180)

    def AngleOfVectors(self, first, second):
        return self.NormalizeAngle(self.AngleOfReference(first) - self.AngleOfReference(second))

    def NormalizeAngle(self, angle):
        if angle > -180:
            turn = -360
        else:
            turn = 360
        while not (angle > -180 and angle <= 180):
            angle += turn
        return angle

    def mult(self, scalar):
        return VECTOR2(self.x * scalar, self.y * scalar)

    def __add__(self, other):
        return VECTOR2(self.x + other.x, self.y + other.y)

    def __repr__(self):
        return f"x: {self.x}, y: {self.y}"


class CAMERA:
    def __init__(self, position=POINT(0, 0, 0)):
        self.position = position
        self.is_up = False
        self.is_left = False
        self.is_right = False
        self.is_down = False


class PLAYER:

    def __init__(self, position=POINT(0, 0, 0), velocity=0):
        self.position = position
        self.velocity = velocity
        self.vector = VECTOR2(0, 0)
        self.steering_angle = 0.0
        self.is_w = False
        self.is_a = False
        self.is_s = False
        self.is_d = False


def solve_angle(angle, add):
    angle += add
    angle %= 360
    return angle


def to_radians(angle):
    angle = float(angle - 90)
    angle = (angle / 180) * 3.14159
    return angle


def blit_rotate(surf, image, pos, originPos, angle):
    angle = -angle
    # calculate the axis aligned bounding box of the rotated image
    w, h = image.get_size()
    box = [pygame.math.Vector2(p) for p in [(0, 0), (w, 0), (w, -h), (0, -h)]]
    box_rotate = [p.rotate(angle) for p in box]
    min_box = (min(box_rotate, key=lambda p: p[0])[0], min(box_rotate, key=lambda p: p[1])[1])
    max_box = (max(box_rotate, key=lambda p: p[0])[0], max(box_rotate, key=lambda p: p[1])[1])

    # calculate the translation of the pivot
    pivot = pygame.math.Vector2(originPos[0], -originPos[1])
    pivot_rotate = pivot.rotate(angle)
    pivot_move = pivot_rotate - pivot

    # calculate the upper left origin of the rotated image
    origin = (pos[0] - originPos[0] + min_box[0] - pivot_move[0], pos[1] - originPos[1] - max_box[1] + pivot_move[1])

    # get a rotated image
    rotated_image = pygame.transform.rotate(image, angle)

    # rotate and blit the image
    surf.blit(rotated_image, origin)


# параметры окна
size = (1200, 800)
screen = pygame.display.set_mode(size)
clock = pygame.time.Clock()
screen.fill((200, 200, 200))
pygame.display.set_caption("Initial N")
pygame.mixer.init()
runGame = True

# изображения
trueno = pygame.image.load("trueno.png")

# Следование до точки
destinations = []
high_destinations = []
is_achieved = True
is_reverse = False
destination = POINT(0, 0, 0)
departure = POINT(0, 0, 0)
destination_angle = 0
pygame.font.init()
heading_font = pygame.font.SysFont("Arial", 20)
current_vector_point = None     # Start direction vector
current_wall_point = None       # Wall first point

# пременные
fps = 60
player = PLAYER(POINT(300, 300, 0), 0)
camera = CAMERA(POINT(0, 0, 0))
show_lines = 1

# карта стен
walls = []

# Путь
last_add_way = 0
way = []


def research_destinations():
    global high_destinations
    global STEP_SIZE

    high_destinations = []
    path_x, path_y, yaw, directions = [], [], [], []
    tmp_low_destintaions = destinations.copy()

    tmp_low_destintaions.insert(0, POINT(player.position.x,
                                         player.position.y,
                                         angle=np.deg2rad(player.position.angle + 90),
                                         direction=-1))

    for i in range(len(tmp_low_destintaions) - 1):
        s_x = tmp_low_destintaions[i].x
        s_y = tmp_low_destintaions[i].y
        s_yaw = tmp_low_destintaions[i].angle
        g_x = tmp_low_destintaions[i + 1].x
        g_y = tmp_low_destintaions[i + 1].y
        g_yaw = tmp_low_destintaions[i + 1].angle

        path_i = calc_optimal_path(s_x, s_y, s_yaw,
                                   g_x, g_y, g_yaw, max_c, STEP_SIZE)

        for j in range(len(path_i.x)):
            high_destinations.append(POINT(path_i.x[j],
                                           path_i.y[j],
                                           angle=path_i.yaw[j] + 90,
                                           direction=-path_i.directions[j]))


# Окно системы проектирования
layout = [  [sg.Text('Автомобиль')],
            [sg.Text('Ширина'), sg.InputText(default_text=max_c)],
            [sg.Text('Длина'), sg.InputText(default_text=max_c)],
            [sg.HorizontalSeparator()],
            [sg.Text('Алгоритм следования')],
            [sg.Text('Соотношение значимости'), sg.Slider(orientation='h', range=(0, 1), resolution=0.05, default_value=0.5)],
            [sg.Text('Коэффициент руления'), sg.InputText(default_text=STEERING_CF)],
            [sg.Text('Дальность второй точки'), sg.Slider(orientation='h', range=(2, 50), resolution=1, default_value=NEXT_POINT_OFFSET)],
            [sg.HorizontalSeparator()],
            [sg.Text('Алгоритм построения пути')],
            [sg.Text('Минимальный радиус'), sg.InputText(default_text=max_c)],
            [sg.Text('Шаг дискретизации'), sg.InputText(default_text=STEP_SIZE)],
            [sg.Button('Ok'), sg.Button('Cancel')] ]


def CAD_window():
    global runGame
    global max_c
    global STEP_SIZE
    global FIRST_POINT_CF
    global STEERING_CF
    global NEXT_POINT_OFFSET
    window = sg.Window('Система проектирования БПТС', layout)
    last_update = {0: '0', 1: '0', 3: '0', 4: '0', 5: '0', 7: '0', 8: '0'}

    while runGame:
        event, values = window.read(timeout=100)
        if event == sg.WIN_CLOSED or event == 'Cancel':  # if user closes window or clicks cancel
            runGame = False
            break

        FIRST_POINT_CF = float(values[3])
        try:
            if float(values[4]) > 0:
                STEERING_CF = float(values[4])
        except:
            pass
        NEXT_POINT_OFFSET = int(values[5])
        try:
            if float(values[7]) > 0:
                max_c = float(values[7])
        except:
            pass
        try:
            if float(values[8]) != 0:
                STEP_SIZE = float(values[8])
        except:
            pass

        if last_update[7] != values[7] or last_update[8] != values[8]:
            research_destinations()
            last_update = values
            print(values)


CAD_window_task = threading.Thread(target=CAD_window, args=())
CAD_window_task.start()

# отрисовка
while runGame:
    # Отслеживание событий для движения
    for event in pygame.event.get():
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                is_reverse = False
                current_vector_point = POINT(pygame.mouse.get_pos()[0] - camera.position.x,
                                             pygame.mouse.get_pos()[1] - camera.position.y, direction = 1)

                departure = POINT(player.position.x, player.position.y, 0)
            if event.button == 2:
                destinations.clear()
                high_destinations.clear()
                is_achieved = True
            if event.button == 3:
                current_wall_point = POINT(pygame.mouse.get_pos()[0] - camera.position.x,
                                           pygame.mouse.get_pos()[1] - camera.position.y, direction=1)

        if event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                if current_vector_point != None:
                    current_vector_point.angle = vector_angle(VECTOR2(current_vector_point.x - (pygame.mouse.get_pos()[0] - camera.position.x),
                                                                      current_vector_point.y - (pygame.mouse.get_pos()[1] - camera.position.y)))
                    destinations.append(current_vector_point)
                    current_vector_point = None
            if event.button == 3:
                if current_wall_point != None:
                    walls.append(STRAIGHT(current_wall_point.x,
                                          current_wall_point.y,
                                          pygame.mouse.get_pos()[0] - camera.position.x,
                                          pygame.mouse.get_pos()[1] - camera.position.y))
                    current_wall_point = None

        if event.type == pygame.KEYDOWN:
            if event.key == 99:
                is_achieved = False
            elif event.key == 118:
                is_achieved = True
            elif event.key == 27:
                runGame = False
            elif event.key == 108:
                if show_lines == 3:
                    show_lines = 0
                else:
                    show_lines += 1
            elif event.key == 122:
                walls = []
            elif event.key == 98:
                way = []
            elif event.key == 100:
                player.is_d = True
            elif event.key == 97:
                player.is_a = True
            elif event.key == 119:
                player.is_w = True
            elif event.key == 115:
                player.is_s = True

            elif event.key == 1073741904:
                camera.is_left = True
            elif event.key == 1073741906:
                camera.is_up = True
            elif event.key == 1073741903:
                camera.is_right = True
            elif event.key == 1073741905:
                camera.is_down = True

            elif event.key == 120:
                research_destinations()

        if event.type == pygame.KEYUP:
            if event.key == 27:
                runGame = False
            elif event.key == 100:
                player.is_d = False
            elif event.key == 97:
                player.is_a = False
            elif event.key == 119:
                player.is_w = False
            elif event.key == 115:
                player.is_s = False

            elif event.key == 1073741904:
                camera.is_left = False
            elif event.key == 1073741906:
                camera.is_up = False
            elif event.key == 1073741903:
                camera.is_right = False
            elif event.key == 1073741905:
                camera.is_down = False

    # Применение событий для машины
    if not is_achieved:
        if not is_reverse:
            if player.velocity < 5:
                player.velocity += 1 * (1 / 2)
            player.steering_angle = max(min(destination_angle * STEERING_CF, 5), -5)
        else:
            if player.velocity > -2:
                player.velocity -= 1 * (1 / 2)
            player.steering_angle = -max(min(destination_angle * STEERING_CF, 5), -5)
    else:
        if not is_reverse:
            if player.velocity > 1:
                player.velocity -= 1 * (1 / 2)
        else:
            if player.velocity < 0:
                player.velocity += 1 * (1 / 2)
        player.steering_angle = 0
    # if player.is_d and player.steering_angle <= 3.5:
    #     player.steering_angle += 0.2 * (1 / 2)
    # if player.is_a and player.steering_angle >= -3.5:
    #     player.steering_angle -= 0.2 * (1 / 2)
    # if player.steering_angle <= 3.5:
    #     player.steering_angle += 0.2 * (1 / 2)
    # if player.is_a and player.steering_angle >= -3.5:
    #     player.steering_angle -= 0.2 * (1 / 2)

    # Применение событий для камеры
    if camera.is_left:
        camera.position.x += 20
    if camera.is_right:
        camera.position.x -= 20
    if camera.is_up:
        camera.position.y += 20
    if camera.is_down:
        camera.position.y -= 20

    ## перемещение

    # Слежение камеры за машиной
    # camera.position.x -= (camera.position.x + player.position.x - 600) * 0.15 * (1 / 2)
    # camera.position.y -= (camera.position.y + player.position.y - 400) * 0.15 * (1 / 2)

    player.position.angle = solve_angle(player.position.angle,
                                        player.steering_angle * (player.vector.get_length() / 10))

    direction_vector = VECTOR2(math.cos(to_radians(player.position.angle)) * player.velocity * (1 / 2),
                               math.sin(to_radians(player.position.angle)) * player.velocity * (1 / 2))
    player.vector.median(direction_vector.x, direction_vector.y, 0.7)

    player.position.x += player.vector.x
    player.position.y += player.vector.y

    # притормаживание
    if player.velocity > 0:
        player.velocity -= 0.25 * (1 / 2)
    # Возврат руля
    if (not player.is_d) and (not player.is_a) and player.steering_angle > 0:
        player.steering_angle -= 0.1
    if (not player.is_a) and (not player.is_d) and player.steering_angle < 0:
        player.steering_angle += 0.1

    screen.fill((200, 200, 200))
    blit_rotate(screen, trueno, (player.position.x + camera.position.x, player.position.y + camera.position.y),
                (29, 76), player.position.angle)

    point_front = POINT(player.position.x + math.cos(to_radians(player.position.angle)) * 1000,
                        player.position.y + math.sin(to_radians(player.position.angle)) * 1000)
    point_half_left = POINT(player.position.x + math.cos(to_radians(player.position.angle + 35)) * 1000,
                            player.position.y + math.sin(to_radians(player.position.angle + 35)) * 1000)

    point_half_right = POINT(player.position.x + math.cos(to_radians(player.position.angle - 35)) * 1000,
                             player.position.y + math.sin(to_radians(player.position.angle - 35)) * 1000)
    point_left = POINT(player.position.x + math.cos(to_radians(player.position.angle + 90)) * 1000,
                       player.position.y + math.sin(to_radians(player.position.angle + 90)) * 1000)

    point_right = POINT(player.position.x + math.cos(to_radians(player.position.angle - 90)) * 1000,
                        player.position.y + math.sin(to_radians(player.position.angle - 90)) * 1000)

    if (time() - last_add_way > 0.1):
        way.append(POINT(player.position.x, player.position.y))
        last_add_way = time()
    for i in range(1, len(way)):
        pygame.draw.line(screen, pygame.Color(255, 0, 255), [way[i - 1].x + camera.position.x, way[i - 1].y + camera.position.y],
                         [way[i].x + camera.position.x, way[i].y + camera.position.y], 2)

    for wall in walls:
        pygame.draw.line(screen, pygame.Color(255, 0, 0), [wall.x1 + camera.position.x, wall.y1 + camera.position.y],
                         [wall.x2 + camera.position.x, wall.y2 + camera.position.y], 2)

        point_front = wall.intersection(player.position.x, player.position.y,
                                        point_front.x, point_front.y)
        point_half_left = wall.intersection(player.position.x, player.position.y,
                                            point_half_left.x, point_half_left.y)
        point_half_right = wall.intersection(player.position.x, player.position.y,
                                             point_half_right.x, point_half_right.y)
        point_left = wall.intersection(player.position.x, player.position.y,
                                       point_left.x, point_left.y)
        point_right = wall.intersection(player.position.x, player.position.y,
                                        point_right.x, point_right.y)

    if show_lines == 0 or show_lines == 2:
        pygame.draw.line(screen, pygame.Color(0, 0, 255),
                         [player.position.x + camera.position.x, player.position.y + camera.position.y],
                         [point_front.x + camera.position.x, point_front.y + camera.position.y], 2)
        pygame.draw.circle(screen, pygame.Color(0, 0, 0),
                           (point_front.x + camera.position.x, point_front.y + camera.position.y), 5.0)
        pygame.draw.line(screen, pygame.Color(0, 0, 255),
                         [player.position.x + camera.position.x, player.position.y + camera.position.y],
                         [point_half_left.x + camera.position.x, point_half_left.y + camera.position.y], 2)
        pygame.draw.circle(screen, pygame.Color(0, 0, 0),
                           (point_half_left.x + camera.position.x, point_half_left.y + camera.position.y), 5.0)
        pygame.draw.line(screen, pygame.Color(0, 0, 255),
                         [player.position.x + camera.position.x, player.position.y + camera.position.y],
                         [point_half_right.x + camera.position.x, point_half_right.y + camera.position.y], 2)
        pygame.draw.circle(screen, pygame.Color(0, 0, 0),
                           (point_half_right.x + camera.position.x, point_half_right.y + camera.position.y), 5.0)
        pygame.draw.line(screen, pygame.Color(0, 0, 255),
                         [player.position.x + camera.position.x, player.position.y + camera.position.y],
                         [point_left.x + camera.position.x, point_left.y + camera.position.y], 2)
        pygame.draw.circle(screen, pygame.Color(0, 0, 0),
                           (point_left.x + camera.position.x, point_left.y + camera.position.y), 5.0)
        pygame.draw.line(screen, pygame.Color(0, 0, 255),
                         [player.position.x + camera.position.x, player.position.y + camera.position.y],
                         [point_right.x + camera.position.x, point_right.y + camera.position.y], 2)
        pygame.draw.circle(screen, pygame.Color(0, 0, 0),
                           (point_right.x + camera.position.x, point_right.y + camera.position.y), 5.0)

    is_crash = False
    front_corner_l = POINT(
        player.position.x + (75 * math.cos(to_radians(player.position.angle + 20))),
        player.position.y + (75 * math.sin(to_radians(player.position.angle + 20))))
    front_corner_r = POINT(
        player.position.x + (75 * math.cos(to_radians(player.position.angle - 20))),
        player.position.y + (75 * math.sin(to_radians(player.position.angle - 20))))
    rear_corner_l = POINT(
        player.position.x + (75 * math.cos(to_radians(player.position.angle + 160))),
        player.position.y + (75 * math.sin(to_radians(player.position.angle + 160))))
    rear_corner_r = POINT(
        player.position.x + (75 * math.cos(to_radians(player.position.angle - 160))),
        player.position.y + (75 * math.sin(to_radians(player.position.angle - 160))))

    # отрисовка "векторов" направлений

    if current_vector_point != None:
            pygame.draw.line(screen, pygame.Color(255, 255, 0),
                             [current_vector_point.x + camera.position.x, current_vector_point.y + camera.position.y],
                             [pygame.mouse.get_pos()[0],
                              pygame.mouse.get_pos()[1]], 3)

    if current_wall_point != None:
            pygame.draw.line(screen, pygame.Color(0, 255, 100),
                             [current_wall_point.x + camera.position.x, current_wall_point.y + camera.position.y],
                             [pygame.mouse.get_pos()[0],
                              pygame.mouse.get_pos()[1]], 3)

    if (show_lines == 1 or show_lines == 2):
        # pygame.draw.line(screen, pygame.Color(0, 255, 0),
        #                  [player.position.x + camera.position.x, player.position.y + camera.position.y],
        #                  [player.position.x + (player.vector.x * 10) + camera.position.x,
        #                   player.position.y + (player.vector.y * 10) + camera.position.y], 3)

        # pygame.draw.line(screen, pygame.Color(255, 0, 0),
        #                  [player.position.x + camera.position.x, player.position.y + camera.position.y],
        #                  [player.position.x + (direction_vector.x * 10) + camera.position.x,
        #                   player.position.y + (direction_vector.y * 10) + camera.position.y], 3)

        angle_vector = VECTOR2(math.cos(to_radians(player.position.angle)) * 10 * (1 / 2),
                               math.sin(to_radians(player.position.angle)) * 10 * (1 / 2))
        # pygame.draw.line(screen, pygame.Color(255, 0, 0),
        #                  [player.position.x + camera.position.x, player.position.y + camera.position.y],
        #                  [player.position.x + (angle_vector.x * 20) + camera.position.x,
        #                   player.position.y + (angle_vector.y * 20) + camera.position.y], 3)

        if len(destinations) > 0:
            pygame.draw.line(screen, pygame.Color(255, 255, 0),
                             [player.position.x + camera.position.x, player.position.y + camera.position.y],
                             [destinations[0].x + camera.position.x,
                              destinations[0].y + camera.position.y], 3)
            for destination_iterator in range(1, len(destinations)):
                pygame.draw.line(screen, pygame.Color(150, 150, 150),
                                 [destinations[destination_iterator].x + camera.position.x,
                                  destinations[destination_iterator].y + camera.position.y],
                                 [destinations[destination_iterator - 1].x + camera.position.x,
                                  destinations[destination_iterator - 1].y + camera.position.y], 3)
            while len(destinations) > 0 and VECTOR2(player.position.x - destinations[0].x, player.position.y - destinations[0].y).get_length() < 20:
                destinations.pop(0)

        if len(high_destinations) > 0:
            for destination_iterator in range(1, len(high_destinations)):
                if high_destinations[destination_iterator].direction == 1:
                    pygame.draw.line(screen, pygame.Color(255, 150, 150),
                                     [high_destinations[destination_iterator].x + camera.position.x,
                                      high_destinations[destination_iterator].y + camera.position.y],
                                     [high_destinations[destination_iterator - 1].x + camera.position.x,
                                      high_destinations[destination_iterator - 1].y + camera.position.y], 3)
                elif high_destinations[destination_iterator].direction == -1:
                    pygame.draw.line(screen, pygame.Color(150, 150, 255),
                                     [high_destinations[destination_iterator].x + camera.position.x,
                                      high_destinations[destination_iterator].y + camera.position.y],
                                     [high_destinations[destination_iterator - 1].x + camera.position.x,
                                      high_destinations[destination_iterator - 1].y + camera.position.y], 3)

            if high_destinations[0].direction == 1:
                is_reverse = False
            else:
                is_reverse = True

            destination_vector = None
            if len(high_destinations) > NEXT_POINT_OFFSET:
                # Векторы, указывающие на напрвления до следующих точек
                first_point_vector = VECTOR2(high_destinations[0].x - player.position.x, high_destinations[0].y - player.position.y)
                second_point_vector = VECTOR2(high_destinations[NEXT_POINT_OFFSET].x - high_destinations[0].x, high_destinations[NEXT_POINT_OFFSET].y - high_destinations[0].y)
                destination_vector = (first_point_vector.mult(FIRST_POINT_CF) + second_point_vector.mult(1 - FIRST_POINT_CF))
                destination_vector = destination_vector.mult(0.5)
                pygame.draw.circle(screen, pygame.Color(255, 0, 0),
                                   (high_destinations[NEXT_POINT_OFFSET].x + camera.position.x, high_destinations[NEXT_POINT_OFFSET].y + camera.position.y), 5.0)

            else:
                destination_vector = VECTOR2(high_destinations[0].x - player.position.x, high_destinations[0].y - player.position.y)
            pygame.draw.circle(screen, pygame.Color(0, 255, 0),
                               (high_destinations[0].x + camera.position.x,
                                high_destinations[0].y + camera.position.y), 5.0)

            # Угол между этим вектором и курсом автомобиля
            destination_angle = destination_vector.AngleOfVectors(destination_vector, angle_vector)

            if VECTOR2(player.position.x - high_destinations[0].x, player.position.y - high_destinations[0].y).get_length() > 50:
                research_destinations()

            while not is_achieved and VECTOR2(player.position.x - high_destinations[0].x, player.position.y - high_destinations[0].y).get_length() < 10:
                high_destinations.pop(0)
                departure = POINT(player.position.x, player.position.y, 0)
                if len(high_destinations) == 0:
                    is_achieved = True

        # pygame.draw.line(screen, pygame.Color(0, 160, 160),
        #                  [destination.x + camera.position.x, destination.y + camera.position.y],
        #                  [departure.x + camera.position.x, departure.y + camera.position.y], 3)

        # label = heading_font.render(str(destination_vector.AngleOfVectors(destination_vector, angle_vector)), True, (50, 50, 50))
        # label_rect = label.get_rect()
        # label_rect.center = (130, 25)
        # screen.blit(label, label_rect)
        #
        # label = heading_font.render(str(destination_vector.AngleOfVectors(destination_vector, departure_vector)), True, (50, 50, 50))
        # label_rect = label.get_rect()
        # label_rect.center = (130, 50)
        # screen.blit(label, label_rect)

    pygame.draw.circle(screen, pygame.Color(255, 0, 0),
                       [destination.x + camera.position.x, destination.y + camera.position.y], 5)

    for wall in walls:
        if wall.intersection_bool(front_corner_l.x, front_corner_l.y,
                                  front_corner_r.x, front_corner_r.y):
            is_crash = True
            '''pygame.draw.line(screen, pygame.Color(255, 0, 0),
                             [front_corner_l.x + camera.position.x, front_corner_l.y + camera.position.y],
                             [front_corner_r.x + camera.position.x, front_corner_r.y + camera.position.y], 3)'''

        if wall.intersection_bool(front_corner_l.x, front_corner_l.y,
                                  rear_corner_l.x, rear_corner_l.y):
            is_crash = True
            '''pygame.draw.line(screen, pygame.Color(255, 0, 0),
                             [front_corner_l.x + camera.position.x, front_corner_l.y + camera.position.y],
                             [rear_corner_l.x + camera.position.x, rear_corner_l.y + camera.position.y], 3)'''

        if wall.intersection_bool(rear_corner_r.x, rear_corner_r.y,
                                  front_corner_r.x, front_corner_r.y):
            is_crash = True
            '''pygame.draw.line(screen, pygame.Color(255, 0, 0),
                             [rear_corner_r.x + camera.position.x, rear_corner_r.y + camera.position.y],
                             [front_corner_r.x + camera.position.x, front_corner_r.y + camera.position.y], 3)'''

        if wall.intersection_bool(rear_corner_r.x, rear_corner_r.y,
                                  rear_corner_l.x, rear_corner_l.y):
            is_crash = True
            '''pygame.draw.line(screen, pygame.Color(255, 0, 0),
                             [rear_corner_r.x + camera.position.x, rear_corner_r.y + camera.position.y],
                             [rear_corner_l.x + camera.position.x, rear_corner_l.y + camera.position.y], 3)'''

        if is_crash:
            player = PLAYER(POINT(300, 300, 0), 0)

    pygame.display.update()

    clock.tick(fps)

pygame.quit()
