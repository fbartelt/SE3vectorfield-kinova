# %%
import os
import numpy as np
from uaibot import Robot


def circle_rn(n_points, u, v, radius=1.0, center=None, mid=False, dv=10.0):
    points = []
    n = u.shape[0]
    u = np.array(u).ravel()
    v = np.array(v).ravel()

    if center is None:
        center = np.zeros((n,))
    else:
        center = np.array(center).ravel()

    if isinstance(radius, (int, float)):
        radius = np.array([radius] * 2)
    elif isinstance(radius, (list, tuple, np.ndarray)) and len(radius) > 2:
        print("Radius must be a scalar or a 2D vector")
        return None
    else:
        radius = np.array(radius).ravel()

    for i in range(n_points):
        angle = 2 * np.pi * i / n_points
        p_nolimit = (radius[0] * np.cos(angle)) * u
        if mid:
            p_limit = (radius[1] * (np.sin(angle) + 1)) / 2 * v + np.deg2rad(dv)
        else:
            p_limit = (radius[1] * np.sin(angle)) * v
        points.append(center + p_nolimit + p_limit)

    points = np.array(points)
    return points


def get_points_from_curve(curve):
    points = []
    for H in curve:
        points.append(np.array(H[:3, -1]))
    return np.array(points).T


def config_mapping(q, maptype="from_kinova"):
    if maptype == "from_kinova":
        q = np.deg2rad(np.array(q).ravel())
        delta = np.array([0] + [np.pi] * 6).ravel()
        q = q + delta
    else:
        # maptype == 'to_kinova'
        q = np.rad2deg(np.array(q).ravel())
        delta = np.array([0] + [180] * 6).ravel()
        q = q - delta
        q = np.array([qi % 360 for qi in q]).ravel()
    return q


def Smap(xi):
    S_ = np.eye(4)
    S_[0, 1] = -xi[5]
    S_[0, 2] = xi[4]
    S_[1, 2] = -xi[3]
    S_ = S_ - S_.T
    S_[0, 3] = xi[0]
    S_[1, 3] = xi[1]
    S_[2, 3] = xi[2]
    return S_


def expSE3(A):
    S_ = A[:3, :3]
    v = A[:3, 3]
    res = np.eye(4)
    theta = np.sqrt(S_[1, 0] ** 2 + S_[0, 2] ** 2 + S_[2, 1] ** 2)

    if theta < 1e-6:
        R = np.eye(3)
        res[:3, :3] = R
        res[:3, 3] = v
    else:
        R = (
            np.eye(3)
            + np.sin(theta) / theta * S_
            + ((1 - np.cos(theta)) / (theta**2) * S_ @ S_)
        )
        U = (
            np.eye(3)
            + ((1 - np.cos(theta)) / (theta**2)) * S_
            + (((theta - np.sin(theta)) / (theta**3)) * S_ @ S_)
        )
        res[:3, :3] = R
        res[:3, 3] = U @ v

    return res


def EE_dist(V, W):
    c_maxCosTheta = 0.999

    V, W = np.array(V), np.array(W)
    Z = np.linalg.inv(V) @ W
    Q = Z[:3, :3]
    Q_inv = np.linalg.inv(Q)
    u = np.array(Z[:3, -1]).reshape(-1, 1)
    cos_theta = 0.5 * (np.trace(Q) - 1)
    sin_theta = 1.0 / (2.0 * np.sqrt(2)) * np.linalg.norm(Q - Q_inv)
    theta = np.arctan2(sin_theta, cos_theta)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    if cos_theta > c_maxCosTheta:
        alpha = -(1.0 / 12)
    else:
        alpha = (2.0 - 2 * cos_theta - (theta**2)) / (4.0 * (1 - cos_theta) ** 2)

    X_bar = alpha * (Q + Q_inv) + (1 - 2 * alpha) * np.eye(3)
    distance = np.sqrt(2.0 * (theta**2) + u.T @ X_bar @ u)

    return distance


def resample_curve(curve, epsilon):
    ss = [0] * len(curve)
    for i, H in enumerate(curve):
        if i > 0:
            if i == len(curve) - 1:
                H_next = curve[0]
            else:
                H_next = curve[i + 1]

            D = EE_dist(H, H_next)
            ss[i] = ss[i - 1] + D

    resampled_curve = [curve[0]]
    i = 0
    for _ in range(len(curve)):
        if i == len(curve) - 1:
            break
        for j in range(i, len(curve)):
            if ss[j] - ss[i] > epsilon:
                resampled_curve.append(curve[j])
                i = j
                break
    return resampled_curve


kinova = Robot.create_kinova_gen3(htm=np.eye(4), name="kinova")
n_points = 5000
# n_points = 50000
radius = 0.15
dx = 0.0
dy = 0.4
height = 0.4

u = np.array(list(map(int, [(i % 2) == 0 for i in range(7)])))
v = np.logical_not(u).astype(int)
u = u / np.linalg.norm(u)
v = v / np.linalg.norm(v)
center = np.array([0] + [np.pi] * 6)
radius = np.array([np.pi, np.deg2rad(50)])  # mid=False
radius = np.array([np.deg2rad(180), np.deg2rad(50)])  # mid=True, dv=5.0
curve_q = circle_rn(
    n_points=n_points, u=u, v=v, radius=radius, center=center, mid=True, dv=5.0
)
curve = [np.array(kinova.fkm(q=q)) for q in curve_q]

# resampled_curve = resample_curve(curve, 0.008)
# print(len(curve), len(resampled_curve))
# curve = np.array(resampled_curve)
curve = np.array(curve)

file_name = "resampled_curve2.npy"
print("Current path:", os.getcwd())
print(f"Saved resampled curve as '{file_name}'")
np.save(file_name, curve)