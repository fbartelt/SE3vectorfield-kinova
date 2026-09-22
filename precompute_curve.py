# %%
import os
import numpy as np
from uaibot import Robot
from pathlib import Path

def cylindrical_surface(
    n_points,
    radius=0.15,
    axial_amplitude=0.1,
    n_axial_oscillations=2,
    center=None,
    theta_offset=0.0,
):
    """
    Generate a closed SE(3) curve on a cylindrical surface for path following.

    The end-effector position wraps once around the cylinder while oscillating
    along the cylinder axis, covering a band of the surface without
    self-intersection. The orientation is chosen so that the tool z-axis
    points radially inward (normal to the surface), which is the natural
    configuration for inspection, welding, or painting tasks on pipes,
    tanks, or fuselage sections.

    Parameters
    ----------
    n_points : int
        Number of sampled points along the curve.
    radius : float
        Radius of the cylinder.
    axial_amplitude : float
        Amplitude of the axial oscillation (half of the band height).
    n_axial_oscillations : int
        Number of axial oscillations per full revolution. Must be a
        positive integer so that the curve closes smoothly.
    center : array-like, optional
        Center of the cylinder in world frame. Defaults to the origin.
    theta_offset : float
        Angular offset in radians (rotation of the whole path around the
        cylinder axis). Defaults to 0.

    Returns
    -------
    curve : np.ndarray of shape (n_points, 4, 4)
        Array of homogeneous transformation matrices in SE(3).
    """
    if center is None:
        center = np.zeros(3)
    else:
        center = np.array(center).ravel()

    curve = np.zeros((n_points, 4, 4))
    dcurve = np.zeros((n_points, 4, 4))
    for i in range(n_points):
        s_param = i / n_points
        theta = 2.0 * np.pi * s_param + theta_offset
        c, s = np.cos(theta), np.sin(theta)

        # Position on the cylinder surface
        H = np.eye(4)
        H[0, 3] = center[0] + radius * c
        H[1, 3] = center[1] + radius * s
        H[2, 3] = center[2] + axial_amplitude * np.sin(n_axial_oscillations * theta)

        # Orientation: tool z-axis points radially inward (normal to surface)
        # x_T: tangent to the circle (opposite traversal direction so frame stays right-handed)
        # y_T: world +z direction (pointing up)
        # z_T: inward radial direction (normal to the surface)
        R = np.array(
            [
                [s, 0.0, -c],
                [-c, 0.0, -s],
                [0.0, 1.0, 0.0],
            ]
        )
        H[:3, :3] = R
        curve[i] = H

        # Derivative with respect to theta
        dR_dtheta = np.array(
            [
                [c, 0.0, s],
                [s, 0.0, -c],
                [0.0, 0.0, 0.0],
            ]
        )
        dp_dtheta = np.array(
            [
                -radius * s,
                radius * c,
                axial_amplitude
                * n_axial_oscillations
                * np.cos(n_axial_oscillations * theta),
            ]
        )

        dH_dtheta = np.zeros((4, 4))
        dH_dtheta[:3, :3] = dR_dtheta
        dH_dtheta[:3, 3] = dp_dtheta

        # Convert to derivative with respect to the normalized parameter s
        # since theta = 2*pi*s + theta_offset, dtheta/ds = 2*pi
        dcurve[i] = dH_dtheta * (2.0 * np.pi)

    return curve, dcurve


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
radius = 0.15

# ----------------------------------------------------------------------
#                                CIRCLE
# ----------------------------------------------------------------------

# dx = 0.0
# dy = 0.4
# height = 0.4
# u = np.array(list(map(int, [(i % 2) == 0 for i in range(7)])))
# v = np.logical_not(u).astype(int)
# u = u / np.linalg.norm(u)
# v = v / np.linalg.norm(v)
# center = np.array([0] + [np.pi] * 6)
# radius = np.array([np.pi, np.deg2rad(50)])  # mid=False
# radius = np.array([np.deg2rad(180), np.deg2rad(50)])  # mid=True, dv=5.0
# curve_q = circle_rn(
#     n_points=n_points, u=u, v=v, radius=radius, center=center, mid=True, dv=5.0
# )
# curve = [np.array(kinova.fkm(q=q)) for q in curve_q]
# curve = np.array(curve)

# ----------------------------------------------------------------------
#                          CYLINDRICAL SURFACE
# ----------------------------------------------------------------------
axial_amplitude = 0.1 / 2
n_axial_oscillations = 3
center = np.array([0.0, 0.25, 0.6])
radius = 0.07
htm = np.eye(4)
height = 2.0 * axial_amplitude + 0.05
n_points = 5000
curve, dcurve = cylindrical_surface(
    n_points=n_points,
    radius=radius,
    axial_amplitude=axial_amplitude,
    n_axial_oscillations=n_axial_oscillations,
    center=center,
)

# ----------------------------------------------------------------------
#                               RESAMPLING
# ----------------------------------------------------------------------

# resampled_curve = resample_curve(curve, 0.008)
# print(len(curve), len(resampled_curve))
# curve = np.array(resampled_curve)

# ----------------------------------------------------------------------
#                               SAVE CURVE
# ----------------------------------------------------------------------

data_folder = "./data"
curve_file_name = "cylindrical.npy"
dcurve_file_name = "cylindrical_derivative.npy"
directory = Path(data_folder)
directory.mkdir(parents=True, exist_ok=True)
curve_path = os.path.join(data_folder, curve_file_name)
dcurve_path = os.path.join(data_folder, dcurve_file_name)
np.save(curve_path, curve)
print(f"Saved resampled curve at '{curve_path}'")
np.save(dcurve_path, dcurve)
print(f"Saved resampled curve at '{dcurve_path}'")

# ----------------------------------------------------------------------
#                               VISUALIZE CURVE
# ----------------------------------------------------------------------

# import uaibot as ub
# from uaibot.simobjects.curve import CurveSE3
# from uaibot.simobjects.cylinder import Cylinder
#
# curve_ub = CurveSE3(points=curve)
# htm = np.eye(4)
# htm[:3, 3] = center
# height = 2.0 * axial_amplitude + 0.05
# cylinder_reference = Cylinder(htm=htm, radius=radius, height=height)
# sim = ub.Simulation([curve_ub, cylinder_reference])
# sim.run_in_browser()
