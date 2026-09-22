# %%
import os
import sys
import pickle
import webbrowser
import numpy as np
from tqdm import tqdm
from uaibot import Robot, Utils, Simulation, PointCloud, Frame, Cylinder
from uaibot.simobjects.curve import Curve, CurveSE3
from scipy.linalg import block_diag

from precompute_curve import cylindrical_surface


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


def get_position_from_curve(curve):
    points = []
    for H in curve:
        points.append(np.array(H[:3, -1]))
    return np.array(points).T


# ----------------------------------------------------------------------
#                               LOAD CURVE
# ----------------------------------------------------------------------

path = "./"
file_name = "cylindrical.npy"
curve_path = os.path.join(path, file_name)
print(f"Loading raw curve from {file_name}...")
# curve = np.load(file_name, allow_pickle=True)

# curve = [H for H in curve_]

# %%
# ----------------------------------------------------------------------
#                               SETUP
# ----------------------------------------------------------------------
axial_amplitude = 0.1 / 2
n_axial_oscillations = 3
center = np.array([0.0, 0.25, 0.6])
radius = 0.10
htm = np.eye(4)
height = 2.0 * axial_amplitude + 0.05
n_points = 5000
cylinder_height = height 
htm[:3, 3] = center

curve = cylindrical_surface(
    n_points=n_points,
    radius=radius,
    axial_amplitude=axial_amplitude,
    n_axial_oscillations=n_axial_oscillations,
    center=center,
)
curve_ub = CurveSE3(points=curve)


kinova = Robot.create_kinova_gen3()
target = CurveSE3(
    points=curve,
    size=0.05,
    color="cyan",
    frame_size=0.1,
    num_frames=10,
)

kinova.set_ani_frame(q=config_mapping([0, 10, 0, 15, 0, 40, 30], "from_kinova"))
sim = Simulation.create_sim_grid([kinova, target])
sim.set_parameters(width=1280, height=720)

cylinder = Cylinder(
    htm=htm,
    radius=radius,
    height=cylinder_height,
    opacity=0.5,
    color="blue",
)
sim.add(cylinder)

# ----------------------------------------------------------------------
#                            CONTROL LOOP
# ----------------------------------------------------------------------

T = 40.0
dt = 0.01

kt1, kt2, kt3 = 0.03 * 10, 1.0, 0.75
kn1, kn2 = 0.1 * 10, 0.75
eta = 10.0
eta_lim = 1 / dt
eta_self = 0.6
delta_collision = 0.025
gain_qp = 1.0
h_gdf, eps_gdf = 2e-3, 1e-3

q0 = np.array(kinova.q.copy())
q = q0
n = len(q)

imax = int(T / dt)
dist_hist = []
q_hist = []
qdot_hist = []
cost_hist = []

curve = [H for H in curve]
delta_from_real = np.array([0] + [np.pi] * 6).reshape(-1, 1)
q_min = np.array(kinova.joint_limit[:, 0]) + delta_from_real
q_max = np.array(kinova.joint_limit[:, 1]) + delta_from_real

for i in tqdm(range(imax)):
    J, H = kinova.jac_geo()
    J, H = np.array(J), np.array(H)
    q_ = np.array(kinova.q.copy())
    q_hist.append(q_)

    free_config, msg, info = kinova.check_free_configuration(check_joint=False)
    if not free_config:
        print(msg)

    xi, min_dist, min_index = kinova.vector_field_SE3(
        H,
        curve,
        kt1=kt1,
        kt2=kt2,
        kt3=kt3,
        kn1=kn1,
        kn2=kn2,
        # curve_derivative=curve_derivative,
        delta=1e-3,
        ds=1e-3,
        mode="c++",
    )
    p = np.array(H[:3, -1]).reshape(-1, 1)
    omega = np.array(xi[3:]).reshape(-1, 1)
    v = np.array(xi[:3]).reshape(-1, 1)
    pdot = (np.cross(omega.ravel(), p.ravel()) + v.ravel()).reshape(-1, 1)
    v = pdot
    twist_ = np.vstack((v, omega))

    H_qp = 2 * (J.transpose() @ J + 1e-4 * np.identity(7))
    f_qp = -2 * gain_qp * np.array(J.T @ twist_).reshape(-1)
    dist_struct = kinova.compute_dist(obj=cylinder, h=h_gdf, eps=eps_gdf)
    dist_struct_auto = kinova.compute_dist_auto(h=h_gdf, eps=eps_gdf)
    A_qp_env = dist_struct.jac_dist_mat
    A_qp_self = dist_struct_auto.jac_dist_mat
    A_lim = np.vstack([np.eye(n), -np.eye(n)])
    A_qp = np.vstack([A_qp_env, A_qp_self, A_lim])
    b_qp_env = -eta * (dist_struct.dist_vect - delta_collision)
    b_qp_self = -eta_self * (dist_struct_auto.dist_vect - delta_collision)
    b_lim = np.vstack(
        [
            -eta_lim * (q - q_min),  # lower
            -eta_lim * (q_max - q),  # upper
        ]
    )
    b_qp = np.vstack([b_qp_env, b_qp_self, b_lim])
    # min 0.5u^T H u + f^T u, s.t. Au >= b
    qdot = Utils.solve_qp(
        np.matrix(H_qp), np.matrix(f_qp).T, np.matrix(A_qp), np.matrix(b_qp)
    )
    qdot = np.array(qdot).reshape(-1, 1)
    cost = (qdot.T @ H_qp @ qdot + f_qp.T @ qdot).item()
    qdot_unconstrained = gain_qp * np.linalg.solve(
        J.T @ J + 1e-4 * np.identity(7), J.T @ twist_
    ).reshape(-1, 1)

    violation = A_qp @ qdot_unconstrained - b_qp
    active = violation < -1e-6
    # print("active constraints:", np.sum(active))
    # print("dist_vect:", dist_struct.dist_vect.ravel())
    # print("b_qp:", b_qp.ravel())

    # qdot = Utils.dp_inv(J, 1e-4) @ twist_
    qdot_hist.append(qdot)
    # print(qdot)
    # break
    q = q_ + qdot * dt
    kinova.add_ani_frame(time=i * dt, q=q)
    # target.add_ani_frame(time=i * dt, initial_ind=0, final_ind=len(curve) - 1)
    # H = expSE3(Smap(xi) * dt) @ H # VECTOR FIELD TESTING
    dist_hist.append(min_dist)
    cost_hist.append(cost)

sim.run_in_browser()

# %%
""" PLOT DIST """
import plotly.graph_objects as go

go.Figure(go.Scatter(y=dist_hist)).show()
go.Figure(go.Scatter(y=cost_hist)).show()
