# %%
import os
import sys
import pickle
import webbrowser
import numpy as np
from uaibot import Robot, Utils, Simulation, PointCloud, Frame
from pathlib import Path


def open_in_browser(filename: str):
    """
    Opens an HTML file in the system's default web browser.
    Works cross-platform (Linux, macOS, Windows).
    """
    path = Path(filename).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    # Convert to file:// URL and open
    webbrowser.open_new_tab(path.as_uri())


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


def get_points_from_curve(curve):
    points = []
    for H in curve:
        points.append(np.array(H[:3, -1]))
    return np.array(points).T


path = "./"
file_name = f"{path}/resampled_curve2.npy"
print(f"Loading raw curve from {file_name}...")

curve = np.load(file_name, allow_pickle=True)
# curve = [H for H in curve_]
# %%
""" EXPERIMENT EXPECTED MOVEMENT """
print("creating kinova")
kinova = Robot.create_kinova_gen3(name="kinova")
print("created")
point_mat = get_points_from_curve(curve)
target = PointCloud(name="target", points=point_mat, size=0.01, color="cyan")

# kinova.set_ani_frame(q=config_mapping([0, 0, 0, 5, 0, 10, 0], "from_kinova"))
kinova.set_ani_frame(q=config_mapping([0, 10, 0, 15, 0, 40, 30], "from_kinova"))
# kinova.set_ani_frame(q=config_mapping([0, 10, 0, 15, 0, 40., 180.72], "from_kinova"))

sim = Simulation.create_sim_grid([kinova, target])

frames = []
n_frames = 20
frame_htms_ = curve[np.linspace(0, len(curve) - 1, n_frames).astype(int)]
frame_htms_ = [H for H in frame_htms_]

# Manually improve frames distribution:
frame_htms = frame_htms_[:-10]
frame_htms.append(frame_htms_[-8])
frame_htms.append(frame_htms_[-5])

for i, htm in enumerate(frame_htms):
    frame = Frame(htm, name=f"frame_{i}", size=0.1)
    frames.append(frame)

sim.add(frames)

kt1, kt2, kt3 = 0.03, 1.0, 0.75
kn1, kn2 = 0.1, 0.75

q0 = np.array(kinova.q.copy())
q = q0

T = 200.0
dt = 0.01
imax = int(T / dt)
dist_hist = []
q_hist = []
qdot_hist = []

curve = [H for H in curve]

for i in range(imax):
    J, H = kinova.jac_geo()
    H_qp = 2 * (J.transpose() * J + 0.0001 * np.identity(7))
    q_ = np.array(kinova.q.copy())
    q_hist.append(q_)
    # q_[1:] = np.array([(qi + np.pi) for qi in q_.ravel()[1:]]).reshape(-1, 1)
    free_config, msg, info = kinova.check_free_configuration(check_joint=False)
    if not free_config:
        print(msg)
    # xi, min_dist, min_index = kinova.vector_field_se3(H, curve, kt1=kt1, kt2=kt2, kt3=kt3, kn1=kn1, kn2=kn2,
    #                                                   delta=1e-3, ds=1e-3,
    #                                                   mode='c++')
    xi, min_dist, min_index = kinova.vector_field_se3(
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
    # f_qp = np.array(J.T @ twist_).reshape(-1, 1)
    # b_qp = eta * np.array(np.vstack((q -q_max, q_min - q))).reshape(-1, 1)
    qdot = Utils.dp_inv(J, 1e-4) @ twist_
    # qdot, *_ = solve_qp(H_qp, f_qp.ravel().astype(np.double), A_qp.T, b_qp.ravel())
    qdot = np.array(qdot).reshape(-1, 1)
    qdot_hist.append(qdot)
    # print(qdot)
    # break
    q = q_ + qdot * dt
    kinova.add_ani_frame(time=i * dt, q=q)
    target.add_ani_frame(time=i * dt, initial_ind=0, final_ind=len(curve) - 1)
    # H = expSE3(Smap(xi) * dt) @ H # VECTOR FIELD TESTING
    dist_hist.append(min_dist)

sim.save(path, "expected_movement")
open_in_browser('expected_movement.html')

# %%
""" PLOT DIST """
import plotly.graph_objects as go

go.Figure(go.Scatter(y=dist_hist)).show()

# %%
""" CHECK CURVE """
point_mat = get_points_from_curve(curve)
target = PointCloud(name="target", points=point_mat, size=0.01, color="cyan")

# kinova.set_ani_frame(q=config_mapping([0, 0, 0, 5, 0, 10, 0], "from_kinova"))

sim = Simulation.create_sim_grid(target)

frames = []
n_frames = 20
frame_htms_ = curve[np.linspace(0, len(curve) - 1, n_frames).astype(int)]
frame_htms_ = [H for H in frame_htms_]

# Manually improve frames distribution:
# frame_htms = frame_htms_[:-10]
# frame_htms.append(frame_htms_[-8])
# frame_htms.append(frame_htms_[-5])

# for i, htm in enumerate(frame_htms):
#     frame = Frame(htm, name=f"frame_{i}", size=0.1)
#     frames.append(frame)

frame = Frame(htm=np.eye(4), name="test_frame", size=0.1)
sim.add(frame)

for i, H in enumerate(curve):
    target.add_ani_frame(time=i * 0.01, initial_ind=0, final_ind=i)
    frame.add_ani_frame(time=i * 0.01, htm=H)

sim.save("/tmp", "curve_check")
open_in_browser(os.path.join("/tmp", "curve_check.html"))
# %%
