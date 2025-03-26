#%%
import pickle
import numpy as np
from uaibot import Robot, Utils, Simulation, PointCloud, Frame

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

curve = np.load('/home/fbartelt/Documents/Projetos/SE3vectorfield-kinova/resampled_curve.npy', allow_pickle=True)
# curve = [H for H in curve_]
#%%
""" EXPERIMENT EXPECTED MOVEMENT """
print("creaqting kinova")
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

kt1, kt2, kt3 = 0.03, 1.0, 1.0
kn1, kn2 = 0.1, 1.0

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
    H_qp = 2*(J.transpose() * J + 0.0001 *np.identity(7))
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

sim.run()

#%%
""" PLOT DIST """
import plotly.graph_objects as go

go.Figure(go.Scatter(y=dist_hist)).show()
#%%
""" CHECK QDOT"""
qdot_hist = np.array(qdot_hist).reshape(-1, 7)

for i in range(7):
    print(f'joint {i}')
    print(np.max(qdot_hist[:, i]) * 180/np.pi)
    print(np.min(qdot_hist[:, i]) * 180/np.pi)

# %%
""" BASIC TESTING """
kinova = Robot.create_kinova_gen3(name="kinova")
kinova.set_ani_frame(q=config_mapping([0., 0., 0., 0., 0., 0., 0.], "from_kinova"))
speeds = [5, 0, -5, 0, -5, 0, 5, 0]

dt = 0.01
i = 1

for joint in range(7):
    for speed in speeds:
        for _ in range(500):
            q_ = kinova.q.copy()
            q_[joint] += (speed * np.pi / 180.0) * dt
            kinova.add_ani_frame(time=i*dt, q=q_)
            i += 1

sim = Simulation.create_sim_grid([kinova])
sim.run()
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
    target.add_ani_frame(time=i*0.01, initial_ind=0, final_ind=i)
    frame.add_ani_frame(time=i*0.01, htm=H)

sim.run()
# %%
