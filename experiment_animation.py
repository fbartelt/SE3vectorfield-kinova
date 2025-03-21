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

with open('config_data.pkl', 'rb') as f:
    data = pickle.load(f)

config_hist = data['config_hist']
time_hist = data['time_hist']

curve = np.load('resampled_curve.npy', allow_pickle=True)
kinova = Robot.create_kinova_gen3(name="kinova")
point_mat = get_points_from_curve(curve)
target = PointCloud(name="target", points=point_mat, size=0.01, color="cyan")

kinova.set_ani_frame(q=config_mapping([0, 0, 0, 5, 0, 10, 0], "from_kinova"))
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

for i, q_ in enumerate(config_hist):
    q_rad = config_mapping(q_, "from_kinova")
    kinova.add_ani_frame(time=time_hist[i], q=q_rad)
    target.add_ani_frame(time=time_hist[i], initial_ind=0, final_ind=len(curve) - 1)

sim.run()

# %%
