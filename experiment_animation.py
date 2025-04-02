#%%
import pickle
import numpy as np
from uaibot import Robot, Utils, Simulation, PointCloud, Frame
from scipy.interpolate import interp1d

def config_mapping(q, maptype="from_kinova"):
    if maptype == "from_kinova":
        q = np.deg2rad(np.array(q).ravel())
        delta = np.array([0] + [np.pi] * 6).ravel()
        q = np.unwrap(q + delta)
        q = (q + np.pi) % (2 * np.pi) - np.pi 
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

# q_ = [q.astype(float) for q in data['config_hist']]
# with open('mapped_data.pkl', 'wb') as f:
#     data = {'config_hist':q_, 'time_hist': data['time_hist']}
#     pickle.dump(data, f)
file_name = 'config' # config
with open(f'data.pkl', 'rb') as f:
    data = pickle.load(f)

config_hist = data['config_hist']
time_hist = data['hist_time']

config_hist = np.load('config_hist_exp.npy')
time_hist = np.load('time_hist_exp.npy')

curve_file = 'resampled_curve2'
curve = np.load(f'{curve_file}.npy', allow_pickle=True)
print("creating kinova")
kinova = Robot.create_kinova_gen3(name="kinova")
print("kinova created")
point_mat = get_points_from_curve(curve)
target = PointCloud(name="target", points=point_mat, size=0.01, color="cyan")

# kinova.set_ani_frame(q=config_mapping([0, 0, 0, 5, 0, 10, 0], "from_kinova"))
kinova.set_ani_frame(q=config_mapping([0, 10, 0, 15, 0, 40, 30], "from_kinova"))

sim = Simulation.create_sim_grid([kinova, target])

frames = []
n_frames = 20
frame_htms_ = curve[np.linspace(0, len(curve) - 1, n_frames).astype(int)]
frame_htms_ = [H for H in frame_htms_]

# Manually improve frames distribution:
frame_htms = frame_htms_[:-10]
frame_htms.append(frame_htms_[-8])
frame_htms.append(frame_htms_[-5])

print("frames")
for i, htm in enumerate(frame_htms):
    frame = Frame(htm, name=f"frame_{i}", size=0.1)
    frames.append(frame)
del frames[1]
sim.add(frames)
# sim.set_parameters(ambient_light_intensity=0.01)
sim.run()
#%%
# # Interpolate configurations for smoother movement:
time_hist = np.array(time_hist)
# config_hist = np.array(config_hist)
config_hist = np.array([config_mapping(q, "from_kinova") for q in config_hist])
fps = 200
config_upsampled = []
t_upsampled = []

def angle_interp(start, end, t):
    qs = []
    start = np.unwrap(start)
    end = np.unwrap(end)
    # at = np.arange(0, 1, ratio)
    for t_ in t:
        q_ = start + ((((((end - start) % (2 * np.pi)) + (3 * np.pi)) % (2 * np.pi)) - np.pi) / (t[-1] - t[0])) * (t_ - t[0])
        qs.append(q_)
    return qs

for i, t in enumerate(time_hist[:-1]):
    t_eval = np.arange(t, time_hist[i + 1], 1/fps)
    # q_interval = np.array([config_hist[i, :], config_hist[i + 1, :]]).T
    # f_interp = interp1d([t, time_hist[i + 1]], q_interval, kind='linear', fill_value='extrapolate')
    # q_ = (np.unwrap(f_interp(t_eval).T) + np.pi) % (2 * np.pi) - np.pi
    q_ = angle_interp(config_hist[i, :], config_hist[i + 1, :], t_eval)
    config_upsampled.extend(q_)
    t_upsampled.extend(t_eval)

t_upsampled.append(time_hist[-1])
config_upsampled.append(config_hist[-1])

config_upsampled = np.array(config_upsampled)
t_upsampled = np.array(t_upsampled)

# t_upsampled = np.arange(0, time_hist[-1], 1/fps)
# f_interp = interp1d(time_hist, config_hist.T, kind='cubic', fill_value='extrapolate')
# config_upsampled = f_interp(t_upsampled).T

# final_index = np.nonzero(np.array(time_hist) > 180)[0][0]
final_index = np.nonzero(np.array(t_upsampled) > 180)[0][0]
# for i, q_ in enumerate(config_hist[:final_index]):
for i, q_ in enumerate(config_upsampled[:final_index]):
    # q_rad = config_mapping(q_, "from_kinova")
    # kinova.add_ani_frame(time=time_hist[i], q=q_)
    # target.add_ani_frame(time=time_hist[i], initial_ind=0, final_ind=len(curve) - 1)
    kinova.add_ani_frame(time=t_upsampled[i], q=q_)
    target.add_ani_frame(time=t_upsampled[i], initial_ind=0, final_ind=len(curve) - 1)
sim.set_parameters(width=1480, height=960)
sim.save('./', 'real_exp')

# %%
