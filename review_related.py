# %%
import os
import pickle
import numpy as np
import plotly.graph_objects as go
from uaibot import Robot
from plotly.subplots import make_subplots

exp_case = "cylindrical"
if "cyl" in exp_case:
    curve_file_name = "cylindrical.npy"
elif "circ" in exp_case:
    curve_file_name = "circle.npy"
else:
    raise FileNotFoundError("This file is not part of this experiment")
data_path = "./data"
dcurve_file_name = "cylindrical_derivative.npy"
curve_path = os.path.join(data_path, curve_file_name)
dcurve_path = os.path.join(data_path, dcurve_file_name)
experiment_data_name = f"kinova_experiment_{exp_case}.pkl"
data_path = os.path.join(data_path, experiment_data_name)

with open(data_path, "rb") as f:
    data = pickle.load(f)

config_hist = data["hist_q"]
qdot_hist = data["hist_qdot"]
hist_index = data["hist_index"]
hist_dist = data["hist_dist"]
hist_time = data["hist_time"]
hist_time_vf = data["hist_time_vf"]

kinova = Robot.create_kinova_gen3(name="kinova")

curve = np.load(curve_path)


def get_pos_ori_error(state, closest_point):
    state = np.array(state)
    closest_point = np.array(closest_point)
    p_near = closest_point[:3, 3].ravel()
    ori_near = closest_point[:3, :3]
    p_curr = state[:3, 3].ravel()
    ori_curr = state[:3, :3]
    pos_err = np.linalg.norm(p_near - p_curr) * 100
    trace_ = np.trace(ori_near @ ori_curr.T)
    acos = np.arccos((trace_ - 1) / 2)
    # checks if acos is nan
    if np.isnan(acos):
        acos = 0
    ori_err = acos * 180 / np.pi
    return pos_err, ori_err


ori_errs = []
pos_errs = []

for i, q in enumerate(config_hist[:-1]):
    q_ = np.array(q.copy()).reshape(-1, 1)
    state = np.array(kinova.fkm(q=q_))
    closest_point = np.array(curve[hist_index[i]])

    pos_err, ori_err = get_pos_ori_error(state, closest_point)
    pos_errs.append(pos_err)
    ori_errs.append(ori_err)

final_index = len(hist_time)
init_index = 1

time_vec = np.array(hist_time[init_index:final_index]) - hist_time[init_index]
hist_dist = hist_dist[init_index:final_index]
pos_errs = pos_errs[init_index:final_index]
ori_errs = ori_errs[init_index:final_index]

# %%
steady_time = 40 if "circ" in exp_case else 30
steady_state = np.nonzero(time_vec > steady_time)[0][0]
dist_ss = np.array(hist_dist[steady_state:])
pos_ss = np.array(pos_errs[steady_state:])
ori_ss = np.array(ori_errs[steady_state:])

print(f"RMSE steady state distance: {np.sqrt(np.mean(dist_ss**2))}")
print(f"RMSE steady state pos error: {np.sqrt(np.mean(pos_ss**2))}")
print(f"RMSE steady state orientation error: {np.sqrt(np.mean(ori_ss**2))}")

print(f"mean+std steady state distance: {np.mean(dist_ss)} ± {np.std(dist_ss)}")
print(f"mean+std steady state pos error: {np.mean(pos_ss)} ± {np.std(pos_ss)}")
print(f"mean+std steady state orientation error: {np.mean(ori_ss)} ± {np.std(ori_ss)}")
