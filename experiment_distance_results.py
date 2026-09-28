# %%
import os
import pickle
import numpy as np
import plotly.graph_objects as go
from uaibot import Robot
from plotly.subplots import make_subplots

exp_case = "circle"
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


def plot_errors(
    time_vec,
    dist_hist,
    pos_errors,
    ori_errors,
    width=718.110,
    height=450,
):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.02)
    fig.add_trace(
        go.Scatter(x=time_vec, y=dist_hist, showlegend=False, line=dict(width=3)),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=time_vec, y=pos_errors, showlegend=False, line=dict(width=3)),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=time_vec, y=ori_errors, showlegend=False, line=dict(width=3)),
        row=3,
        col=1,
    )
    fig.update_xaxes(
        title_text="Time (s)", gridcolor="gray", zerolinecolor="gray", row=3, col=1
    )
    fig.update_xaxes(
        title_text="", gridcolor="gray", zerolinecolor="gray", row=1, col=1
    )
    fig.update_xaxes(
        title_text="", gridcolor="gray", zerolinecolor="gray", row=2, col=1
    )
    fig.update_yaxes(
        title_text="Distance D",
        gridcolor="gray",
        zerolinecolor="gray",
        row=1,
        col=1,
        title_standoff=30,
    )
    fig.update_yaxes(
        title_text="Pos. error (cm)",
        gridcolor="gray",
        zerolinecolor="gray",
        row=2,
        col=1,
        title_standoff=30,
    )
    fig.update_yaxes(
        title_text="Ori. error (deg)",
        gridcolor="gray",
        zerolinecolor="gray",
        row=3,
        col=1,
        title_standoff=30,
    )
    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=0),
        plot_bgcolor="white",
        paper_bgcolor="white",
        width=width,
        height=height,
    )

    return fig


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

fig = plot_errors(time_vec, hist_dist, pos_errs, ori_errs)

fig.show()
