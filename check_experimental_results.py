#%%
import pickle
import numpy as np
import plotly.graph_objects as go
from uaibot import Robot
from plotly.subplots import make_subplots

with open("/home/fbartelt/Documents/Projetos/SE3vectorfield-kinova/data.pkl", "rb") as f:
    data = pickle.load(f)

config_hist = data["config_hist"]
hist_index = data["hist_index"]
hist_dist = data["hist_dist"]
hist_time = data["hist_time"]

kinova = Robot.create_kinova_gen3(name="kinova")

curve = np.load('/home/fbartelt/Documents/Projetos/SE3vectorfield-kinova/resampled_curve.npy')

ori_errs = []
pos_errs = []
for i, q in enumerate(config_hist[:-1]):
# for i in range(np.minimum(len(config_hist), len(hist_index))):
    # q = config_hist[i]
    q_ = np.array(q.copy()).reshape(-1, 1)
    state = np.array(kinova.fkm(q=q_))
    closest_point = np.array(curve[hist_index[i]])
    p_near = np.array(closest_point[:3, 3]).ravel()
    ori_near = np.array(closest_point[:3, :3])
    p_curr = np.array(state[:3, 3]).ravel()
    ori_curr = np.array(state[:3, :3])
    pos_errs.append(np.linalg.norm(p_near - p_curr) * 100)
    trace_ = np.trace(ori_near @ ori_curr.T)
    acos = np.arccos((trace_ - 1) / 2)
    # checks if acos is nan
    if np.isnan(acos):
        acos = 0
    ori_errs.append(acos * 180 / np.pi)
    # ori_errs.append(np.linalg.norm(np.eye(3) - ori_near @ ori_curr.T, 'fro'))

# makes a figure with two plots, one above another. First the position error, then the orientation error
dt = 0.01
# time_vec = np.arange(0, len(pos_errs) * dt, dt)
# time_vec = np.arange(0, len(pos_errs))
time_vec = hist_time
fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.02)
fig.add_trace(go.Scatter(x=time_vec, y=hist_dist, showlegend=False, line=dict(width=3)), row=1, col=1)
fig.add_trace(go.Scatter(x=time_vec, y=pos_errs, showlegend=False, line=dict(width=3)), row=2, col=1)
fig.add_trace(go.Scatter(x=time_vec, y=ori_errs, showlegend=False, line=dict(width=3)), row=3, col=1)
fig.update_xaxes(title_text="Time (s)", gridcolor='gray', zerolinecolor='gray', row=3, col=1)
fig.update_xaxes(title_text="", gridcolor='gray', zerolinecolor='gray', row=1, col=1)
fig.update_xaxes(title_text="", gridcolor='gray', zerolinecolor='gray', row=2, col=1)
fig.update_yaxes(title_text="Distance D", gridcolor='gray', zerolinecolor='gray', row=1, col=1, title_standoff=30)
fig.update_yaxes(title_text="Pos. error (cm)", gridcolor='gray', zerolinecolor='gray', row=2, col=1, title_standoff=30)
fig.update_yaxes(title_text="Ori. error (deg)", gridcolor='gray', zerolinecolor='gray', row=3, col=1, title_standoff=30)
fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))

fig.update_layout(plot_bgcolor='white', paper_bgcolor='white',
                  width=718.110, height=605.9155)
fig.show()
# %%
