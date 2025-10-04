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

curve = np.load('/home/fbartelt/Documents/Projetos/SE3vectorfield-kinova/resampled_curve2.npy')

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

final_index = np.nonzero(np.array(hist_time) > 170.845)[0][0]
init_index = np.nonzero(np.array(hist_dist) > 0.3961)[0][0]
time_vec = np.array(hist_time[init_index:final_index]) - hist_time[init_index]
hist_dist = hist_dist[init_index:final_index]
pos_errs = pos_errs[init_index:final_index]
ori_errs = ori_errs[init_index:final_index]
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

# fig.update_layout(plot_bgcolor='white', paper_bgcolor='white',
                #   width=718.110, height=605.9155)
fig.update_layout(plot_bgcolor='white', paper_bgcolor='white',
                  width=718.110, height=450)
# fig.update_layout(plot_bgcolor='white', paper_bgcolor='white',
#                   width=800, height=600)
fig.show()
# %%
"""CREATE ANIMATION"""
def animate_distance(distances, pos_errors, ori_errors, time_data, fig=None):
    """Create an animation of the distance metric between the object and the
    target curve, along with the position and orientation errors.

    Parameters
    ----------
    distances : list or np.ndarray
        List of EC-distances between the object and the target curve.
    pos_errors : list or np.ndarray
        List of position errors in centimeters.
    ori_errors : list or np.ndarray
        List of orientation errors in degrees.
    time_data : list or np.ndarray
        List of time values.
    fig : plotly.graph_objects.Figure, optional
        Existing figure to add the animation to. If None, a new figure is
        created. The default is None.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Resulting plotly figure.
    """
    width_ = 2
    gridcolor = 'rgba(0, 0, 0, 0.2)'
    if fig is None:
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.02)
        fig.add_trace(go.Scatter(x=time_data, y=distances, showlegend=False, line=dict(width=3)), row=1, col=1)
        fig.add_trace(go.Scatter(x=time_data, y=pos_errors, showlegend=False, line=dict(width=3)), row=2, col=1)
        fig.add_trace(go.Scatter(x=time_data, y=ori_errors, showlegend=False, line=dict(width=3)), row=3, col=1)
        fig.update_xaxes(title_text="Time (s)", gridcolor=gridcolor, zerolinecolor='gray', zerolinewidth=width_, gridwidth=width_, row=3, col=1)
        fig.update_xaxes(title_text="", gridcolor=gridcolor, zerolinecolor='gray', zerolinewidth=width_, gridwidth=width_, row=1, col=1)
        fig.update_xaxes(title_text="", gridcolor=gridcolor, zerolinecolor='gray', zerolinewidth=width_, gridwidth=width_, row=2, col=1)
        fig.update_yaxes(title_text="Distance D", gridcolor=gridcolor, zerolinecolor='gray', zerolinewidth=width_, gridwidth=width_, row=1, col=1, title_standoff=30)
        fig.update_yaxes(title_text="Pos. error (cm)", gridcolor=gridcolor, zerolinecolor='gray', zerolinewidth=width_, gridwidth=width_, row=2, col=1, title_standoff=30)
        fig.update_yaxes(title_text="Ori. error (deg)", gridcolor=gridcolor, zerolinecolor='gray', zerolinewidth=width_, gridwidth=width_, row=3, col=1, title_standoff=30)
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=0), font=dict(size=16))
        fig.update_layout(plot_bgcolor='white', paper_bgcolor='white', width=1480, height=960)
    
    print("Creating frames")
    frames = [go.Frame(data=[
            go.Scatter(x=time_data[:k], y=distances[:k], showlegend=False, line=dict(width=3), mode='lines'),
            go.Scatter(x=time_data[:k], y=pos_errors[:k], showlegend=False, line=dict(width=3), mode='lines'),
            go.Scatter(x=time_data[:k], y=ori_errors[:k], showlegend=False, line=dict(width=3), mode='lines'),
        ], name=f'frame_{k}') for k in range(len(time_data))]
    
    fig.update(frames=frames)
    print("Addind layout")
    layout=go.Layout(
            # width=600,
            # height=600,
            # margin=dict(r=5, l=5, b=5, t=5),
            # xaxis=dict(range=[0, time_data[-1]], autorange=False, title="Time (s)"),
            # yaxis=dict(
            #     range=[0, np.], autorange=False, title="Value of metric <i>D</i>"
            # ),
            xaxis=dict(range=[-0.1, time_data[-1]], autorange=False),
            # xaxis=dict(autorange=True),
            # xaxis2=dict(zeroline=False),
            # xaxis3=dict(zeroline=False),
            yaxis=dict(range=[-1.1*np.min(distances), np.max(distances)*1.1], autorange=False),
            yaxis2=dict(range=[-1.1*np.min(pos_errors), np.max(pos_errors)*1.1], autorange=False),
            yaxis3=dict(range=[-1.1*np.min(ori_errors), np.max(ori_errors)*1.1], autorange=False),
            updatemenus=[
                dict(
                    type="buttons",
                    buttons=[
                        dict(
                            label="Play",
                            method="animate",
                            args=[
                                None,
                                {
                                    "frame": {"duration": 82.9999, "redraw": False},
                                    "fromcurrent": True,
                                    "transition": {
                                        "duration": 0.01,
                                        "easing": "cubic-in-out",
                                    },
                                },
                            ],
                        ),
                        dict(
                            label="Pause",
                            method="animate",
                            args=[
                                [None],
                                {
                                    "frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate",
                                    "transition": {"duration": 0},
                                },
                            ],
                        ),
                    ],
                    direction= "left",
                    pad= {"r": 10, "t": 87},
                    showactive= False,
                    x= 0.1,
                    xanchor= "right",
                    y= 0,
                    yanchor= "top"
                )
            ]
    )

    fig.update_layout(layout)
    # fig.update_xaxes(zeroline=True, row=1, col=1, zerolinecolor='gray', gridwidth=1.2, gridcolor='gray')
    # fig.update_xaxes(zeroline=True, row=2, col=1, zerolinecolor='gray', gridwidth=1.2, gridcolor='gray')
    # fig.update_xaxes(zeroline=True, row=3, col=1, zerolinecolor='gray', gridwidth=1.2, gridcolor='gray')
    # fig.update_yaxes(zeroline=True, row=1, col=1, zerolinecolor='gray', gridwidth=1.2, gridcolor='gray')
    # fig.update_yaxes(zeroline=True, row=2, col=1, zerolinecolor='gray', gridwidth=1.2, gridcolor='gray')
    # fig.update_yaxes(zeroline=True, row=3, col=1, zerolinecolor='gray', gridwidth=1.2, gridcolor='gray')

    return fig

skip = 10
time_data = time_vec[::skip]
pos_errors = pos_errs[::skip]
ori_errors = ori_errs[::skip]
distances = hist_dist[::skip]

fig = animate_distance(distances, pos_errors, ori_errors, time_data, fig=None)
# fig.show()
# %%
import plotly.io as pio
# fig.update_layout(transition = {'duration': 90})
pio.write_html(fig, file='./plotly_animation.html', auto_play=False)
# %%
