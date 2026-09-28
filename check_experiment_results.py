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

# %%
# ----------------------------------------------------------------------
#                                ANIMATION
# ----------------------------------------------------------------------


def animate_distance(
    distances,
    pos_errors,
    ori_errors,
    time_data,
    fig=None,
    total_duration=None,
):
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
    total_duration: float
        Total duration of the animation in seconds. If None, total time
        of time_data is used.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Resulting plotly figure.
    """
    width_ = 2
    gridcolor = "rgba(0, 0, 0, 0.2)"

    if total_duration is None:
        total_duration = time_data[-1] - time_data[0]

    n_frames = len(time_data)
    frame_duration_ms = 1000.0 * total_duration / n_frames

    if fig is None:
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.02)
        fig.add_trace(
            go.Scatter(x=time_data, y=distances, showlegend=False, line=dict(width=3)),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=time_data, y=pos_errors, showlegend=False, line=dict(width=3)),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=time_data, y=ori_errors, showlegend=False, line=dict(width=3)),
            row=3,
            col=1,
        )
        fig.update_xaxes(
            title_text="Time (s)",
            gridcolor=gridcolor,
            zerolinecolor="gray",
            zerolinewidth=width_,
            gridwidth=width_,
            row=3,
            col=1,
        )
        fig.update_xaxes(
            title_text="",
            gridcolor=gridcolor,
            zerolinecolor="gray",
            zerolinewidth=width_,
            gridwidth=width_,
            row=1,
            col=1,
        )
        fig.update_xaxes(
            title_text="",
            gridcolor=gridcolor,
            zerolinecolor="gray",
            zerolinewidth=width_,
            gridwidth=width_,
            row=2,
            col=1,
        )
        fig.update_yaxes(
            title_text="Distance D",
            gridcolor=gridcolor,
            zerolinecolor="gray",
            zerolinewidth=width_,
            gridwidth=width_,
            row=1,
            col=1,
            title_standoff=30,
        )
        fig.update_yaxes(
            title_text="Pos. error (cm)",
            gridcolor=gridcolor,
            zerolinecolor="gray",
            zerolinewidth=width_,
            gridwidth=width_,
            row=2,
            col=1,
            title_standoff=30,
        )
        fig.update_yaxes(
            title_text="Ori. error (deg)",
            gridcolor=gridcolor,
            zerolinecolor="gray",
            zerolinewidth=width_,
            gridwidth=width_,
            row=3,
            col=1,
            title_standoff=30,
        )
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=0), font=dict(size=16))
        fig.update_layout(
            plot_bgcolor="white", paper_bgcolor="white", width=1480, height=960
        )

    print("Creating frames")
    frames = [
        go.Frame(
            data=[
                go.Scatter(
                    x=time_data[:k],
                    y=distances[:k],
                    showlegend=False,
                    line=dict(width=3),
                    mode="lines",
                ),
                go.Scatter(
                    x=time_data[:k],
                    y=pos_errors[:k],
                    showlegend=False,
                    line=dict(width=3),
                    mode="lines",
                ),
                go.Scatter(
                    x=time_data[:k],
                    y=ori_errors[:k],
                    showlegend=False,
                    line=dict(width=3),
                    mode="lines",
                ),
            ],
            name=f"frame_{k}",
        )
        for k in range(len(time_data))
    ]

    fig.update(frames=frames)
    print("Addind layout")
    layout = go.Layout(
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
        yaxis=dict(
            range=[-1.1 * np.min(distances), np.max(distances) * 1.1], autorange=False
        ),
        yaxis2=dict(
            range=[-1.1 * np.min(pos_errors), np.max(pos_errors) * 1.1], autorange=False
        ),
        yaxis3=dict(
            range=[-1.1 * np.min(ori_errors), np.max(ori_errors) * 1.1], autorange=False
        ),
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
                                "frame": {
                                    "duration": frame_duration_ms,
                                    "redraw": False,
                                },
                                "fromcurrent": True,
                                "transition": {
                                    "duration": 0.0,
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
                direction="left",
                pad={"r": 10, "t": 87},
                showactive=False,
                x=0.1,
                xanchor="right",
                y=0,
                yanchor="top",
            )
        ],
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

fig = animate_distance(
    distances,
    pos_errors,
    ori_errors,
    time_data,
    fig=None,
    total_duration=180,
)
# fig.show()
# %%
import plotly.io as pio

# fig.update_layout(transition = {'duration': 90})
pio.write_html(fig, file="./plotly_animation.html", auto_play=False)
# %%
