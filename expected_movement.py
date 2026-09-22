# %%
import os
import numpy as np
from tqdm import tqdm
from uaibot import Robot, Utils, Simulation, Cylinder
from uaibot.simobjects.curve import CurveSE3
from plotly.subplots import make_subplots
# from precompute_curve import cylindrical_surface


def config_mapping(q, maptype="from_kinova"):
    if maptype == "from_kinova":
        q = np.deg2rad(np.array(q).ravel())
        delta = np.array([0] + [np.pi] * 6).ravel()
        q = q + delta
    # maptype == 'to_kinova'
    else:
        q = np.rad2deg(np.array(q).ravel())
        delta = np.array([0] + [180] * 6).ravel()
        q = q - delta
        q = np.array([qi % 360 for qi in q]).ravel()
    return q


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


# ----------------------------------------------------------------------
#                               LOAD CURVE
# ----------------------------------------------------------------------

path = "./data"
curve_file = "cylindrical.npy"
derivative_file = "cylindrical_derivative.npy"
curve_path = os.path.join(path, curve_file)
dcurve_path = os.path.join(path, derivative_file)
print(f"Loading raw curve from {curve_path}...")
curve = np.load(curve_path, allow_pickle=True)
print(f"Loading raw curve from {dcurve_path}...")
dcurve = np.load(dcurve_path, allow_pickle=True)

# %%
# ----------------------------------------------------------------------
#                               SETUP
# ----------------------------------------------------------------------
axial_amplitude = 0.1 / 2
n_axial_oscillations = 3
center = np.array([0.0, 0.25, 0.6])
radius = 0.07
htm = np.eye(4)
height = 2.0 * axial_amplitude + 0.05
n_points = 5000
cylinder_height = height
htm[:3, 3] = center

# curve, dcurve = cylindrical_surface(
#     n_points=n_points,
#     radius=radius,
#     axial_amplitude=axial_amplitude,
#     n_axial_oscillations=n_axial_oscillations,
#     center=center,
# )
curve_ub = CurveSE3(points=curve)


kinova = Robot.create_kinova_gen3()
target = CurveSE3(
    points=curve,
    size=0.02,
    color="cyan",
    frame_size=0.1,
    num_frames=10,
)

kinova.set_ani_frame(q=config_mapping([0, 10, 0, 15, 0, 40, 30], "from_kinova"))
sim = Simulation.create_sim_grid([kinova, target])
sim.set_parameters(width=1280, height=720)

cylinder = Cylinder(
    htm=htm,
    # 90% of original radius to force D->0
    radius=radius * 0.9,
    height=cylinder_height,
    opacity=0.3,
    color="blue",
)
sim.add(cylinder)

# ----------------------------------------------------------------------
#                            CONTROL LOOP
# ----------------------------------------------------------------------

T = 40.0
dt = 0.01
print_log = False # Prints the constraint violations

# Vector field
kt1, kt2, kt3 = 0.1, 1.0, 0.75
kn1, kn2 = 0.1 * 10, 0.75
# CBF
eta = 10.0
eta_lim = 1 / 1e-3
eta_self = 0.6
delta_collision = 0.005
gain_qp = 1.0
# Generalized distance
h_gdf, eps_gdf = 2e-3, 1e-3

q0 = np.array(kinova.q.copy())
q = q0
n = len(q)

imax = int(T / dt)
dist_hist = []
q_hist = []
qdot_hist = []
cost_hist = []
pos_errors, ori_errors = [], []

curve = [H for H in curve]
delta_from_real = np.array([0] + [np.pi] * 6).reshape(-1, 1)
q_min = np.array(kinova.joint_limit[:, 0]) + delta_from_real
q_max = np.array(kinova.joint_limit[:, 1]) + delta_from_real
stderr_ = ""

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
        curve_derivative=dcurve,
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

    # ------------------------------------------------------------------
    #                           QP with CBF
    # ------------------------------------------------------------------
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

    if print_log:
        violation = np.asarray(A_qp @ qdot_unconstrained - b_qp).ravel()
        active = violation < -1e-6
        total_active = np.sum(active)

        if total_active > 0:
            # Sizes of each block in A_qp
            n_env = A_qp_env.shape[0]  # = dist_struct.dist_vect.size
            n_self = A_qp_self.shape[0]  # = dist_struct_auto.dist_vect.size
            n_lim = A_lim.shape[0]  # = 2 * n

            # Flattened views of the vectors we need for reporting
            dist_env = np.asarray(dist_struct.dist_vect).ravel()
            dist_self = np.asarray(dist_struct_auto.dist_vect).ravel()
            b_env = np.asarray(b_qp_env).ravel()
            b_self = np.asarray(b_qp_self).ravel()
            b_lim_v = np.asarray(b_lim).ravel()
            q_flat = np.array(q_).ravel()
            q_min_f = np.asarray(q_min).ravel()
            q_max_f = np.asarray(q_max).ravel()

            lines = [f"Active Constraints: {total_active}"]
            for i in np.flatnonzero(active):
                v = violation[i]
                if i < n_env:
                    # environment (robot <-> obstacle) distance constraint
                    j = i
                    lines.append(
                        f"  [ENV  dist #{j}] violation={v:.3e}  "
                        f"dist={dist_env[j]:.4f}  b={b_env[j]:.3e}"
                    )
                elif i < n_env + n_self:
                    # self-collision distance constraint
                    j = i - n_env
                    lines.append(
                        f"  [SELF dist #{j}] violation={v:.3e}  "
                        f"dist={dist_self[j]:.4f}  b={b_self[j]:.3e}"
                    )
                else:
                    # joint limit
                    k = i - n_env - n_self
                    if k < n:
                        j = k
                        kind = "lower"
                        limit_val = q_min_f[j]
                    else:
                        j = k - n
                        kind = "upper"
                        limit_val = q_max_f[j]
                    lines.append(
                        f"  [JOINT {kind} #{j}] violation={v:.3e}  "
                        f"q[{j}]={q_flat[j]:.4f}  limit={limit_val:.4f}  b={b_lim_v[k]:.3e}"
                    )

            stderr_ = "\n" + "\n".join(lines)
            print(stderr_)
    else:
        stderr_ = ""

    pos_err, ori_err = get_pos_ori_error(H, curve[min_index])

    # ------------------------------------------------------------------
    #                        DAMPED PSEUDOINVERSE
    # ------------------------------------------------------------------
    # qdot = Utils.dp_inv(J, 1e-4) @ twist_

    # ------------------------------------------------------------------
    #                              LOG
    # ------------------------------------------------------------------
    q = q_ + qdot * dt
    kinova.add_ani_frame(time=i * dt, q=q)
    qdot_hist.append(qdot)
    dist_hist.append(min_dist)
    cost_hist.append(cost)
    pos_errors.append(pos_err)
    ori_errors.append(ori_err)

sim.run_in_browser()

# ----------------------------------------------------------------------
#                                  PLOT
# ----------------------------------------------------------------------
import plotly.graph_objects as go


def plot_errors():
    time_vec = np.arange(start=0, stop=T, step=dt)
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
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))

    return fig


plot_errors().show()
go.Figure(go.Scatter(y=cost_hist)).show()
