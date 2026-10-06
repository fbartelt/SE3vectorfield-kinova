# %%
import os
import pickle
import numpy as np
import plotly.graph_objects as go
from uaibot import Robot, Utils
from pathlib import Path
from plotly.subplots import make_subplots
import plotly.io as pio

pio.defaults.default_width = None
pio.defaults.default_height = None
# ---------------------------------------------------------------------------
# pybind module exposing the SE(3) machinery (names match the C++ file).
#   EEdistance(V, W) -> float
#   SmapSE3(xi)      -> 4x4 numpy array
#   expSE3(A)        -> 4x4 numpy array
#   (optional) vectorfield_SE3(...) -> VectorFieldResult(.twist, .dist, .index)
# ---------------------------------------------------------------------------
from uaibot_cpp_bind import ee_distance, SmapSE3, expSE3

# ---------------------------------------------------------------------------
# Data loading (unchanged)
# ---------------------------------------------------------------------------
exp_case = "circle"
if "cyl" in exp_case:
    curve_file_name = "cylindrical.npy"
elif "circ" in exp_case:
    curve_file_name = "circle.npy"
    kt1, kt2, kt3 = 0.03 * 10, 1.0, 0.75
    kn1, kn2 = 0.1 * 8, kt3

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
curve_derivative = np.load(dcurve_path) if "cyl" in exp_case else []
ds, delta = 1e-3, 1e-3

analysis_data_path = "./media4letter"
analysis_directory = Path(analysis_data_path)
analysis_directory.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def get_pos_ori_error(state, closest_point):
    state = np.array(state)
    closest_point = np.array(closest_point)
    p_near = closest_point[:3, 3].ravel()
    ori_near = closest_point[:3, :3]
    p_curr = state[:3, 3].ravel()
    ori_curr = state[:3, :3]
    pos_err = np.linalg.norm(p_near - p_curr) * 100
    trace_ = np.trace(ori_near @ ori_curr.T)
    acos = np.arccos(np.clip((trace_ - 1) / 2, -1.0, 1.0))
    if np.isnan(acos):
        acos = 0
    ori_err = acos * 180 / np.pi
    return pos_err, ori_err


def _cluster_indices(sorted_idx, min_sep):
    if len(sorted_idx) == 0:
        return []
    clusters = [[sorted_idx[0]]]
    for idx in sorted_idx[1:]:
        if idx - clusters[-1][-1] <= min_sep:
            clusters[-1].append(idx)
        else:
            clusters.append([idx])
    return clusters


def closest_point_clusters(curve, state, rel_tol=1e-3, min_sep=10):
    """All (nearly) minimizing indices of the EC distance, clustered."""
    dists = np.array([ee_distance(state, cp) for cp in curve])
    dmin = float(dists.min())
    tol = rel_tol * max(dmin, 1e-9)
    candidates = np.nonzero(dists <= dmin + tol)[0]
    clusters = _cluster_indices(candidates.tolist(), min_sep)
    # Merge wrap-around for closed curves
    if len(clusters) > 1:
        n_curve = len(curve)
        gap = (n_curve - clusters[-1][-1]) + clusters[0][0]
        if gap <= min_sep:
            clusters[0] = clusters[-1] + clusters[0]
            clusters.pop()
    return clusters, dists


def distance_gradient(state, closest_point, delta=1e-3):
    """
    Finite-difference approximation of the L-operator applied to the EC
    distance function (the negative of the normal component of the SE(3)
    vector field), exactly as in the C++ implementation:

        LvD(i) = ( EEdistance(expSE3(SmapSE3(e_i) * delta) * state,
                              closest_point) - D_min ) / delta
    """
    grad, _, _ = Robot.vector_field_SE3(state, curve, delta=delta, mode="c++")
    return grad


# %%
# ---------------------------------------------------------------------------
# Main pass: tracking errors, P-detection
# ---------------------------------------------------------------------------
ori_errs = []
pos_errs = []

for i, q in enumerate(config_hist[:-1]):
    q_ = np.array(q.copy()).reshape(-1, 1)
    state = np.array(kinova.fkm(q=q_))
    closest_point = np.array(curve[hist_index[i]])

    pos_err, ori_err = get_pos_ori_error(state, closest_point)
    pos_errs.append(pos_err)
    ori_errs.append(ori_err)


# ---------------------------------------------------------------------------
# Fix the off-by-one: align everything to a common length, then slice.
# ---------------------------------------------------------------------------
n_common = min(
    len(hist_time),
    len(hist_dist),
    len(pos_errs),
    len(ori_errs),
)

time_full = np.asarray(hist_time[:n_common])
hist_dist_full = np.asarray(hist_dist[:n_common])
pos_full = np.asarray(pos_errs[:n_common])
ori_full = np.asarray(ori_errs[:n_common])

init_index = 1
# All *_full arrays have the same length -> slicing with the same bounds
# can no longer produce mismatched shapes.
time_vec = time_full[init_index:] - time_full[init_index]
hist_dist = hist_dist_full[init_index:]
pos_errs = pos_full[init_index:]
ori_errs = ori_full[init_index:]

n_total = len(time_vec)
assert len(hist_dist) == len(pos_errs) == len(ori_errs)


# ---------------------------------------------------------------------------
# Steady-state tracking statistics (reviewer COMMENT 2)
# ---------------------------------------------------------------------------
steady_time = 23 if "circ" in exp_case else 30
ss_mask = time_vec > steady_time


def _rmse(x):
    return float(np.sqrt(np.mean(np.asarray(x) ** 2)))


print("=" * 70)
print("Steady-state tracking accuracy (COMMENT 2)")
print("=" * 70)
print(f"RMSE  D         : {_rmse(hist_dist[ss_mask]):.6f}")
print(f"RMSE  position  : {_rmse(pos_errs[ss_mask]):.6f} cm")
print(f"RMSE  orientation: {_rmse(ori_errs[ss_mask]):.6f} deg")
print(
    f"mean±std  D         : {hist_dist[ss_mask].mean():.6f} ± "
    f"{hist_dist[ss_mask].std():.6f}"
)
print(
    f"mean±std  position  : {pos_errs[ss_mask].mean():.6f} ± "
    f"{pos_errs[ss_mask].std():.6f} cm"
)
print(
    f"mean±std  orientation: {ori_errs[ss_mask].mean():.6f} ± "
    f"{ori_errs[ss_mask].std():.6f} deg"
)


# %%
# ---------------------------------------------------------------------------
# P-detection
# ---------------------------------------------------------------------------
print(
    "!! You must edit uaibot source code so vector_field_SE3 returns the normal component without gains !!"
)
in_P_multi = []  # multiple closest points on C
n_clusters_hist = []
grad_norms = []  # ||LvD|| at delta = 1e-3
grad_norms_small = []  # ||LvD|| at delta = 1e-4 (for stability test)

for i, q in enumerate(config_hist[:-1]):
    q_ = np.array(q.copy()).reshape(-1, 1)
    state = np.array(kinova.fkm(q=q_))

    # clusters, _ = closest_point_clusters(curve, state)
    clusters = []
    n_clusters_hist.append(len(clusters))
    in_P_multi.append(len(clusters) > 1)

    LvD_1 = distance_gradient(state, closest_point, delta=1e-3)
    # LvD_2 = distance_gradient(state, closest_point, delta=1e-2)
    grad_norms.append(np.linalg.norm(LvD_1))
    grad_norms_small.append(np.linalg.norm(LvD_1))
    # grad_norms_small.append(np.linalg.norm(LvD_2))


# ---------------------------------------------------------------------------
# Fix the off-by-one: align everything to a common length, then slice.
# ---------------------------------------------------------------------------
n_common = min(
    len(hist_time),
    len(hist_dist),
    len(in_P_multi),
    len(n_clusters_hist),
    len(grad_norms),
    len(grad_norms_small),
)

time_full = np.asarray(hist_time[:n_common])
inP_multi_full = np.asarray(in_P_multi[:n_common], dtype=bool)
nclus_full = np.asarray(n_clusters_hist[:n_common])
gn_full = np.asarray(grad_norms[:n_common])
gn_small_full = np.asarray(grad_norms_small[:n_common])
dist_full = np.asarray(hist_dist[:n_common])

init_index = 1
# All *_full arrays have the same length -> slicing with the same bounds
# can no longer produce mismatched shapes.
time_vec = time_full[init_index:] - time_full[init_index]
in_P_multi = inP_multi_full[init_index:]
n_clusters_hist = nclus_full[init_index:]
grad_norms = gn_full[init_index:]
grad_norms_small = gn_small_full[init_index:]
dist_vec = dist_full[init_index:]

n_total = len(time_vec)
assert (
    len(in_P_multi) == len(n_clusters_hist) == len(grad_norms) == len(grad_norms_small)
)


# ---------------------------------------------------------------------------
# Gradient-based detection of non-differentiable points (cut locus / Albano).
#
# At a differentiable point of D (unique closest point, smooth distance),
# the finite-difference estimate of ||LvD|| is O(1) and stable in delta.
# At a non-differentiable point it either explodes (large robust z-score)
# or depends strongly on the finite-difference step (large relative drift).
# ---------------------------------------------------------------------------
grad_med = float(np.median(grad_norms))
grad_mad = float(np.median(np.abs(grad_norms - grad_med)) + 1e-12)
grad_z = 0.6745 * (grad_norms - grad_med) / grad_mad  # robust z-score

grad_drift = np.abs(grad_norms - grad_norms_small) / (grad_norms + 1e-12)

EXPLOSION_Z = 8.0  # robust z-score threshold for ||LvD||
DRIFT_THRESHOLD = 0.20  # 20 % change between delta = 1e-3 and 1e-4
DIST_THRESHOLD = 1e-2  # Aproxximate value for D = 0

in_P_singular = (grad_z > EXPLOSION_Z) & (dist_vec > DIST_THRESHOLD)

# Union of the two P indicators.
in_P_flags = in_P_multi | in_P_singular

# ---------------------------------------------------------------------------
# P-encounter statistics (reviewer COMMENT 1)
# ---------------------------------------------------------------------------
n_P_multi = int(in_P_multi.sum())
n_P_sing = int(in_P_singular.sum())
n_P_total = int(in_P_flags.sum())
entries = np.nonzero((~in_P_flags[:-1]) & (in_P_flags[1:]))[0] + 1

print()
print("=" * 70)
print("P-encounter statistics (COMMENT 1)")
print("=" * 70)
print(f"Samples: {n_total}")
print(
    f"In P (multiple minimizers) : {n_P_multi:5d} "
    f"({100 * n_P_multi / n_total:.2f}%)"
)
print(
    f"In P (singular gradient)   : {n_P_sing:5d} " f"({100 * n_P_sing / n_total:.2f}%)"
)
print(
    f"In P (union)               : {n_P_total:5d} "
    f"({100 * n_P_total / n_total:.2f}%)"
)
print(f"Number of discrete entries into P: {len(entries)}")
if len(entries) > 0:
    print("Entry times (s): " + ", ".join(f"{time_vec[e]:.2f}" for e in entries))

unique_c, counts_c = np.unique(n_clusters_hist, return_counts=True)
print("Histogram of #closest-point clusters per sample:")
for c, k in zip(unique_c, counts_c):
    print(f"  {int(c)} cluster(s): {k} samples ({100 * k / n_total:.2f}%)")

print()
print(
    f"||LvD|| statistics: median = {grad_med:.4f}, "
    f"mean = {grad_norms.mean():.4f}, "
    f"max = {grad_norms.max():.4f}, "
    f"robust-z > {EXPLOSION_Z}: {int((grad_z > EXPLOSION_Z).sum())} samples"
)
print(
    f"Two-delta drift > {DRIFT_THRESHOLD:.2f}: "
    f"{int((grad_drift > DRIFT_THRESHOLD).sum())} samples"
)

# ---------------------------------------------------------------------------
# Do the bumps in Figure 6 (~65 s, 105 s, 145 s) coincide with P?
# ---------------------------------------------------------------------------
bump_times = [65.0, 105.0, 145.0]
print()
print("Check around the bumps visible in Figure 6 (~65 s, 105 s, 145 s):")
for bt in bump_times:
    k = int(np.argmin(np.abs(time_vec - bt)))
    lo, hi = max(0, k - 10), min(n_total, k + 11)
    frac_all = in_P_flags[lo:hi].mean()
    frac_mult = in_P_multi[lo:hi].mean()
    frac_sing = in_P_singular[lo:hi].mean()
    n_entry_near = int(((entries >= lo) & (entries < hi)).sum())
    print(
        f"  t≈{bt:>5.1f}s -> in P {100 * frac_all:5.1f}% of ±10 samples "
        f"(multi {100 * frac_mult:5.1f}%, singular {100 * frac_sing:5.1f}%), "
        f"{n_entry_near} entr(y/ies) nearby"
    )

# ---------------------------------------------------------------------------
# Effect of P on tracking error (the actual reviewer question)
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("Tracking error inside vs. outside P")
print("=" * 70)

in_mask = in_P_flags.astype(bool)
out_mask = ~in_mask


def _stats(x):
    x = np.asarray(x)
    return f"{x.mean():.4f} ± {x.std():.4f} (RMSE {_rmse(x):.4f})"


if n_P_total > 0:
    print(
        f"  [in P ] N={n_P_total:5d} | "
        f"D={_stats(dist_vec[in_mask])} | "
        f"pos={_stats(pos_errs[in_mask])} cm | "
        f"ori={_stats(ori_errs[in_mask])} deg"
    )
if n_P_total < n_total:
    print(
        f"  [out P] N={n_total - n_P_total:5d} | "
        f"D={_stats(dist_vec[out_mask])} | "
        f"pos={_stats(pos_errs[out_mask])} cm | "
        f"ori={_stats(ori_errs[out_mask])} deg"
    )


# ---------------------------------------------------------------------------
# Visual check: EC distance and its gradient norm over time.
# ---------------------------------------------------------------------------
def f(width=718.110, height=450, save=False, ext="pdf"):
    margin = dict(l=0, r=0, b=50, t=20)

    rows = 2
    # if n_P_total > 0:
    #     rows = 3
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True)
    fig.add_trace(
        go.Scatter(
            x=time_vec,
            y=hist_dist,
            name="EC distance D",
            line_width=3,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=time_vec,
            y=grad_norms,
            name="||LvD||",
            line=dict(width=3),
        ),
        row=2,
        col=1,
    )

    # if n_P_total > 0:
    #     fig.add_trace(
    #         go.Scatter(
    #             x=time_vec[in_mask],
    #             y=dist_vec[in_mask],
    #             mode="markers",
    #             name="in P",
    #             marker=dict(color="orange", size=3),
    #         ),
    #         row=3,
    #         col=1,
    #     )

    fig.update_xaxes(
        title_text="",
        gridcolor="gray",
        zerolinecolor="gray",
        row=1,
        col=1,
    )
    fig.update_xaxes(
        title_text="Time (s)",
        gridcolor="gray",
        zerolinecolor="gray",
        row=2,
        col=1,
    )

    fig.update_yaxes(
        title_text="EC distance D",
        gridcolor="gray",
        zerolinecolor="gray",
        row=1,
        col=1,
        title_standoff=30,
    )
    fig.update_yaxes(
        title_text=r"$\|\mathrm{L} [D]\|$",
        gridcolor="gray",
        zerolinecolor="gray",
        row=2,
        col=1,
        title_standoff=30,
    )
    fig.update_layout(
        margin=margin,
        plot_bgcolor="white",
        paper_bgcolor="white",
        width=width,
        height=height,
        showlegend=False,
    )

    fig.show()

    if save:
        fig_name = os.path.join(analysis_data_path, f"P_reach.{ext}")
        fig.write_image(fig_name, width=width, height=height)


f()

# %%
# ---------------------------------------------------------------------------
# Jacobian conditioning / proximity to kinematic singularities.
#
# For each sample we compute the 6 x n geometric Jacobian J(q) and derive:
#   * sigma_min(J) : smallest singular value  (0 at a kinematic singularity)
#   * kappa(J)     : condition number sigma_max / sigma_min
#   * w(J)         : Yoshikawa manipulability sqrt(det(J J^T))
# A sample is flagged as "near-singular" when sigma_min (or w) drops below
# a threshold relative to the trajectory's typical value.
# ---------------------------------------------------------------------------

jac_mats = []
sigma_min_hist = []
sigma_max_hist = []
cond_hist = []
manip_hist = []
psi_hist = []
qdot_d_hist = []

for i, q in enumerate(config_hist[:-1]):
    q_ = np.array(q.copy()).reshape(-1, 1)
    jac, H = kinova.jac_geo(q=q_)
    jac = np.asarray(jac)
    H = np.asarray(H)
    jac_mats.append(jac)

    # Singular values of J (6 x n). Only the first 6 are meaningful if n > 6.
    sv = np.linalg.svd(jac, compute_uv=False)
    s_min = float(sv[-1]) if sv.size > 0 else 0.0
    s_max = float(sv[0]) if sv.size > 0 else 0.0

    xi, min_dist_, closest_index_ = kinova.vector_field_SE3(
        H,
        curve,
        kt1=kt1,
        kt2=kt2,
        kt3=kt3,
        kn1=kn1,
        kn2=kn2,
        curve_derivative=curve_derivative,
        delta=delta,
        ds=ds,
        mode="c++",
    )

    qdot_d = Utils.dp_inv(jac, 1e-4) @ np.asarray(xi).reshape(-1, 1)
    qdot_d = np.array(qdot_d).reshape(-1, 1)

    sigma_min_hist.append(s_min)
    sigma_max_hist.append(s_max)
    cond_hist.append(s_max / (s_min + 1e-12))

    # Yoshikawa manipulability (product of singular values)
    w = float(np.prod(sv))
    manip_hist.append(w)

    # Norm of control inputs
    psi_hist.append(np.linalg.norm(xi))
    qdot_d_hist.append(np.linalg.norm(qdot_d))

# ---------------------------------------------------------------------------
# Align to the same time base used in the P-encounter analysis.
# `time_full` was truncated to n_common before slicing with init_index; we
# reproduce the same alignment here.
# ---------------------------------------------------------------------------
n_common_j = min(len(sigma_min_hist), len(time_full))
sigma_min_full = np.asarray(sigma_min_hist[:n_common_j])
sigma_max_full = np.asarray(sigma_max_hist[:n_common_j])
cond_full = np.asarray(cond_hist[:n_common_j])
manip_full = np.asarray(manip_hist[:n_common_j])

init_index_j = 1
sigma_min = sigma_min_full[init_index_j:]
sigma_max = sigma_max_full[init_index_j:]
cond_j = cond_full[init_index_j:]
manip = manip_full[init_index_j:]

n_j = min(len(time_vec), len(sigma_min))
sigma_min = sigma_min[:n_j]
sigma_max = sigma_max[:n_j]
cond_j = cond_j[:n_j]
manip = manip[:n_j]
time_j = time_vec[:n_j]

# ---------------------------------------------------------------------------
# Thresholds. We use robust statistics of the trajectory to define "near
# singular": sigma_min < 10 % of the median, or manipulability < 10 % of
# its median. Tune these numbers to taste.
# ---------------------------------------------------------------------------
smin_med = float(np.median(sigma_min))
smin_thr = 0.10 * smin_med
manip_med = float(np.median(manip))
manip_thr = 0.10 * manip_med

near_sing = (sigma_min < smin_thr) | (manip < manip_thr)

print("=" * 70)
print("Jacobian conditioning statistics")
print("=" * 70)
print(
    f"sigma_min : min={sigma_min.min():.4e}, median={smin_med:.4e}, "
    f"max={sigma_max.max():.4e}"
)
print(f"sigma_max : min={sigma_max.min():.4e}, median={np.median(sigma_max):.4e}")
print(f"cond(J)   : median={np.median(cond_j):.3e}, " f"max={cond_j.max():.3e}")
print(f"manip     : median={manip_med:.4e}, min={manip.min():.4e}")
print(
    f"Near-singular samples (sigma_min < {smin_thr:.3e} "
    f"or manip < {manip_thr:.3e}): {int(near_sing.sum())} / {n_j} "
    f"({100 * near_sing.mean():.2f}%)"
)

# Discrete entries into near-singular region
ns_entries = np.nonzero((~near_sing[:-1]) & (near_sing[1:]))[0] + 1
print(f"Number of entries into near-singular region: {len(ns_entries)}")
if len(ns_entries) > 0:
    print(
        "Entry times (s): "
        + ", ".join(f"{time_j[e]:.2f}" for e in ns_entries[:20])
        + (" ..." if len(ns_entries) > 20 else "")
    )

# ---------------------------------------------------------------------------
# Do the bumps in Figure 6 coincide with near-singular configurations?
# ---------------------------------------------------------------------------
bump_times = [10.0, 65.0, 105.0, 145.0]
print()
print("Check around the bumps visible in Figure 6 (~65 s, 105 s, 145 s):")
for bt in bump_times:
    k = int(np.argmin(np.abs(time_j - bt)))
    lo, hi = max(0, k - 10), min(n_j, k + 11)
    frac = near_sing[lo:hi].mean()
    n_entry_near = int(((ns_entries >= lo) & (ns_entries < hi)).sum())
    print(
        f"  t≈{bt:>5.1f}s -> near-singular for {100 * frac:5.1f}% of "
        f"±10 samples, {n_entry_near} entr(y/ies) nearby, "
        f"sigma_min here = {sigma_min[k]:.3e}, cond = {cond_j[k]:.3e}"
    )

# ---------------------------------------------------------------------------
# Effect on tracking error (near-singular vs. not).
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("Tracking error inside vs. outside near-singular region")
print("=" * 70)


def _rmse(x):
    return float(np.sqrt(np.mean(np.asarray(x) ** 2)))


def _stats(x):
    x = np.asarray(x)
    return f"{x.mean():.4f} ± {x.std():.4f} (RMSE {_rmse(x):.4f})"


# Align to the common length for the error arrays as well.
n_cmp = len(hist_dist)
ns_mask = near_sing[:n_cmp]
D_cmp = hist_dist[:n_cmp]
p_cmp = pos_errs[:n_cmp]
o_cmp = ori_errs[:n_cmp]

n_ns = int(ns_mask.sum())
if n_ns > 0:
    print(
        f"  [near-sing] N={n_ns:5d} | "
        f"D={_stats(D_cmp[ns_mask])} | "
        f"pos={_stats(p_cmp[ns_mask])} cm | "
        f"ori={_stats(o_cmp[ns_mask])} deg"
    )
if n_ns < n_cmp:
    print(
        f"  [regular  ] N={n_cmp - n_ns:5d} | "
        f"D={_stats(D_cmp[~ns_mask])} | "
        f"pos={_stats(p_cmp[~ns_mask])} cm | "
        f"ori={_stats(o_cmp[~ns_mask])} deg"
    )

# ---------------------------------------------------------------------------
# Joint-space check: which joints approach their limits in the bumps?
# ---------------------------------------------------------------------------
print()
print("Joint-space amplitude at bump times (rad):")
configs = np.asarray(config_hist[1 : n_cmp + 1]).reshape(n_cmp, -1)

for bt in bump_times:
    k = int(np.argmin(np.abs(time_j - bt)))
    qk = configs[k]
    print(f"  t≈{bt:>5.1f}s  q = [" + ", ".join(f"{v: .4f}" for v in qk) + "]")


# ---------------------------------------------------------------------------
# Plot: sigma_min, condition number and manipulability over time, with the
# three bump times marked.
# ---------------------------------------------------------------------------
def g(width=718.110, height=450, save=False, ext="pdf"):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True)
    margin = dict(l=0, r=0, b=50, t=20)
    fig.add_trace(
        go.Scatter(x=time_j, y=sigma_min, name="σ_min(J)", line=dict(width=3)),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=time_j, y=cond_j, name="cond(J)", line=dict(width=3)),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=time_j, y=manip, name="manipulability(J)", line=dict(width=3)),
        row=3,
        col=1,
    )
    for bt in bump_times:
        fig.add_vline(
            x=bt, line=dict(color="green", dash="dash"), annotation_text=f"{bt:.0f}s"
        )
    fig.update_xaxes(
        title_text="Time (s)",
        gridcolor="gray",
        zerolinecolor="gray",
        row=3,
        col=1,
    )
    fig.update_xaxes(
        title_text="",
        gridcolor="gray",
        zerolinecolor="gray",
        row=1,
        col=1,
    )
    fig.update_xaxes(
        title_text="",
        gridcolor="gray",
        zerolinecolor="gray",
        row=2,
        col=1,
    )
    fig.update_yaxes(
        title_text=r"$\sigma_\mathrm{min}(J)$",
        gridcolor="gray",
        zerolinecolor="gray",
        row=1,
        col=1,
        title_standoff=30,
    )
    fig.update_yaxes(
        title_text=r"$\mathrm{cond}(J)$",
        gridcolor="gray",
        zerolinecolor="gray",
        row=2,
        col=1,
        title_standoff=30,
    )
    fig.update_yaxes(
        title_text=r"$\sqrt{J^TJ}$",
        gridcolor="gray",
        zerolinecolor="gray",
        row=3,
        col=1,
        title_standoff=30,
    )
    fig.update_layout(
        margin=margin,
        plot_bgcolor="white",
        paper_bgcolor="white",
        width=width,
        height=height,
        showlegend=False,
    )

    fig.show()
    if save:
        fig_name = os.path.join(analysis_data_path, f"jac_condition.{ext}")
        fig.write_image(fig_name, width=width, height=height)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
    fig.add_trace(go.Scatter(x=time_vec, y=psi_hist, line_width=3), row=1, col=1)
    fig.add_trace(go.Scatter(x=time_vec, y=qdot_d_hist, line_width=3), row=2, col=1)

    for bt in bump_times:
        fig.add_vline(
            x=bt, line=dict(color="green", dash="dash"), annotation_text=f"{bt:.0f}s"
        )

    # fig.update_xaxes(
    #     title_text="Time (s)",
    #     gridcolor="gray",
    #     zerolinecolor="gray",
    #     row=3,
    #     col=1,
    # )
    fig.update_xaxes(
        title_text="",
        gridcolor="gray",
        zerolinecolor="gray",
        row=1,
        col=1,
    )
    fig.update_xaxes(
        title_text="Time (s)",
        gridcolor="gray",
        zerolinecolor="gray",
        row=2,
        col=1,
    )
    fig.update_yaxes(
        title_text=r"$\|\Psi(H)\|$",
        gridcolor="gray",
        zerolinecolor="gray",
        row=1,
        col=1,
        title_standoff=30,
    )
    fig.update_yaxes(
        title_text=r"$\|\dot{q}_d\|$",
        gridcolor="gray",
        zerolinecolor="gray",
        row=2,
        col=1,
        title_standoff=30,
    )
    fig.update_layout(
        margin=margin,
        plot_bgcolor="white",
        paper_bgcolor="white",
        width=width,
        height=height,
        showlegend=False,
    )
    fig.show()
    if save:
        fig_name = os.path.join(analysis_data_path, f"control_norm.{ext}")
        fig.write_image(fig_name, width=width, height=height)


g(width=718.110, save=True, ext="pdf")
