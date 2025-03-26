import numpy as np
import time
from multiprocessing import shared_memory
from uaibot import Robot, Utils, Simulation, PointCloud
import pickle


def circle_rn(n_points, u, v, radius=1.0, center=None, mid=False, dv=10.0):
    points = []
    n = u.shape[0]
    u = np.array(u).ravel()
    v = np.array(v).ravel()

    if center is None:
        center = np.zeros((n,))
    else:
        center = np.array(center).ravel()

    if isinstance(radius, (int, float)):
        radius = np.array([radius] * 2)
    elif isinstance(radius, (list, tuple, np.ndarray)) and len(radius) > 2:
        print("Radius must be a scalar or a 2D vector")
        return None
    else:
        radius = np.array(radius).ravel()

    for i in range(n_points):
        angle = 2 * np.pi * i / n_points
        p_nolimit = (radius[0] * np.cos(angle)) * u
        if mid:
            p_limit = (radius[1] * (np.sin(angle) + 1)) / 2 * v + np.deg2rad(dv)
        else:
            p_limit = (radius[1] * np.sin(angle)) * v
        points.append(center + p_nolimit + p_limit)

    points = np.array(points)
    return points


def Smap(xi):
    S_ = np.eye(4)
    S_[0, 1] = -xi[5]
    S_[0, 2] = xi[4]
    S_[1, 2] = -xi[3]
    S_ = S_ - S_.T
    S_[0, 3] = xi[0]
    S_[1, 3] = xi[1]
    S_[2, 3] = xi[2]
    return S_


def expSE3(A):
    S_ = A[:3, :3]
    v = A[:3, 3]
    res = np.eye(4)
    theta = np.sqrt(S_[1, 0] ** 2 + S_[0, 2] ** 2 + S_[2, 1] ** 2)

    if theta < 1e-6:
        R = np.eye(3)
        res[:3, :3] = R
        res[:3, 3] = v
    else:
        R = (
            np.eye(3)
            + np.sin(theta) / theta * S_
            + ((1 - np.cos(theta)) / (theta**2) * S_ @ S_)
        )
        U = (
            np.eye(3)
            + ((1 - np.cos(theta)) / (theta**2)) * S_
            + (((theta - np.sin(theta)) / (theta**3)) * S_ @ S_)
        )
        res[:3, :3] = R
        res[:3, 3] = U @ v

    return res


def get_points_from_curve(curve):
    points = []
    for H in curve:
        points.append(np.array(H[:3, -1]))
    return np.array(points).T


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


def compute_nextq(
    robot,
    q,
    curve,
    i,
    kt1=1.0,
    kt2=1.0,
    kt3=1.0,
    kn1=1.0,
    kn2=1.0,
    dt=0.01,
    hist_index=[],
    hist_dist=[],
    delta=1e-3,
    ds=1e-3,
):
    q = np.array(q.copy()).reshape(-1, 1)
    J, H = robot.jac_geo(q=q)
    xi, min_dist, min_index = robot.vector_field_se3(
        H,
        curve,
        kt1=kt1,
        kt2=kt2,
        kt3=kt3,
        kn1=kn1,
        kn2=kn2,
        # curve_derivative=curve_derivative,
        delta=delta,
        ds=ds,
        mode="c++",
    )
    hist_index.append(min_index)
    hist_dist.append(min_dist)
    p = np.array(H[:3, -1]).reshape(-1, 1)
    omega = np.array(xi[3:]).reshape(-1, 1)
    v = np.array(xi[:3]).reshape(-1, 1)
    pdot = (np.cross(omega.ravel(), p.ravel()) + v.ravel()).reshape(-1, 1)
    v = pdot
    twist_ = np.vstack((v, omega))
    qdot = Utils.dp_inv(J, 1e-4) @ twist_
    qdot = np.array(qdot).reshape(-1, 1)
    q = q + qdot * dt
    qdot_deg = qdot * 180 / np.pi

    return qdot_deg, hist_index, hist_dist


## Attach to shared memory for current_config
try:
    shm_current = shared_memory.SharedMemory(
        name="current_config", create=True, size=32
    )  # 8 elements * 4 bytes (float32)
except FileExistsError:
    shm_current = shared_memory.SharedMemory(name="current_config")

# Attach to shared memory for config_velocity
try:
    shm_velocity = shared_memory.SharedMemory(
        name="config_velocity", create=True, size=32
    )  # 8 elements * 4 bytes (float32)
except FileExistsError:
    shm_velocity = shared_memory.SharedMemory(name="config_velocity")

# Create numpy arrays backed by shared memory
current_config = np.ndarray(
    (8,), dtype=np.float32, buffer=shm_current.buf
)  # 1 flag + 7 data elements
config_velocity = np.ndarray(
    (8,), dtype=np.float32, buffer=shm_velocity.buf
)  # 1 flag + 7 data elements

# Initialize shared memory arrays
current_config[:] = 0  # Set all elements (flag and data) to 0
config_velocity[:] = 0  # Set all elements (flag and data) to 0

hist_index = []
hist_dist = []
config_hist = []
time_hist = []
try:
    print("Start control.py")
    print("Creating UAIBot Kinova")
    kinova = Robot.create_kinova_gen3(name="kinova")
    
    # n_points = 1000
    # radius = 0.15
    # dx = 0.0
    # dy = 0.4
    # height = 0.4

    # u = np.array(list(map(int, [(i % 2) == 0 for i in range(7)])))
    # v = np.logical_not(u).astype(int)
    # u = u / np.linalg.norm(u)
    # v = v / np.linalg.norm(v)
    # center = np.array([0] + [np.pi] * 6)
    # radius = np.array([np.pi, np.deg2rad(50)])  # mid=False
    # radius = np.array([np.deg2rad(180), np.deg2rad(50)])  # mid=True, dv=5.0
    # print("Creating curve in q")
    # curve_q = circle_rn(
    #     n_points=n_points, u=u, v=v, radius=radius, center=center, mid=True, dv=5.0
    # )
    # print("Creating curve in HTM")
    # curve = [np.array(kinova.fkm(q=q)) for q in curve_q]

    print("Loading curve from .npy")
    curve_ = np.load("/home/fbartelt/Documents/Projetos/SE3vectorfield-kinova/resampled_curve.npy")
    curve = [H for H in curve_]
    point_mat = get_points_from_curve(curve)
    target = PointCloud(name="target", points=point_mat, size=0.01, color="cyan")

    # kinova.set_ani_frame(q=config_mapping([0, 0, 0, 5, 0, 10, 0], "from_kinova"))
    kinova.set_ani_frame(q=config_mapping([0, 10, 0, 15, 0, 40, 30], "from_kinova"))
    # kinova.set_ani_frame(q=config_mapping([0, 10, 0, 15, 0, 40., 180.72], "from_kinova"))
    sim = Simulation.create_sim_grid([kinova, target])
    print("!! Ready !!")
    # kt1, kt2, kt3 = 0.03, 0.5, 1.0
    # kn1, kn2 = 0.02, 20.0
    kt1, kt2, kt3 = 0.03, 1.0, 1.0
    # 0.03
    kn1, kn2 = 0.08, 1.0
    # 0.08
    dt = 0.01
    i = 1
    set_time = True
    t0 = time.time()

    while True:
        # Read current_config from shared memory
        # if current_config[0] == 1:
        if True:
            config = current_config.copy()
            config[1:] = config_mapping(config[1:], maptype="from_kinova")
            print(f"Received current_config: {config}")
            # if current_config[0] == 1 and set_time:
            #     t0 = time.time()
            #     time_ = 0
            #     set_time = False
            #     hist_index = []
            #     hist_dist = []
            #     config_hist = []
            #     time_hist = []
            config_hist.append(config[1:])
            time_ = time.time() - t0
            time_hist.append(time_)

            velocity = np.array((7,), dtype=np.float32)

            # velocity = np.array(
            #     [
            #         4.5776367e-04,
            #         1.5007750e01,
            #         1.8000201e02,
            #         2.2999522e02,
            #         3.5999899e02,
            #         5.4996429e01,
            #         config[-1] + np.deg2rad(5),
            #     ]
            # )

            qd, hist_index, hist_dist = compute_nextq(
                kinova,
                config[1:],
                curve,
                i=0,
                kt1=kt1,
                kt2=kt2,
                kt3=kt3,
                kn1=kn1,
                kn2=kn2,
                dt=dt,
                hist_index=hist_index,
                hist_dist=hist_dist,
                ds=1e-3
            )

            kinova.add_ani_frame(time=time_, q=config[1:])
            target.add_ani_frame(time=time_, initial_ind=0, final_ind=len(curve) - 1)
            # velocity = config_mapping(qd.ravel(), maptype="to_kinova").ravel()
            velocity = qd.ravel()

            print(f"Sent config_velocity: {velocity}")
            config_velocity[1:] = velocity
            config_velocity[0] = 1
            # time.sleep(0.1)
            current_config[0] = 0
            i += 1
        else:
            pass
            # print("Waiting for current_config")
            # print(current_config)
        # time.sleep(0.1)

except KeyboardInterrupt:
    print("Shutting down control.py")
    sim.save("./", "real_exp_test2")
    with open('data.pkl', 'wb') as f:
        data = {'config_hist': config_hist, 'hist_index': hist_index, 'hist_dist': hist_dist, 'hist_time': time_hist}
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("saved pickled data")
    # shm_current.close()
    # shm_velocity.close()

finally:
    # Clean up shared memory
    shm_current.close()
    shm_velocity.close()
