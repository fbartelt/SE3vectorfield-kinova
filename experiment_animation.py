# %%
import os
import pickle
import numpy as np
from uaibot import Robot, Simulation
from uaibot.simobjects.curve import CurveSE3


def config_mapping(q, maptype="from_kinova"):
    """UaiBot does not have initial theta in DH + Kinova returns config
    in degrees
    """
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


exp_case = "cylindrical"
if "cyl" in exp_case:
    curve_file_name = "cylindrical.npy"
elif "circ" in exp_case:
    curve_file_name = "circle.npy"
else:
    raise FileNotFoundError("This file does not belong in this experiment")
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

kinova = Robot.create_kinova_gen3(
    name="kinova",
)

curve = np.load(curve_path)


target = CurveSE3(
    points=curve,
    size=0.02,
    color="cyan",
    frame_size=0.1,
    num_frames=5,
)

kinova.set_ani_frame(q=config_mapping([0, 10, 0, 15, 0, 40, 30], "from_kinova"))

sim = Simulation.create_sim_grid([kinova, target])

sim.set_parameters(background_color="white", ambient_light_intensity=2)

# final_index = np.nonzero(np.array(hist_time) > 160)[0][0]
final_index = len(config_hist)

for i, q_ in enumerate(config_hist[:final_index]):
    q = np.asarray(q_)
    time = float(hist_time[i])
    kinova.add_ani_frame(time=time, q=q)

sim.set_parameters(width=1480, height=960)

sim.run_in_browser()
