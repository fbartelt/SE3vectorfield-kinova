# Adapted from
# https://github.com/Kinovarobotics/Kinova-kortex2_Gen3_G3L/blob/master/api_python/examples/108-Gen3_torque_control/01-torque_control_cyclic.py

# Control Modes (https://github.com/Kinovarobotics/Kinova-kortex2_Gen3_G3L/blob/master/api_python/doc/markdown/enums/ActuatorConfig/ControlMode.md)
# 0 - None, 1 - position, 2 - velocity, 3 - torque, 4 - current, 5 - custom, 6 - torque high velocity
import sys
import os
import numpy as np
import time
import argparse
import threading
import pickle
from uaibot import Robot, Utils

from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.ActuatorConfigClientRpc import ActuatorConfigClient
from kortex_api.autogen.client_stubs.ActuatorCyclicClientRpc import ActuatorCyclicClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.client_stubs.DeviceConfigClientRpc import DeviceConfigClient
from kortex_api.autogen.client_stubs.DeviceManagerClientRpc import DeviceManagerClient
from kortex_api.autogen.messages import (
    ActuatorConfig_pb2,
    Base_pb2,
    BaseCyclic_pb2,
    Common_pb2,
)
from kortex_api.RouterClient import RouterClientSendOptions
import utilities

"""
THIS SCRIPT USES THE ADVANCED FUNCTIONS FOR 1KHZ CONTROL
"""
# =========================================================================
# qdot limits (from manual) in deg/s
QDOT_UPPER_BOUND = np.array([79.64, 79.64, 79.64, 79.64, 69.91, 69.91, 69.91]).reshape(
    -1, 1
)
QDOT_LOWER_BOUND = -QDOT_UPPER_BOUND
# q limits (from manual) in rad
Q_UPPER_BOUND = np.array(
    [
        np.inf,
        np.deg2rad(128.9),
        np.inf,
        np.deg2rad(147.8),
        np.inf,
        np.deg2rad(120.3),
        np.inf,
    ]
).reshape(-1, 1)
Q_LOWER_BOUND = -Q_UPPER_BOUND
# Defaults
DEFAULT_CYCLIC_SAMPLING_TIME = 0.001
DEFAULT_MAX_EXPERIMENT_TIME = 60 * 3  # SECONDS
DEFAULT_PRINT_STATS = 1
DEFAULT_ACTION_TIMEOUT_DURATION = 60  # seconds (time to assume communication failed)
# VECTOR FIELD GAINS
# kn = kn1 * tanh(kn2 * sqrt(D))
# kt = kt1 * (1 - kt2 * tanh(kn3 * sqrt(D))
kt1, kt2, kt3 = 0.07, 1.0, 0.75
kn1, kn2 = 0.1, kt3
ds, delta = 1e-3, 1e-3
# PATHS
file_parent_path = os.path.dirname(__file__)
curve_path = f"{file_parent_path}/resampled_curve2.npy"
save_path = f"{file_parent_path}/kinova_experiment.pkl"
print_interval = 1  # second


def progress_bar(i, imax, bar_length=20, return_bar=False):
    percent = i / imax
    filled_length = int(np.ceil(bar_length * percent))
    bar = "█" * filled_length + "-" * (bar_length - filled_length)
    bar = f"Progress: |{bar}| {percent:.1%}"
    if return_bar:
        return bar
    else:
        print(f"\r{bar}", end="\r")
        if i == imax:
            print()  # Move to the next line on completion


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


class kinovaExperiment:
    def __init__(
        self,
        router,
        router_real_time,
        curve,
        curve_derivative=[],
        kn1=1.0,
        kn2=1.0,
        kt1=1.0,
        kt2=1.0,
        kt3=1.0,
        ds=1e-3,
        delta=1e-3,
        print_interval=1,
        qdot_lb=QDOT_LOWER_BOUND,
        qdot_ub=QDOT_UPPER_BOUND,
        q_lb=Q_LOWER_BOUND,
        q_ub=Q_UPPER_BOUND,
    ):

        # Maximum allowed waiting time during actions (in seconds)
        self.ACTION_TIMEOUT_DURATION = DEFAULT_ACTION_TIMEOUT_DURATION

        # Create required services
        device_manager = DeviceManagerClient(router)

        self.actuator_config = ActuatorConfigClient(router)
        self.base = BaseClient(router)
        self.base_cyclic = BaseCyclicClient(router_real_time)

        self.base_command = BaseCyclic_pb2.Command()
        self.base_feedback = BaseCyclic_pb2.Feedback()
        self.base_custom_data = BaseCyclic_pb2.CustomData()

        # Detect all devices
        device_handles = device_manager.ReadAllDevices()
        self.actuator_count = self.base.GetActuatorCount().count

        # Only actuators are relevant for this example
        for handle in device_handles.device_handle:
            if (
                handle.device_type == Common_pb2.BIG_ACTUATOR
                or handle.device_type == Common_pb2.SMALL_ACTUATOR
            ):
                self.base_command.actuators.add()
                self.base_feedback.actuators.add()

        # Change send option to reduce max timeout at 3ms
        self.sendOption = RouterClientSendOptions()
        # self.sendOption.andForget = False
        # self.sendOption.delay_ms = 0
        # self.sendOption.timeout_ms = 3

        self.cyclic_t_end = (
            30  # Total duration of the thread in seconds. 0 means infinite.
        )
        self.cyclic_thread = {}

        self.kill_the_thread = False
        self.already_stopped = False
        self.cyclic_running = False

        # ====================================================================
        # =========================== EDIT HERE ==============================
        self.robot = Robot.create_kinova_gen3(name="kinova")
        self.kn1 = kn1
        self.kn2 = kn2
        self.kt1 = kt1
        self.kt2 = kt2
        self.kt3 = kt3
        self.curve = curve
        self.curve_derivative = curve_derivative
        self.ds = ds
        self.delta = delta
        self.print_interval = print_interval  # in seconds
        self.hist_q = []
        self.hist_qdot = []
        self.hist_time = []  # accumulates t_now - t_initi
        self.hist_dist = []  # accumulates distance to closest point
        self.hist_closest_index = []  # accumulates i* (closest point index)
        self.qdot_lb = qdot_lb
        self.qdot_ub = qdot_ub
        # ====================================================================

    # Create closure to set an event after an END or an ABORT
    def check_for_end_or_abort(self, e):
        """Return a closure checking for END or ABORT notifications

        Arguments:
        e -- event to signal when the action is completed
            (will be set when an END or ABORT occurs)
        """

        def check(notification, e=e):
            print("EVENT : " + Base_pb2.ActionEvent.Name(notification.action_event))
            if (
                notification.action_event == Base_pb2.ACTION_END
                or notification.action_event == Base_pb2.ACTION_ABORT
            ):
                e.set()

        return check

    def MoveToHomePosition(self):
        # Make sure the arm is in Single Level Servoing mode
        base_servo_mode = Base_pb2.ServoingModeInformation()
        base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
        self.base.SetServoingMode(base_servo_mode)

        # Move arm to ready position
        print("Moving the arm to a safe position")
        action_type = Base_pb2.RequestedActionType()
        action_type.action_type = Base_pb2.REACH_JOINT_ANGLES
        action_list = self.base.ReadAllActions(action_type)
        action_handle = None
        for action in action_list.action_list:
            if action.name == "Home":
                action_handle = action.handle

        if action_handle == None:
            print("Can't reach safe position. Exiting")
            return False

        e = threading.Event()
        notification_handle = self.base.OnNotificationActionTopic(
            self.check_for_end_or_abort(e), Base_pb2.NotificationOptions()
        )

        self.base.ExecuteActionFromReference(action_handle)

        print("Waiting for movement to finish ...")
        finished = e.wait(self.ACTION_TIMEOUT_DURATION)
        self.base.Unsubscribe(notification_handle)

        if finished:
            print("Cartesian movement completed")
        else:
            print("Timeout on action notification wait")
        return finished

        return True

    def InitCyclic(self, sampling_time_cyclic, t_end, print_stats):

        if self.cyclic_running:
            return True

        # Move to Home position first
        if not self.MoveToHomePosition():
            return False

        print("Init Cyclic")
        sys.stdout.flush()

        base_feedback = self.SendCallWithRetry(self.base_cyclic.RefreshFeedback, 3)
        if base_feedback:
            self.base_feedback = base_feedback

            # Init command frame
            for x in range(self.actuator_count):
                self.base_command.actuators[x].flags = 1  # servoing
                self.base_command.actuators[x].position = self.base_feedback.actuators[
                    x
                ].position
                # Add velocities too
                self.base_command.actuators[x].velocity = self.base_feedback.actuators[
                    x
                ].velocity

            # Set arm in LOW_LEVEL_SERVOING
            base_servo_mode = Base_pb2.ServoingModeInformation()
            base_servo_mode.servoing_mode = Base_pb2.LOW_LEVEL_SERVOING
            self.base.SetServoingMode(base_servo_mode)

            # Send first frame
            self.base_feedback = self.base_cyclic.Refresh(
                self.base_command, 0, self.sendOption
            )

            # Set actuators in velocity mode now that the command is equal to measure
            control_mode_message = ActuatorConfig_pb2.ControlModeInformation()
            control_mode_message.control_mode = ActuatorConfig_pb2.ControlMode.Value(
                "VELOCITY"
            )

            # Set every joint to receive velocity commands
            # first actuator as id = 1
            for device_id in range(self.actuator_count):
                self.SendCallWithRetry(
                    self.actuator_config.SetControlMode,
                    3,
                    control_mode_message,
                    device_id,
                )

            # This decouples control loop from main() checks
            # Init cyclic thread
            self.cyclic_t_end = t_end
            self.cyclic_thread = threading.Thread(
                target=self.RunCyclic, args=(sampling_time_cyclic, print_stats)
            )
            self.cyclic_thread.daemon = True
            self.cyclic_thread.start()
            return True

        else:
            print("InitCyclic: failed to communicate")
            return False

    def RunCyclic(self, t_sample, print_stats):
        """t_sample is 1ms (1kHz) in this case."""
        self.cyclic_running = True
        print("Run Cyclic")
        sys.stdout.flush()
        cyclic_count = 0  # Counts refresh
        stats_count = 0  # Counts stats prints
        failed_cyclic_count = 0  # Count communication timeouts

        # Initial first and last actuator torques; avoids unexpected movement due to torque offsets
        q, qdot = np.zeros((self.actuator_count, 1))
        for i in range(self.actuator_count):
            q[i] = self.base_feedback.actuators[i].position
            qdot[i] = self.base_feedback.actuators[i].velocity

        # =============================================================
        # =================== EDIT HERE IF NECESSARY ==================
        q_rad = config_mapping(q, "from_kinova")
        self.robot.set_ani_frame()
        self.hist_q.append(q_rad)
        self.hist_qdot.append(qdot)
        self.hist_time.append(0.0)
        self.hist_dist.append(0.0)
        self.hist_closest_index.append(0)
        # =============================================================

        t_now = time.time()
        t_cyclic = t_now  # cyclic time
        t_stats = t_now  # print  time
        t_init = t_now  # init   time

        print("Running torque control example for {} seconds".format(self.cyclic_t_end))

        while not self.kill_the_thread:
            t_now = time.time()

            # Cyclic Refresh (only updates after 1ms)
            if (t_now - t_cyclic) >= t_sample:
                t_cyclic = t_now

                # =============================================================
                # =================== EDIT HERE IF NECESSARY ==================
                for i in range(self.actuator_count):
                    q[i] = self.base_feedback.actuators[i].position
                    qdot[i] = self.base_feedback.actuators[i].velocity

                q_rad = config_mapping(q, "from_kinova")
                curr_time = t_now - t_init
                qdot_d, vfdata, log_msg = self.control_step(
                    q_rad, current_time=curr_time, final_time=self.cyclic_t_end
                )
                if any(q_rad >= self.q_ub or q_rad <= self.q_lb):
                    print(f"JOINT LIMIT VIOLATION: {q.ravel()} (deg)")
                    self.kill_the_thread = True
                    break
                # vfdata is (distance, closest_index)
                # qdot is in deg/s
                # =============================================================

                for i in range(self.actuator_count):
                    self.base_command.actuators[i].velocity = qdot_d[i].item()
                    # Apparentely this prevents some errors (REMOVE IF BUG)
                    self.base_command.actuators[i].position = (
                        self.base_feedback.actuators[i].position
                    )

                # Incrementing identifier ensure actuators can reject out of time frames
                self.base_command.frame_id += 1
                if self.base_command.frame_id > 65535:
                    self.base_command.frame_id = 0
                for i in range(self.actuator_count):
                    self.base_command.actuators[i].command_id = (
                        self.base_command.frame_id
                    )

                # Frame is sent
                try:
                    self.base_feedback = self.base_cyclic.Refresh(
                        self.base_command, 0, self.sendOption
                    )
                except:
                    failed_cyclic_count = failed_cyclic_count + 1
                cyclic_count = cyclic_count + 1

                # =============================================================
                # =================== EDIT HERE IF NECESSARY ==================
                self.hist_q.append(q_rad)
                self.hist_qdot.append(qdot)
                self.hist_time.append(curr_time)
                self.hist_dist.append(vfdata[0])
                self.hist_closest_index.append(vfdata[1])
                # =============================================================

            # Stats Print
            if print_stats and ((t_now - t_stats) > self.print_interval):
                t_stats = t_now
                stats_count = stats_count + 1

                cyclic_count = 0
                failed_cyclic_count = 0
                print(log_msg)
                sys.stdout.flush()

            if self.cyclic_t_end != 0 and (t_now - t_init > self.cyclic_t_end):
                print("Cyclic Finished")
                sys.stdout.flush()
                break
        self.cyclic_running = False
        return True

    def StopCyclic(self):
        print("Stopping the cyclic and putting the arm back in position mode...")
        if self.already_stopped:
            return

        # Kill the  thread first
        if self.cyclic_running:
            self.kill_the_thread = True
            self.cyclic_thread.join()

        # Set first actuator back in position mode
        control_mode_message = ActuatorConfig_pb2.ControlModeInformation()
        control_mode_message.control_mode = ActuatorConfig_pb2.ControlMode.Value(
            "POSITION"
        )
        # device_id = 1  # first actuator has id = 1
        for device_id in range(self.actuator_count):
            self.SendCallWithRetry(
                self.actuator_config.SetControlMode, 3, control_mode_message, device_id
            )

        base_servo_mode = Base_pb2.ServoingModeInformation()
        base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
        self.base.SetServoingMode(base_servo_mode)
        self.cyclic_t_end = 0.1

        self.already_stopped = True

        print("Clean Exit")

    # =========================================================================
    # ================================ EDIT HERE ==============================
    def control_step(self, q, current_time, final_time):
        J, H = self.robot.jac_geo(q=q)
        t0 = time.perf_counter()

        xi, min_dist, closest_index = self.robot.vector_field_se3(
            H,
            self.curve,
            kt1=self.kt1,
            kt2=self.kt2,
            kt3=self.kt3,
            kn1=self.kn1,
            kn2=self.kn2,
            # curve_derivative=curve_derivative,
            delta=self.delta,
            ds=self.ds,
            mode="c++",
        )

        t1 = time.perf_counter()
        solver_time_ms = (t1 - t0) * 1000.0

        # xi = np.array(xi).ravel()
        p = np.array(H[:3, -1]).reshape(-1, 1)
        omega = np.array(xi[3:]).reshape(-1, 1)
        v = np.array(xi[:3]).reshape(-1, 1)
        pdot = (np.cross(omega.ravel(), p.ravel()) + v.ravel()).reshape(-1, 1)
        twist_ = np.vstack((pdot, omega))

        qdot = Utils.dp_inv(J, 1e-4) @ twist_
        qdot = np.array(qdot).reshape(-1, 1)
        # Kinova commands are in deg / s
        qdot_deg = qdot * 180 / np.pi
        qdot_deg = np.clip(qdot_deg, self.qdot_lb, self.qdot_ub)

        log_msg = f"\rDist = {min_dist} (VF took {solver_time_ms}ms)"
        bar = progress_bar(i=current_time, imax=final_time, return_bar=True)
        log_msg = "[" + log_msg + "]" + bar

        return qdot_deg, (min_dist, closest_index), log_msg

    # =========================================================================

    @staticmethod
    def SendCallWithRetry(call, retry, *args):
        i = 0
        arg_out = []
        while i < retry:
            try:
                arg_out = call(*args)
                break
            except:
                i = i + 1
                continue
        if i == retry:
            print("Failed to communicate")
        return arg_out


def save_data(path, kinova_exp):
    save_path = os.path.splitext(path)[0] + ".pkl"
    with open(save_path, "wb") as f:
        data = {
            "hist_q": kinova_exp.hist_q,
            "hist_qdot": kinova_exp.hist_qdot,
            "hist_index": kinova_exp.hist_closest_index,
            "hist_dist": kinova_exp.hist_dist,
            "hist_time": kinova_exp.hist_time,
        }
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("Saved pickled data.")


def main():
    # Import the utilities helper module
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cyclic_time",
        type=float,
        help="delay, in seconds, between cylic control call",
        default=DEFAULT_CYCLIC_SAMPLING_TIME,
    )
    parser.add_argument(
        "--duration",
        type=int,
        help="experiment duration, in seconds (0 means infinite)",
        default=DEFAULT_MAX_EXPERIMENT_TIME,
    )
    parser.add_argument(
        "--print_stats",
        default=DEFAULT_PRINT_STATS,
        help="print stats in command line or not (0 to disable)",
        type=lambda x: (str(x).lower() not in ["false", "0", "no"]),
    )
    args = utilities.parseConnectionArguments(parser)

    # Create connection to the device and get the router
    with utilities.DeviceConnection.createTcpConnection(args) as router:

        with utilities.DeviceConnection.createUdpConnection(args) as router_real_time:

            print(f"Loading curve from {curve_path}")
            curve_raw = np.load(curve_path, allow_pickle=True)
            curve = [H for H in curve_raw]

            kinova_exp = kinovaExperiment(
                router,
                router_real_time,
                curve=curve,
                kn1=kn1,
                kn2=kn2,
                kt1=kt1,
                kt2=kt2,
                kt3=kt3,
                ds=ds,
                delta=delta,
                print_interval=print_interval,
            )

            success = kinova_exp.InitCyclic(
                args.cyclic_time, args.duration, args.print_stats
            )

            if success:
                try:
                    while kinova_exp.cyclic_running:
                        time.sleep(0.5)
                except KeyboardInterrupt:
                    print("\nKeyboard interrupt received.")
                except Exception as e:
                    print(f"\nUnexpected error: {e}")
                finally:
                    # Stop Cyclic thread
                    kinova_exp.StopCyclic()
                    # Save data after experiment is killed
                    save_data(save_path, kinova_exp)

            return 0 if success else 1


if __name__ == "__main__":
    exit(main())
