import numpy as np
from multiprocessing import shared_memory
import numpy as np
import time
import threading
import pickle

from kortex_api.TCPTransport import TCPTransport
from kortex_api.RouterClient import RouterClient
from kortex_api.SessionManager import SessionManager
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.messages import Session_pb2, Base_pb2
from kortex_api.Exceptions.KServerException import KServerException

# Maximum allowed waiting time during actions (in seconds)
TIMEOUT_DURATION = 60 * 5


# Create closure to set an event after an END or an ABORT
def check_for_end_or_abort(e):
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


def send_home(base_client_service):
    print("Going Home....")
    action_type = Base_pb2.RequestedActionType()
    action_type.action_type = Base_pb2.REACH_JOINT_ANGLES
    action_list = base_client_service.ReadAllActions(action_type)
    action_handle = None

    for action in action_list.action_list:
        if action.name == "Home":
            action_handle = action.handle

    base_client_service.ExecuteActionFromReference(action_handle)
    time.sleep(6)
    print("Done!")


def get_current_config(base):
    q = np.zeros((7,), dtype=np.float32)
    try:
        # print("Getting Angles for every joint...")
        input_joint_angles = base.GetMeasuredJointAngles()
    except KServerException as ex:
        print("Unable to get joint angles")
        print(
            "Error_code:{} , Sub_error_code:{} ".format(
                ex.get_error_code(), ex.get_error_sub_code()
            )
        )
        print("Caught expected error: {}".format(ex))
        return np.array([-11] * 7, dtype=np.float32)

    print("Joint ID : Joint Angle")

    for i, joint_angle in enumerate(input_joint_angles.joint_angles):
        q[i] = joint_angle.value
        print(joint_angle.joint_identifier, " : ", joint_angle.value)

    return q


def change_configuration(base, q_next):
    action = Base_pb2.Action()
    action.name = "Example angular action movement"
    action.application_data = ""
    actuator_count = base.GetActuatorCount()
    q_next = np.array(q_next).ravel()

    for joint_id in range(actuator_count.count):
        joint_angle = action.reach_joint_angles.joint_angles.joint_angles.add()
        joint_angle.joint_identifier = joint_id
        joint_angle.value = q_next[joint_id]

    e = threading.Event()
    notification_handle = base.OnNotificationActionTopic(
        check_for_end_or_abort(e), Base_pb2.NotificationOptions()
    )

    print("Executing action")
    base.ExecuteAction(action)

    print("Waiting for movement to finish ...")
    finished = e.wait(TIMEOUT_DURATION)
    base.Unsubscribe(notification_handle)

    if finished:
        print("Angular movement completed")
    else:
        print("Timeout on action notification wait")
    return finished

def send_velocity(base, velocity):

    joint_speeds = Base_pb2.JointSpeeds()

    actuator_count = base.GetActuatorCount().count
    # The 7DOF robot will spin in the same direction for 10 seconds
    for i in range(actuator_count):
        joint_speed = joint_speeds.joint_speeds.add()
        joint_speed.joint_identifier = i 
        joint_speed.value = velocity[i]
        joint_speed.duration = 0
        print ("Sending the joint speeds for 10 seconds...")
        base.SendJointSpeedsCommand(joint_speeds)
        # time.sleep(10)

    return True


# Attach to shared memory for current_config
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

config_hist = []
time_hist = []

if __name__ == "__main__":
    DEVICE_IP = "192.168.1.10"
    DEVICE_PORT = 10000

    # Setup API
    error_callback = lambda kException: print(
        "_________ callback error _________ {}".format(kException)
    )
    transport = TCPTransport()
    router = RouterClient(transport, error_callback)
    transport.connect(DEVICE_IP, DEVICE_PORT)

    # Create session
    print("Creating session for communication")
    session_info = Session_pb2.CreateSessionInfo()
    session_info.username = "admin"
    session_info.password = "admin"
    session_info.session_inactivity_timeout = 60000 * 5  # (milliseconds)
    session_info.connection_inactivity_timeout = 60000 * 5  # (milliseconds)
    print("Session created")

    session_manager = SessionManager(router)
    session_manager.CreateSession(session_info)

    # Create required services
    base_client_service = BaseClient(router)

    send_home(base_client_service)
    print("Going to almost upwards")
    # change_configuration(base_client_service, np.array([0, 0, 0, 5, 0, 10, 0]))
    change_configuration(base_client_service, np.array([0, 10, 0, 15, 0, 40, 30]))
    # change_configuration(base_client_service, np.array([0, 10, 0, 15, 0, 40., 180.72]))
    # change_configuration(base_client_service, np.array([0, 10, 0, 15, 0, 40., 180.72]))
    time.sleep(5)
    try:
        t0 = time.time()
        while True:
            aux_config = get_current_config(base_client_service)
            config_hist.append(aux_config)
            time_hist.append(time.time() - t0)
            # Get current config:
            # if current_config[0] == 0:
            if True:
                new_config = get_current_config(base_client_service)
                # current_config[:] = np.array([1, 2, 3, 4, 5], dtype=np.float32)
                current_config[1:] = new_config
                current_config[0] = 1
                print(f"Sent current_config: {current_config}")

            # if config_velocity[0] == 1:
            if True:
                # Read config_velocity from shared memory
                velocity = config_velocity.copy()
                print(f"Received config_velocity: {velocity}")
                # success = change_configuration(base_client_service, velocity[1:])
                success = send_velocity(base_client_service, velocity[1:])
                if not success:
                    print("Failed to change configuration")
                    break
                config_velocity[0] = 0
            else:
                pass
                # print(f"Waiting velocity, current: {config_velocity}")

            # time.sleep(0.1)

    except KeyboardInterrupt:
        print("Shutting down experiment.py")
        send_home(base_client_service)

        np.save('config_hist_exp.npy', np.array(config_hist))
        print("Saved config hist")
        np.save('time_hist_exp.npy', np.array(time_hist))
        print("Saved time hist")

        print("Closing Session..")
        session_manager.CloseSession()
        router.SetActivationStatus(False)
        transport.disconnect()
        print("Done!")

        shm_current.close()
        shm_velocity.close()

    finally:
        # Clean up shared memory

        send_home(base_client_service)
        # with open('./config_data.pkl', 'wb') as f:
        #     data = {'config_hist': config_hist, 'time_hist': time_hist}
        #     pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print ("Stopping the robot")
        base_client_service.Stop()
        shm_current.close()
        shm_velocity.close()

        print("Closing Session..")
        session_manager.CloseSession()
        router.SetActivationStatus(False)
        transport.disconnect()
        print("Done!")
