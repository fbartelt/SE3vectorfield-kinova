# Kinova GEN3 experiment using Vector Field strategy in SE(3)

## API installation

Check [Kinova Kortex repository](https://github.com/Kinovarobotics/Kinova-kortex2_Gen3_G3L) for latest API Release. This will install the 2.8.0 version of the Python API.

> For non-linux users other files are available [here](https://artifactory.kinovaapps.com/ui/native/generic-local-public/kortex/API/) 

```bash
curl --output-dir "/tmp" -O https://artifactory.kinovaapps.com/artifactory/generic-local-public/kortex/API/2.8.0/kortex_api-2.8.0.post5-py3-none-any.whl

python -m pip install /tmp/kortex_api-2.8.0.post5-py3-none-any.whl
python -m pip install uaibot
```

> Note: Tested with Python 3.14 and `uaibot==1.2.7`.

## Setup

Initialize Kinova Gen 3 (hold button for ~3s) and connect to the robot via Ethernet cable. Run the following command to configure the robot IP address and port:

> Note: you might have to change the interface name. Check with `ip link show`

```bash
chmod +x ./kinova_network_setup.sh
./kinova_network_setup.sh
```

To reverse the changes, run:

```bash
chmod +x ./reset_network.sh
./reset_network.sh
```

## Run basic tests

This will set q0=0 and rotate each joint of the robot by 5 degrees/s and save configuration data. Finally, the robot returns to the default pose.

Log files will be `tests_config.npy` and `tests_time.npy` and should contain the configuration values and timestamps of the tests.
```bash
python ./kinova_basic_tests.py
```

If everything works, you should now perform the experiment by:

## Run the experiment

The kinematic control uses the Advanced Interface of BaseCyclic which has a sampling rate of 1kHz.

There are two possible curves you can track:

- **Cylindrical surface**: The curve oscillates along the world-frame z-axis of a cylinder. The orientation frame is such that the z-axis points inward and the y-axis points in the world z-axis direction. This case requires both the curve and its analytic derivative.
- **Circle in joint space**: A circle in the space of all joints without joint limits. This case requires only the curve.

First, create the required `.npy` files using the `precompute_curve.py` script. You must comment/uncomment the appropriate sections to select the desired curve:

- For the **cylindrical surface**, ensure the cylindrical section is active. This will create two files: `data/cylindrical.npy` (the curve) and `data/cylindrical_derivative.npy` (the curve derivative).
- For the **circle**, ensure the circle section is active and **comment out** `np.save(dcurve_path, dcurve)` since no derivative is needed. This will create only `data/circle.npy`.

Then, run the experiment:

```bash
python ./control.py
```

The robot will go to the configuration `q=[0, 10, 0, 15, 0, 40, 30]` and wait for 5 seconds.

For both the real experiment (`control.py`) and the simulation (`expected_movement.py`), the `RUN_CIRCLE` flag selects which curve to use. Set `RUN_CIRCLE = True` for the circle case, or `False` for the cylindrical case. The scripts handle the rest automatically, but you may adjust gains and other parameters in the respective code sections.

> **Note**: The controller now uses a QP with CBF for joint limits and collisions (self and with virtual cylinder, although this is not relevant for the circle case). The old version used a damped pseudo-inverse approach. Results might differ slightly.

## Results

Expected movement can be checked against the simulation in `expected_movement.py` script.

Data can be analyzed using the `check_experiment_results.py`, although it will consider only data used in the previous work.

You can animate the experiment data using the `experiment_animation.py` script, although you it will probably need heavy modifications.

> [Kinova Tutorials Playlist](https://youtube.com/playlist?list=PLz1XwEYRuku5rZjJWBr6SDi93jgWZ4FHL&si=zSrxxHjoIQz1Fg0k)
