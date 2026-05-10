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

The kinematic control uses the Advanced Interface of BaseCyclic which has a sampling rate of 1kHz. The old script (`control_40hz.py`) runs at 40Hz only, but is safer.

You should first create the curve `.npy` file using the `precompute_curve.py` script

The robot will go to the configuration q=\[0, 10, 0, 15, 0, 40, 30\] and wait for 5 seconds.
```bash
python ./control.py
```

## Results

Expected movement can be checked agains the simulation in `expected_movement.py` script.

Data can be analyzed using the `check_experiment_results.py`, although it will consider only data used in the previous work.

You can animate the experiment data using the `experiment_animation.py` script, although you it will probably need heavy modifications.

> [Kinova Tutorials Playlist](https://youtube.com/playlist?list=PLz1XwEYRuku5rZjJWBr6SDi93jgWZ4FHL&si=zSrxxHjoIQz1Fg0k)
