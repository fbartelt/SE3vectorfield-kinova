# Kinova GEN3 experiment using Vector Field strategy in SE(3)

Steps for reproducing the experiment.

## Kinova API

Kortex API v2.6.0 was used

```bash
# Create conda environment for Python 3.9
conda create -n kinova python=3.9
conda activate kinova

# Install Kinova API
git clone https://github.com/verlab/demos-verlab.github
cd demos-verlab
git checkout c079f4e
cd kinova-demos
pip install -e .
```

## Uaibot

Unpublished private uaibot version was used. Requires Python 3.11.

```bash
git clone git@github.com:fbartelt/uaibot_experimental.git
cd uaibot_experimental
git checkout 5750bd6

python3.11 -m venv py311
source py311/bin/activate
pip install .
```

## Setup

Kinova robot uses TCP for communication. Configure the network:

```bash
chmod +x kinova_network_setup.sh 
./kinova_network_setup.sh
```

To revert network changes:

```bash
chmod +x reset_network.sh
./reset_network.sh
```

## Run

The experiment relies on two Python environments. Both scripts share data using numpy array shared memory

### Precompute curve in SE(3)

Generate and resample the curve for uniform distribution in SE(3). The result is saved in `resampled_curve.npy`

```bash
source py311/bin/activate
py311/bin/python ./precompute_curve.py
```

### Control

Run the control and experiment scripts (order doesn't matter):

```bash
source py311/bin/activate
py311/bin/python ./control.py
```

```bash
conda activate kinova
python ./experiment.py
```

## Visualization

`control.py` stores experiment data in `data.pkl`. Access it with:

```python
import pickle

with open("./data.pkl", "rb") as f:
    data = pickle.load(f)

config_hist = data["config_hist"]
hist_index = data["hist_index"]
hist_dist = data["hist_dist"]
```

Visualize results using `check_experimental_results.py` in the `py311` environment.

## Animation

`experiment.py` stores configurations and timestamps in `config_data.pkl`. Access it with:

```python
import pickle

with open('config_data.pkl', 'rb') as f:
    data = pickle.load(f)

config_hist = data['config_hist']
time_hist = data['time_hist']
```

Animations can be visualized by running the notebook cells in `experiment_animation.py` using *VS Code's IPython interactive cells* or by converting the file into a *Jupyter notebook*.
