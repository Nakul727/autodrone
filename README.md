# AutoDrone

https://github.com/user-attachments/assets/8908715b-f8a1-42b0-93a0-7b08307ad68c

## Project Installation

```zsh
git clone git@github.com:Nakul727/autodrone.git
cd autodrone
git submodule update --init --recursive
```

<br>

We need Python<=3.11 to install pybullet. To use this specific version of python and install
other fixed version for modules, we will use `conda` to manage our environment. Please have conda installed.
The gym-pybullet-drones will be installed as a submodule.

<br>

```zsh
conda env create -f environment.yml
conda activate autodrone
pip install --no-deps -e gym-pybullet-drones/
```
