# CMPT 310 Group Project

## Project Installation

### Install Repository

```
git clone git@github.com:Nakul727/autodrone.git
cd autodrone
git submodule update --init --recursive
```

### Install Conda Virtual Environment

```zsh
conda create -n autodrone python=3.11 -y
conda activate autodrone
```

### Install Requirements

```zsh
pip install -e gym-pybullet-drones
pip install -r requirements.txt
```
