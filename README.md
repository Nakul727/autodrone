# CMPT 310 Group Project

## Project Installation

### Install Repository

```
# Clone the repository
git clone git@github.com:Nakul727/autodrone.git
cd autodrone

# Initialize the submodules
git submodule update --init --recursive
```

### Install Conda Virtual Environment

```zsh
# Create the environment and install packages with compatible versions
conda create -n autodrone python=3.11 pybullet numpy=1.26 gymnasium=0.29 pytest=8 pandas matplotlib scipy -c conda-forge -y

# Activate the new environment
conda activate autodrone
```

### Install Requirements

```zsh
# 1. Install the local submodule whithout its dependencies
pip install --no-deps -e gym-pybullet-drones/

# 2. Install the remaining Python-only packages
pip install -r requirements.txt
```
