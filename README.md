# AutoDrone

“AutoDrone” is a simulated autonomous drone agent trained through reinforcement learning (RL) to perform point-to-point aerial navigation. A PPO learning algorithm is used to evolve the agent from a basic quadcopter platform into a competent navigator. The project offers a hands-on insight into AI agent training and forming a base for more complex systems. By leveraging RL, the approach  offers an alternative to traditional control system by allowing agents to learn through interaction rather than manual tuning. 

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

# Poster

![AutoDrone Poster](assets/poster.png)