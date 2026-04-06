# UR5e Lift Task — Isaac Lab

A custom reinforcement learning environment for the **Universal Robots UR5e** robotic arm with a **Robotiq gripper**, built with [Isaac Lab](https://isaac-sim.github.io/IsaacLab). The agent learns to pick up a cube from a table surface using **Proximal Policy Optimization (PPO)** via the [skrl](https://skrl.readthedocs.io) library.

---

## Overview

The task is a manipulation problem where the robot must grasp a cube placed on a table and lift it to a target height. The environment is fully parallelized using Isaac Lab's Manager-Based RL framework.

**Key details:**

- Robot: Universal Robots UR5e + Robotiq gripper
- Task: Cube grasping and lifting
- RL algorithm: PPO (skrl)
- Environment: 4096 parallel envs (configurable)
- Action space: Joint position control (arm) + gripper binary open/close
- Observation space: Joint positions, joint velocities, cube pose, target pose, last action
- Episode length: 12 seconds

---

## Requirements

| Dependency | Version |
|---|---|
| Isaac Sim | 5.0.0 |
| Isaac Lab | 0.46 |
| skrl | latest |
| Python | 3.11 |
| CUDA | 11.8 |

---

## Installation

**1. Install Isaac Lab** by following the [official guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).

**2. Clone this repository** outside the IsaacLab directory:

```bash
git clone https://github.com/FarkasBalintKaroly/IsaacLab_UR5e_Lift_Task.git
cd IsaacLab_UR5e_Lift_Task
```

**3. Install the package** in editable mode:

```bash
# Replace PATH_TO with the actual path to your IsaacLab installation
PATH_TO/isaaclab.sh -p -m pip install -e source/UR5_lift_cube
```

**4. Place the UR5e USD asset** in the repository root:

```
IsaacLab_UR5e_Lift_Task/
└── ur5e_gripper.usdz      ← required
```

---

## Training

```bash
../../IsaacLab/isaaclab.sh -p scripts/skrl/train.py --task=IsaacOwn-Lift-Cube-UR5E-v1 --headless
```

To reduce the number of parallel environments (e.g. for less VRAM):

```bash
../../IsaacLab/isaaclab.sh -p scripts/skrl/train.py --task=IsaacOwn-Lift-Cube-UR5E-v1 --headless --num_envs 512
```

Training logs are saved to:

```
logs/skrl/lift_ur5/
```

Monitor training with TensorBoard (in a separate terminal):

```bash
../../IsaacLab/isaaclab.sh -p -m tensorboard.main --logdir logs/skrl/lift_ur5
```

---

## Evaluation / Play

```bash
../../IsaacLab/isaaclab.sh -p scripts/skrl/play.py --task=IsaacOwn-Lift-Cube-UR5E-v1 --num_envs 10
```

---

## Project Structure

```
IsaacLab_UR5e_Lift_Task/
├── scripts/
│   └── skrl/
│       ├── train.py              # PPO training entry point
│       └── play.py               # Evaluation / playback
├── source/
│   └── UR5_lift_cube/
│       └── UR5_lift_cube/
│           └── tasks/
│               └── manager_based/
│                   └── lift_cube_ur5e/
│                       ├── lift_cube_ur5e_env_cfg.py   # Full env config
│                       ├── mdp/                        # MDP components
│                       ├── agents/
│                       │   └── skrl_ppo_cfg.yaml       # PPO hyperparameters
│                       └── __init__.py                 # Task registration
├── ur5e_gripper.usdz             # UR5e + gripper asset (required)
└── README.md
```

---

## Known Issues

- **Warp CUDA warning** on startup (`Failed to get driver entry point 'cuDeviceGetUuid'`) — cosmetic, does not affect training.
- **CPU powersave mode** warning — switch to performance mode for faster training:
  ```bash
  sudo cpupower frequency-set -g performance
  ```
- **Missing `setup.py`** — if you get a `ModuleNotFoundError: No module named 'UR5_lift_cube'`, make sure you have installed the package via `pip install -e source/UR5_lift_cube` using the Isaac Lab Python interpreter (see Installation step 3).

---

## License

This project is licensed under the [MIT License](LICENSE).

It builds upon [Isaac Lab](https://github.com/isaac-sim/IsaacLab),
which is distributed under the Apache 2.0 License.

© 2025 Farkas Bálint
