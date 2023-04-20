# About this repository

This repository contains the code associated with the paper "Achieving Goals using Reward Shaping and Curriculum Learning", published at Future Technologies Conference (FTC) 2023.


The code builds on top of [Isaac Gym](https://github.com/gavrielstate/IsaacGymEnvs) and [RL Games](https://github.com/Denys88/rl_games).


# Setup
Create a conda environment following the instructions from installing Isaac Gym. This repository uses isaac gym preview version 3.


Available environments:  
- PickAndPlace (isaacgymenvs/tasks/pick_and_place.py)
- Tetris (isaacgymenvs/tasks/tetris.py)


Curriculum implementation can be found in `rl_games/common/train_methods.py` and it's being used in `rl_games/common/a2c_common.py`.


## Training
Within `isaacgymenvs` folder: `python train.py task=PickAndPlace`. This automatically points to the config files associated with the selected task in `isaacgymenvs/cfg/`, both within `task` and `train` folders.


# Citing

Please cite this work as:
```
@misc{mihai2023achieving,
      title={Achieving Goals using Reward Shaping and Curriculum Learning}, 
      author={Mihai Anca and Jonathan D. Thomas and Dabal Pedamonti and Mark Hansen and Matthew Studley},
      year={2023},
      journal={Future Technologies Conference (FTC) 2023}
}
```