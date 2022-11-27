# A python version of quadruped robot gait controller

This project is a personal project. Its purpose is to learn the most important ideas of the MIT cheetah controller, and reimplement them by myself. 
This project is a simplified python implementation for quadruped robot simulation, based on MIT's open-sourced cheetah 3 locomotion controller and related papers. This project is neither intend to realizing a novel or full-capable locomotion control for challenging terrains and motion abilities, nor to achieving best computational performance. These features can be extended in the future, but this implementation is written in a simple and comprehensive way just to reveal the core idea of the MIT's controllers, including 

    - Convex model predictive controller (MPC) for floating base. 
    - Simple PD controller for floating base (VMC). 
    - Whole body impulse controller (WBIC). 
    - Foothold calculation based on Raibert's Law.
    - Gait scheduler and body/leg trajectory planning.

## Dependencies:
    - Python 3.9+
    - Pinocchio dynamic library
    - Pybullet simulator
    - Numpy, scipy, matplotlib
    - qpsolvers

## Usage

Firstly install [anaconda](https://docs.anaconda.com/anaconda/install/index.html) or [miniconda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).

Run the following commands to install required libraries.
```
conda create --name pydog python=3.9
conda activate pydog
conda install numpy scipy matplotlib qpsolvers
conda install pybullet
conda install pinocchio -c conda-forge 
``` 
After, run the script
```
python RobotSim.py
```
Press `t` when the robot has loaded. It will start to trot in place. Use arrow key to control it go forward/backward/left/right. Use `q` and `e` key to control it turn left or right.

## Reference:
[1] J. Di Carlo, P. M. Wensing, B. Katz, G. Bledt, and S. Kim, Dynamic Locomotion in the MIT Cheetah 3 Through Convex Model-Predictive Control.

[2] G. Bledt, M. J. Powell, B. Katz,
J. Di Carlo, P. M. Wensing, and S. Kim. MIT Cheetah 3: Design and Control of a Robust, Dynamic Quadruped Robot.

[3] D. Kim, J. Di Carlo, B. Katz, G. Bledt, and S. Kim, Highly Dynamic Quadruped Locomotion via Whole-Body Impulse Control and Model Predictive Control.

[4] D. Kim, S. Jorgensen, J. Lee, J. Ahn, J. Luo, and L. Sentis, Dynamic Locomotion For Passive-Ankle Biped Robots And Humanoids Using Whole-Body Locomotion Control.

[5] B. Nemec and L. Zlajpah, Null space velocity control with dynamically consistent pseudo-inverse.

[6] [Unitree A1 Dog model](https://github.com/unitreerobotics/unitree_pybullet)

[7] [The Pinocchio dynamic library](https://github.com/stack-of-tasks/pinocchio)