# [Learning Models of Adversarial Agent Behavior under Partial Observability](https://arxiv.org/pdf/2306.11168.pdf)

### Authors: Sean Ye, Manisha Natarajan, Zixuan Wu, Rohan Paleja, Letian Chen, and Matthew C. Gombolay

### (To Appear at IROS 2023)

---

We present two large scale adversarial tracking environments: Prison Escape and Narco Traffic Interdiction as discussed in the paper. This repository contains the code to run the environments and collect the datasets. The codebase for training models is located [here](https://github.com/CORE-Robotics-Lab/GrAMMI). 

### Installation
After cloning the repository, please use the provided conda environment (`environment.yml`) file to install dependencies:
`conda env create -f environment.yml`

This will create an environment named 'tracking_env'. Please edit the first line of `environment.yml` to name it something else.

## 1.Prison Escape
### About the Environment
A heterogeneous team of cameras, search parties, and helicopters (blue team) must coordinate to track an escaped prisoner
(red team). The game is played on a $2428 \times 2428$ map with varying terrains where each cell on the grid represents 
the $(x,y)$ location. This domain is motivated by scenarios in military surveillance and border patrol, where there is a
need to track and intercept adversarial targets to ensure the safety of the general population. 

### Simulator
Within our environment, we have several classes to represent the terrain, 
different objects (town, camera, etc.), and step all moving objects based 
on various agent policies/heuristics, which you can find under the `Prison_Escape/` folder. 
If you would like to know the details of our environment configuration
(state space, observation space, action space, etc.), please refer to [this file](./Prison_Escape/environment/prisoner_env.py).

**Rendering:** We have two modes for rendering the Prison Escape environment. We have a fast option that is less aesthetic,
and a slow option that is more aesthetic.
For training and debugging, please use the fast option.
For visualizing results, please use the slow rendering option to get the best display.

### Collecting the Dataset
Run `Prison_Escape/collect_demonstrations.py` to collect train and test datasets. Please specify the 
parameters as mentioned in the main function. Each rollout is saved as a numpy file, and includes observations from both the blue
and the red team's perspective, the hideout locations, the current timestep, whether the prisoner was seen, and done to indicate
the end of the episode. All values are stored for every timestep of each rollout.

In our paper, we describe three datasets for Prison Escape. We obtain this by varying the detection factor
in the simulator config file: `Prison_Escape/environment/configs/balance_game.yaml`

## 2. Narco Traffic Interdiction: 
### About the Environment
This domain simulates illegal maritime drug trafficking on a $7884 \times 3538$ grid along the Central American Pacific 
Coastline. The adversary, a drug smuggler, is pursued by a team of heterogeneous tracker agents comprising airplanes and
marine vessels. Airplanes have a larger search radius and speed than marine vessels, but only the vessels can capture 
the smuggler. Smugglers must first reach rendezvous points before heading to the hideouts, representing drug handoffs at
sea. The locations of hideouts and rendezvous points are unknown to the tracking team. Episodes start after the team 
learns one location of the smuggler and end when the smuggler reaches a hideout or is captured by law enforcement.

### Simulator
The Narco Traffic domain is setup very similar to the Prison Escape environment, in that we have several classes to represent the terrain, 
different objects (town, camera, etc.), and step all moving objects based 
on various agent policies/heuristics, which you can find under the `Smuggler/` folder. 
If you would like to know the details of our environment configuration
(state space, observation space, action space, etc.), please refer to [this file](./Smuggler/simulator/smuggler_env.py).


### Collecting the Dataset
Run `Smuggler/collect_dataset.py` to collect train and test datasets. Please specify the 
parameters as mentioned in the main function. Each rollout is saved as a numpy file, and includes observations from both the blue
and the red team's perspective, the hideout locations, the current timestep, whether the smuggler was detected, and done to indicate
the end of the episode. All values are stored for every timestep of each rollout.

[//]: # (In our paper, we describe two datasets for Narco Traffic Interdiction. We obtain this by varying the parameters as specified)

[//]: # (in the simulator config file: `Prison_Escape/environment/configs/balance_game.yaml`)
---
## Citation

If you find our code or paper is useful, please consider citing:

```bibtex
@inproceedings{ye2023grammi,
  title={Learning Models of Adversarial Agent Behavior under Partial
Observability},
  author={Ye, Sean and Natarajan, Manisha and Wu, Zixuan and Paleja, Rohan and Chen, Letian and Gombolay, Matthew},
  booktitle={IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year={2023}
}
```

## License

This code is distributed under an [MIT LICENSE](LICENSE).