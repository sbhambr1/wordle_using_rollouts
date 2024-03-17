# Official Code for the paper: "Reinforcement Learning Methods for Wordle: A POMDP/Adaptive Control Approach" (IEEE CoG 2023)

This repository contains the official code for the paper: "Reinforcement Learning Methods for Wordle: A POMDP/Adaptive Control Approach" by Siddhant Bhambri, Amrita Bhattacharjee & Dimitri Bertsekas, accepted at IEEE CoG 2023. 

Please checkout the shorter version of the paper [here](https://ieeexplore.ieee.org/abstract/document/10333228) and the longer version on [arXiv](https://arxiv.org/pdf/2211.10298).

## Abstract

    In this paper we address the solution of the popular Wordle puzzle, using new reinforcement learning methods, which apply more generally to adaptive control of dynamic systems and to classes of Partially Ob- servable Markov Decision Process (POMDP) problems. These methods are based on approximation in value space and the rollout approach, ad- mit a straightforward implementation, and provide improved performance over various heuristic approaches. For the Wordle puzzle, they yield on- line solution strategies that are very close to optimal at relatively modest computational cost. Our methods are viable for more complex versions of Wordle and related search problems, for which an optimal strategy would be impossible to compute. They are also applicable to a wide range of adaptive sequential decision problems that involve an unknown or fre- quently changing environment whose parameters are estimated on-line.


## Installation

**Clone the repository**:
```bash
git clone https://github.com/sbhambr1/wordle_using_rollouts
cd wordle_using_rollouts
```

**Create a virtual environment**:
```bash
python3 -m venv wordle
source wordle/bin/activate
```

**Install the requirements**:
```bash
pip install -r requirements.txt
```

## Data
Download the data from [here](https://drive.google.com/drive/folders/1Dc33j2bFBPFiTb5Kz52h3YlX-5Kmz3ud?usp=sharing) into the folder 'data' located in the project's root directory.

## Usage
```bash
python3 solver/3b1b_starter_code.py
```

## Citation
If you find this code useful, please consider citing our paper:
```
Bhambri, S., Bhattacharjee, A. and Bertsekas, D., 2023, August. Playing Wordle Using an Online Rollout Algorithm for Deterministic POMDPs. In 2023 IEEE Conference on Games (CoG) (pp. 1-4). IEEE.
```

**Shout out to Grant Sanderson's (3Blue1Brown author) [repository](https://github.com/3b1b/videos/tree/18473e5c84aa7efd9311386cfe63f9eabbf5195f/_2022/wordle) from where the Wordle simulator and Maximum Information Gain heuristic has been adopted!**