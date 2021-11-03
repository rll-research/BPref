# B-Pref

Official codebase for [B-Pref: Benchmarking Preference-BasedReinforcement Learning](https://openreview.net/forum?id=ps95-mkHF_) contains scripts to reproduce experiments.


## Install

```
conda env create -f conda_env.yml
pip install -e .[docs,tests,extra]
cd custom_dmcontrol
pip install -e .
cd custom_dmc2gym
pip install -e .
pip install git+https://github.com/rlworkgroup/metaworld.git@master#egg=metaworld
pip install pybullet
```

## Run experiments using GT rewards


### SAC & SAC + unsupervised pre-training

Experiments can be reproduced with the following:

```
./scripts/[env_name]/run_sac.sh 
./scripts/[env_name]/run_sac_unsuper.sh
```


### PPO & PPO + unsupervised pre-training

Experiments can be reproduced with the following:

```
./scripts/[env_name]/run_ppo.sh 
./scripts/[env_name]/run_ppo_unsuper.sh
```

## Run experiments on irrational teacher

To design more realistic models of human teachers, we consider a common stochastic model and systematically manipulate its terms and operators:

```
teacher_beta: rationality constant of stochastic preference model (default: -1 for perfectly rational model)
teacher_gamma: discount factor to model myopic behavior (default: 1)
teacher_eps_mistake: probability of making a mistake (default: 0)
teacher_eps_skip: hyperparameters to control skip threshold (\in [0,1])
teacher_eps_equal: hyperparameters to control equal threshold (\in [0,1])
```

In B-Pref, we tried the following teachers:

`Oracle teacher`: (teacher_beta=-1, teacher_gamma=1, teacher_eps_mistake=0, teacher_eps_skip=0, teacher_eps_equal=0)

`Mistake teacher`: (teacher_beta=-1, teacher_gamma=1, teacher_eps_mistake=0.1, teacher_eps_skip=0, teacher_eps_equal=0)

`Noisy teacher`: (teacher_beta=1, teacher_gamma=1, teacher_eps_mistake=0, teacher_eps_skip=0, teacher_eps_equal=0)

`Skip teacher`: (teacher_beta=-1, teacher_gamma=1, teacher_eps_mistake=0, teacher_eps_skip=0.1, teacher_eps_equal=0)

`Myopic teacher`: (teacher_beta=-1, teacher_gamma=0.9, teacher_eps_mistake=0, teacher_eps_skip=0, teacher_eps_equal=0)

`Equal teacher`: (teacher_beta=-1, teacher_gamma=1, teacher_eps_mistake=0, teacher_eps_skip=0, teacher_eps_equal=0.1)


### PEBBLE

Experiments can be reproduced with the following:

```
./scripts/[env_name]/[teacher_type]/[max_budget]/run_PEBBLE.sh [sampling_scheme: 0=uniform, 1=disagreement, 2=entropy]
```

### PrefPPO

Experiments can be reproduced with the following:

```
./scripts/[env_name]/[teacher_type]/[max_budget]/run_PrefPPO.sh [sampling_scheme: 0=uniform, 1=disagreement, 2=entropy]
```

note: full hyper-paramters for meta-world will be updated soon!

