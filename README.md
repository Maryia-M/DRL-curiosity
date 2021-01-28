## Uncertainty-Modulated Episodic Curiosity

In this experiments we used library [Stable Baselines 3](https://github.com/DLR-RM/stable-baselines3).

This is based on https://github.com/khalilsarwari/episodic-curiosity

you need to install the following dependencies gym>=0.17, numpy, torch>=1.4.0, dotmap, cloudpickle, pandas, matplotlib

and the library stable_baselines3 itself

# Running an Experiment

Running a normal experiment, no episodic curiousity

`python train_and_test_ppo.py -exp ppo_montezuma`

Running with episodic curiosity

`python train_and_test_ec.py -exp ppo_eco_montezuma`

Running with ICM

`python train_and_test_icm.py -exp ppo_icm_montezuma`
