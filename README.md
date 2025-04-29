# Residual Learning for Robotic Grasping
Remember to activate the python virtual environment before running programs.
## Train models
```
python train.py --log_path runs/{folder name}
```

Somehow, vectorized environments with Stable-Baseline3 don't function properly on MacOS, so run
```
python train.py --train_single_env --test
```
to debug and test for proper implementation.
## Visualization with TensorBoard
```
tensorboard --logdir runs
```

## Results (04/25)
Residual policy: 662/1000
Residual policy (without prediction known): 710/1000


### Notes
When running on lab computers, run the following command in the parent folder to include Drake.
```
source setup.sh
```

In Drake, a state vector of a free body is `[rotation (quaternion) translation angular_velocity linear_velocity]`, which has 4+3+3+3=13 dimensions.
