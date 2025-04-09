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

### Notes
When running on lab computers, run the following command in the parent folder to include Drake.
```
source setup.sh
```
