# Residual Learning for Robotic Grasping
Remember to activate the python virtual environment before running programs.
## Train models
```
python train.py --log_path runs/{folder name}
```
## Visualization with TensorBoard
```
tensorboard --logdir runs
```

### Notes
When running on lab computers, run the following command in the parent folder to include Drake.
```
source setup.sh
```
