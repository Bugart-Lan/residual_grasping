import wandb

api = wandb.Api()
run = api.run("/wandb/run-20250424_143014-oy67gzxd")
print(run.history())