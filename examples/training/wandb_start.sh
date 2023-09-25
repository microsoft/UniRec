# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# !/bin/bash

## Step 1. Initialize sweeps with CLI using configuration file. 
## For more details, please refer to https://docs.wandb.ai/guides/sweeps/initialize-sweeps

# wandb sweep config.yaml


## Step 2. After `wandb sweep`, you would get a sweep id and the hint to use `sweep agent`, like:

## wandb: Creating sweep from: ./wandb.yaml
## wandb: Created sweep with ID: xxx
## wandb: View sweep at: https://wandb.ai/xxx/xxx/xxx/xxx
## wandb: Run sweep agent with: wandb agent xxx/xxx/xxx/xxx

# wandb agent entity/project/sweep_ID



## Here we use the pipeline as example.

# sweep_ID=`wandb sweep ./wandb.yaml | grep "wandb agent" | sed -n 's/^.*wandb agent \(\S*\).*$/\1/p'`
sweep_ID=$(wandb sweep ./wandb.yaml 2>&1 | grep "wandb agent" | sed -n 's/^.*wandb agent \(\S*\).*$/\1/p')  
echo "Sweep ID: $sweep_ID"
wandb agent $sweep_ID
