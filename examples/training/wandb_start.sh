# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# !/bin/bash

## Step 0: Login to wandb
# wandb login

## Step 1. Initialize sweeps with CLI using configuration file.  For more details, please refer to https://docs.wandb.ai/guides/sweeps/initialize-sweeps
# wandb sweep wandb.yaml


## Step 2. After `wandb sweep`, you would get a sweep id and the hint to use `sweep agent`. Messages like:
## wandb: Creating sweep from: ./wandb.yaml
## wandb: Created sweep with ID: xxx
## wandb: View sweep at: https://wandb.ai/xxx/xxx/xxx/xxx
## wandb: Run sweep agent with: wandb agent xxx/xxx/xxx/xxx

## then execute:
# CUDA_VISIBLE_DEVICES=x wandb agent entity/project/sweep_ID

## hint: you can launch multiple agents to speed up the search process, using the same sweep_ID but different CUDA_VISIBLE_DEVICES


## Below is an example of how to get the sweep_ID and launch the agent in an automatic way (single process)
sweep_ID=$(wandb sweep ./wandb.yaml 2>&1 | grep "wandb agent" | sed -n 's/^.*wandb agent \(\S*\).*$/\1/p')  
echo "Sweep ID: $sweep_ID"
wandb agent $sweep_ID
