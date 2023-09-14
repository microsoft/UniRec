# Firstly you need to install and login wandb
# ```
# pip install wandb
# wandb login
# ```


LOCAL_ROOT="/home/jialia/adaretriever/UniRec"
cd $LOCAL_ROOT

wandb sweep --project adaretriever unirec/shell/ada-retrieval/wandb.yaml


# terminal log will be like this:
# wandb: Creating sweep from: unirec/config/wandb_sweep/config_adaretriever.yaml
# wandb: Created sweep with ID: gmnrvw8p
# wandb: View sweep at: https://wandb.ai/unirec2023/adaretriever/sweeps/gmnrvw8p
# wandb: Run sweep agent with: wandb agent unirec2023/adaretriever/gmnrvw8p

# Then follow its last command to launch the sweep agent:
# wandb agent unirec2023/adaretriever/gmnrvw8p

# Parallelize on a multi-GPU machine
# if you want to run on 2 GPUs, you can run in two different terminals:
# CUDA_VISIBLE_DEVICES=0 wandb agent unirec2023/adaretriever/gmnrvw8p
# CUDA_VISIBLE_DEVICES=1 wandb agent unirec2023/adaretriever/gmnrvw8p
# Note: you should set gpu_id=-1 in the run_**.sh to prevent from allocating unexpected GPU.