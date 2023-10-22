# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import os
import sys
from typing import Dict

from unirec.utils import file_io

def parse_cmd_arguments():
    parser = argparse.ArgumentParser()
    
    ## define general arguments
    parser.add_argument("--exp_name", type=str)
    parser.add_argument("--config_file", type=str, help='yaml file to store the default configuration. config_file and config_dir are conflicting. Only need to specify one.')  
    parser.add_argument("--config_dir", type=str, help='yaml folder to store the default configuration. config_file and config_dir are conflicting. Only need to specify one.')  
    parser.add_argument("--seed", type=int, help='the random seed to be fixed for reproducing experiments.') 
    parser.add_argument("--model", type=str)
    parser.add_argument("--dataloader", type=str)
    parser.add_argument("--task", type=str)
    parser.add_argument("--wandb_file", type=str)
    parser.add_argument("--use_wandb", type=int, choices=[0,1], help="whether to enable wandb")
    parser.add_argument("--use_tensorboard", type=int, choices=[0,1], help="whether to enable tensorboard")
    
    # different from `model_file`. `model_file` is used to load model while `checkpoint_dir` is used to 
    # save model. Sometimes we need to load a pretrained model and finetune it, then save it to a specific
    # directory, both `model_file` and `checkpoint_dir` are both required.
    # Note: when `checkpoint_dir` is not given, it would be generated with current time.  
    parser.add_argument("--checkpoint_dir", type=str)   

    
    ### training specific arguments
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--num_workers_train", type=int)
    parser.add_argument("--num_workers_valid", type=int)
    parser.add_argument("--num_workers_test", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--grad_clip_value", type=float)
    parser.add_argument("--score_clip_value", type=float)
    parser.add_argument("--gpu_id", type=int)
    parser.add_argument("--verbose", type=int)
    parser.add_argument("--metrics", type=str)
    parser.add_argument("--key_metric", type=str)
    parser.add_argument("--device", type=str)
    parser.add_argument("--shuffle_train", type=int)
    parser.add_argument("--early_stop", type=int)
    parser.add_argument("--init_method", type=str)
    parser.add_argument("--init_std", type=float)
    parser.add_argument("--init_mean", type=float)
    parser.add_argument("--scheduler", type=str)
    parser.add_argument("--scheduler_factor", type=float)
    parser.add_argument("--optimizer", type=str)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--train_batch_size", type=int)
    parser.add_argument("--valid_batch_size", type=int)
    parser.add_argument("--test_batch_size", type=int)
    parser.add_argument("--learning_rate", type=float)
    
    ### data specific arguments
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--dataset_path", type=str) #DATA_TRAIN_NAME
    parser.add_argument("--data_train_name", type=str)
    parser.add_argument("--data_valid_name", type=str)
    parser.add_argument("--data_test_name", type=str)
    parser.add_argument("--output_path", type=str)  
    parser.add_argument("--train_file_format", type=str) 
    parser.add_argument("--valid_file_format", type=str)
    parser.add_argument("--test_file_format", type=str)
    parser.add_argument("--user_history_file_format", type=str)
    parser.add_argument("--user_history_filename", type=str)
    parser.add_argument("--item_meta_morec_filename", type=str)
    parser.add_argument("--align_dist_filename", type=str)

    parser.add_argument("--test_protocol", type=str)
    parser.add_argument("--valid_protocol", type=str)
    parser.add_argument("--n_sample_neg_train", type=int)
    parser.add_argument("--n_sample_neg_valid", type=int)
    parser.add_argument("--n_sample_neg_test", type=int)
    parser.add_argument("--group_size", type=int)
    
    parser.add_argument("--neg_by_pop_alpha", type=float, help="If neg_by_pop_alpha > 0, negative items will be sampled by item popularity, with neg_by_pop_alpha as the adjustment coeffciency.")

    parser.add_argument("--model_file", type=str, help="checkpoint model file, loaded for evaluation task or continual training")
    parser.add_argument("--load_pretrained_model", type=int, choices=[0, 1], help="load pretrained model for continual training, model_file is required")
    parser.add_argument("--time_seq", type=int, help="if time_seq=0, the time sequence is not considered; else, time_seq is the size of time_embedding table, nn.Embedding(time_seq, embedding_size, padding_idx=0)")
    parser.add_argument("--seq_last", type=int, choices=[0, 1], help="whether to only take the last occurrence of an item in the sequence as the target.")

    parser.add_argument("--use_features", type=int, choices=[0, 1])
    parser.add_argument("--features_filepath", type=str)
    parser.add_argument("--features_shape", type=str)

    parser.add_argument("--use_text_emb", type=int, choices=[0, 1])
    parser.add_argument("--text_emb_path", type=str)
    parser.add_argument("--text_emb_size", type=int)

    ## define model-sepcific related arguments here 
    parser.add_argument("--dropout_prob", type=float)
    parser.add_argument("--hidden_dropout_prob", type=float) ## in SASRec
    parser.add_argument("--attn_dropout_prob", type=float) ## in SASRec   
    parser.add_argument("--embedding_size", type=int)
    parser.add_argument("--use_pre_item_emb", type=int) 
    parser.add_argument("--use_position_emb", type=int) 
    parser.add_argument("--item_emb_path", type=str)
    parser.add_argument("--max_seq_len", type=int)
    parser.add_argument("--history_mask_mode", type=str)
    parser.add_argument("--has_user_bias", type=int)
    parser.add_argument("--has_item_bias", type=int)
    parser.add_argument("--hidden_size", type=int)  # actually no use now
    parser.add_argument("--inner_size", type=int)
    parser.add_argument("--edge_norm", type=str, choices=['none', 'sqrt_degree']) 
    parser.add_argument("--loss_type", type=str)
    parser.add_argument("--ccl_w", type=float)
    parser.add_argument("--ccl_m", type=float)
    parser.add_argument("--distance_type", type=str, help="Distance metric for vector similarity. mlp, dot or cosine.")
    parser.add_argument("--weight_decay", type=float) 
    parser.add_argument("--n_layers", type=int)
    parser.add_argument("--layer_norm_eps", type=float)
    parser.add_argument("--hidden_act", type=str)
    parser.add_argument("--tau", type=float, help="Temperature parameter for softmax type loss")
    parser.add_argument("--asymmetric", type=int, choices=[0, 1], help="whether src item embedding and src item embeding are different in AvgHist")
    parser.add_argument("--user_sequence_alpha", type=float, 
                        help="controls the coef in modeling user history in AvgHist, the coef is `user_sequence_alpha` power of the history length")
    parser.add_argument("--seq_decay", type=float)

    ## ConvFormer-series model
    parser.add_argument("--conv_size", type=int)
    parser.add_argument("--padding_mode", type=str, choices=['circular', 'reflect', 'constant'])
    parser.add_argument("--seq_merge", type=int, choices=[0, 1])
    parser.add_argument("--init_ratio", type=float)

    ## VAE model
    parser.add_argument("--encoder_dims", type=int, nargs='+', help='VAE encoder dimensions')
    parser.add_argument("--decoder_dims", type=int, nargs='+', help='VAE encoder dimensions')
    parser.add_argument("--anneal_cap", type=float, help='max anneal coef')
    parser.add_argument("--total_anneal_steps", type=int, help='anneal reaches anneal_cap after linearly increasing n steps')
    parser.add_argument("--eval_reparameter_sampling_times", type=int, help="controls the diversity in VAE prediction, more sampling times indicate less diversity")

    ## SLIM/EASE model
    parser.add_argument("--l1_coef", type=float, help='coef of L1 norm in SLIM')
    parser.add_argument("--l2_coef", type=float, help='coef of L2 norm in SLIM and EASE')

    ## ADMM SLIM model
    parser.add_argument("--item_spec_reg", type=float, help='item-specific regularization in ADMM SLIM')
    parser.add_argument("--admm_penalty", type=float, help="ADMM penalty coef(rho) in ADMM SLIM")    

    ## FM model
    parser.add_argument("--linear_mode", type=str, help='sparse linear mode in FM')

    # MoRec arguments
    parser.add_argument("--enable_morec", type=int, choices=[0,1]) 
    parser.add_argument("--morec_objectives", type=str, nargs='*', help="Objective list.") 
    parser.add_argument("--morec_objective_controller", type=str, choices=["PID", "Static"], help="Objective Controller.")
    parser.add_argument("--morec_objective_weights", type=str, help="Objective static weights. \
                        For PID controller, the length should be equal to the number of objectives except accuracy. \
                        For Static controller, the length should be the number of all objectives including accuracy and the last weight is for accuracy.")  # [0.3,0.4,0.3]
    parser.add_argument("--morec_ngroup", type=int, help="Number of groups for data sampling for objective like revenue. \
                        The data would be split into N groups according to price. If ngroup is non-positive, group strategy is not used.")
    parser.add_argument("--morec_alpha", type=float, help="learning of Signed SGD in MoRec Data Sampler.")
    parser.add_argument("--morec_lambda", type=float, help="lambda used for auxiliary objectives.")
    parser.add_argument("--morec_expect_loss", type=float, help="Expect accuracy loss of PI controller.")
    parser.add_argument("--morec_beta_min", type=float, help="Min value of beta generated by PI controller.")
    parser.add_argument("--morec_beta_max", type=float, help="Max value of beta generated by PI controller.")
    parser.add_argument("--morec_K_p", type=float, help="Coef of P-part in PI controller.")
    parser.add_argument("--morec_K_i", type=float, help="Coef of I-part in PI controller.")

         
    ## define path related arguments here
    
        
    (args, unknown) = parser.parse_known_args()  
    # print(args)
    parsed_results = {}
    for arg in sorted(vars(args)):
        value = getattr(args, arg)
        if value is not None and value not in ['none', 'None']:
            parsed_results[arg] = value
    
    return parsed_results


def extract_user_given_args(args: Dict) -> Dict:
    """ Extract user-given arguments from the arguments dict returned by `parse_cmd_arguments` function.

    When a default value is given one argument in parser, the argument would be in the Namespace even user has not specified it in command line. 
    This logic results in that some arguments in `args` input in function `parse_arguments` would be overided by the default value in parser. 
    The function enables to distinguish those user-given arguments from the default-value arguments. 
    Actually, the priority is: user-given cmd line args > args dict > default cmd line args > default args in file

    Args:
        args(Dict): the arguments dict returned by `parse_cmd_arguments` function
    
    Returns:
        Dict: user-given arguments dict. 

    Note: the function could be deprecated when there is no default value for all arguments in parser.
    """
    res = {}
    cmd_input = sys.argv
    for k, v in args.items():
        # Note: if there are more types of arguments supported, such as "-xxx", the condition should be updated.
        if any(f"--{k}" in s for s in cmd_input):
            res[k] = v
        else:
            pass
    return res


def parse_arguments(args: Dict=None):
    # priority: input `args` dict > cmd line args > ckpt config > YAML file > default args(base.yaml)
    cmd_arg = parse_cmd_arguments()
    if args is not None:
        # `args` input > cmd line
        cmd_arg.update(args)
    
    config = {}
    # default args in unirec
    default_config_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config'))

    # if `config_dir` not given, use the default config dir
    config_dir = cmd_arg.get('config_dir', default_config_dir)
    print("Load configuration files from {}".format(config_dir))
    config = file_io.load_yaml(os.path.join(config_dir, 'base.yaml'))
    config.update(file_io.load_yaml(os.path.join(config_dir, 'model', cmd_arg['model']+'.yaml')))
    config.update(file_io.load_yaml(os.path.join(config_dir, 'dataset', cmd_arg['dataset']+'.yaml')))  

    # if `config_file` is given, update config with arguments in config file
    if 'config_file' in cmd_arg:
        print("Update configuration with the file {}".format(cmd_arg['config_file']))
        config_from_file = file_io.load_yaml(cmd_arg['config_file']) 
        config.update(config_from_file)

    # cmd line arguments and `args` input
    config.update(cmd_arg)
    config['cmd_args'] = cmd_arg 
    return config
