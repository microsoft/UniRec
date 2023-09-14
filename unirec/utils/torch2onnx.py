import torch
import numpy as np
from unirec.utils import logger, general
import argparse
import random
import os
from onnxruntime import InferenceSession, SessionOptions

def parse_cmd_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--ckpt_file", type=str, help='torch model checkpoint file to be transformed.')
    parser.add_argument("--output_path", type=str, help='output path for onnx model and log.')
    parser.add_argument("--atol", type=float, default=1e-5, help='atol for validation.')
    parser.add_argument("--useful_names", type=str, nargs='*', default=['item_id', 'item_seq'], help="list of input names that are really used in current model's inference") 
        
    (args, unknown) = parser.parse_known_args()  
    # print(args)
    parsed_results = {}
    for arg in sorted(vars(args)):
        value = getattr(args, arg)
        if value is not None and value not in ['none', 'None']:
            parsed_results[arg] = value
    
    return parsed_results

def model_trans_and_validate(model, config, logger):
    logger.log(logger.INFO, "Transforming model ...")
    model.to('cpu')
    model.eval()

    # all possible inputs for model in unirec, currently only support sequential rec model
    dummy_inputs = {
        'user_id': np.random.randint(1, config['n_users'], size=(config['batch_size'],), dtype=np.int64),
        'item_id': np.random.randint(1, config['n_items'], size=(config['batch_size'],), dtype=np.int64),
        'label': np.random.randint(0, 2, size=(config['batch_size'],), dtype=np.int64),
        'item_features': np.ones([config['batch_size'], len(eval(config.get('features_shape', '[]')))], dtype=np.int64),
        'item_seq': np.random.randint(1, config['n_items'], size=(config['batch_size'], config['max_seq_len']), dtype=np.int64),
        'item_seq_len': np.array([config['max_seq_len'] for i in range(config['batch_size'])], dtype=np.int64),
        'item_seq_features': np.ones([config['batch_size'], config['max_seq_len'], len(eval(config.get('features_shape', '[]')))], dtype=np.int64),
        'time_seq': np.random.randint(1, config['time_seq'] if config['time_seq']>0 else 2, size=(config['batch_size'], config['max_seq_len']), dtype=np.int64),
        'session_id': np.ones([config['batch_size']], dtype=np.int64),
    }

    #params for torch.onnx.export
    output_names = ['scores', 'user_embedding', 'item_embedding']
    useful_names = config['useful_names']

    input_names = list(dummy_inputs.keys()) #get the input names of model.forward
    sample_input_tuple = tuple([torch.from_numpy(dummy_inputs[input_name]) for input_name in input_names])
    dynamic_axes = {k: {0: 'batch_size'} for k in input_names+output_names}

    torch.onnx.export(model,               # model being run
                        sample_input_tuple,                         # model input (or a tuple for multiple inputs)
                        os.path.join(config["output_path"], config['exp_name']+".onnx"),   # where to save the model (can be a file or file-like object)
                        export_params=True,         # store the trained parameter weights inside the model file
                        opset_version=15,          # the ONNX version to export the model to
                        do_constant_folding=True,  # whether to execute constant folding for optimization
                        input_names = input_names,   # the model's input names
                        output_names = output_names, # the model's output names
                        dynamic_axes=dynamic_axes    # variable length axes
                    )
    
    logger.log(logger.INFO, "Validating model ...")
    _, ref_scores, ref_user_embedding, ref_item_embedding = model(*sample_input_tuple)
    ref_outputs = [ref_scores.detach().numpy(), ref_user_embedding.detach().numpy(), ref_item_embedding.detach().numpy()]

    # load onnx model and run inference
    options = SessionOptions()
    ort_session = InferenceSession(os.path.join(config["output_path"], config['exp_name']+".onnx"), options, providers=["CPUExecutionProvider"])
    ort_inputs = {k: dummy_inputs[k] for k in input_names if k in useful_names}
    ort_outs = ort_session.run(None, ort_inputs) # all the none-tensor outputs will be ignored

    # compare onnx-runtime and pytorch results
    for name, ref, ort in zip(output_names, ref_outputs, ort_outs):
        if not np.allclose(ref, ort, atol=config['atol']):
            bad_indices = np.logical_not(np.isclose(ref, ort, atol=config['atol']))
            logger.log(logger.INFO, f"[x] {name} values not close enough (atol: {config['atol']})")
            raise ValueError(
                "Outputs values doesn't match between reference model and ONNX exported model: "
                f"Got max absolute difference of: {np.amax(np.abs(ref - ort))} for "
                f"{ref[bad_indices]} vs {ort[bad_indices]}"
            )
        else:
            logger.log(logger.INFO, f"[✓] {name} values close (atol: {config['atol']})")

    logger.log(logger.INFO, 'Done.')


#currently only support sequential rec model
if __name__ == '__main__':
    config = parse_cmd_arguments()
    logger_dir = 'output' if 'output_path' not in config else config['output_path']
    logger_time_str = general.get_local_time_str().replace(':', '')
    logger_rand = random.randint(0, 100)
    mylog = logger.Logger(logger_dir, 'onnx', logger_time_str, logger_rand)
    mylog.log(mylog.INFO, 'config='+str(config))
    mylog.log(mylog.INFO, "Loading model from checkpoint: {0} ...".format(config['ckpt_file']))
    model, cpt_config = general.load_model_freely(config['ckpt_file'], device='cpu')
    cpt_config.update(config)
    model_trans_and_validate(model, cpt_config, mylog)
    