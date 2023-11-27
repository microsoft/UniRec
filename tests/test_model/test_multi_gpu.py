# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import sys

_path = os.path.dirname(__file__)
UNIREC_PATH = os.path.abspath(os.path.join(_path, "../.."))
sys.path.append(UNIREC_PATH)

import re
import copy
import pytest
import torch
import datetime
import subprocess



TOL = 0.05
ABS_TOL = 0.05

@pytest.mark.parametrize(
        "expected_values",
        [
            (
                {'hit@20': 0.21831, 'ndcg@20': 0.09807}
            )
        ]
)
def test_ddp_sh(expected_values):
    gpu_info = subprocess.check_output("nvidia-smi -L", shell=True).decode('utf-8')
    num_gpus = len(gpu_info.strip().split('\n'))
    assert num_gpus > 1, f"At least two GPUs are required to test ddp, while only {num_gpus} GPUs found."
    shell_path = os.path.join(UNIREC_PATH, 'tests/test_model/run_ddp_test.sh')
    output = subprocess.run(["bash", shell_path], capture_output=True, text=True)
    assert "Logger close successfully" in output.stdout, "The shell script has not been completed because logger is not closed successfully. " \
           f"Here are the log message: \n{output.stderr}."
    test_result = eval(re.findall("test result: ({.*})", output.stderr)[0])
    for k,v in expected_values.items():
        assert test_result[k] == pytest.approx(v, rel=TOL, abs=ABS_TOL), f"performance of metric {k} not correct"



if __name__ == "__main__":
    pytest.main(["test_multi_gpu.py", "-s"])
    