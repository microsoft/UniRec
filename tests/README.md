# Tests

In this document we show our test infrastructure and how to contribute tests to the repository.

## Tabel of Contents

---

0. `test_preprocess.py`: used to process the raw dataset and save temporal files in `UniRec/tests/.temp/data`
1. `test_seq_model.py`: used to test train, test, infer and embedding tasks for all sequential models, including metrics, output shape
2. `test_cf_model.py`: used to test train, test, infer and embedding tasks for all CF-based models, including metrics, output shape. Embeddding tasks are only supported for SGD based models like MF and MultiVAE.
3. `test_morec.py`: used to test train pipeline of our proposed multi-objective framework MoRec.
4. `test_multi_gpu.py`: used to test the correctness of DDP settings.

In total, there are 14 test cases in those files.

Note: To pass `test_multi_gpu.py`, you should to use a machine with at least two GPUs. Otherwise it would not pass successfully.

## Usage

**`test_preprocess.py` **should be run first as data preparation for successive tests

---

In current version, we set all test tasks as required.

```bash
cd tests
pytest
```

It takes about 2 minutes to pass all test cases (on A100-80GB). On the machine with two V100-32GB, it takes about 3-4 minutes.
