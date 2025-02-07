# FCCM

## Baseline Implementations and Treatment Effect Estimators:
The valuable contributions of the broader community in advancing open-source AI is acknowledge. The folders 'bmdal_reg' and 'causal_bald' serve as baselines (e.g., LCMD, Causal-BALD) and shared estimators (e.g., DUE-DNN), referencing existing literature as follows:

BMDAL: "[Black-Box Batch Active Learning for Regression](https://arxiv.org/abs/2302.08981)":

```bibtex
@misc{kirsch2023blackbox,
    title={Black-Box Batch Active Learning for Regression},
    author={Andreas Kirsch},
    year={2023},
    eprint={2302.08981},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```
Causal-BALD: [Causal-BALD: Deep Bayesian Active Learning of Outcomes to Infer Treatment-Effects from Observational Data](https://arxiv.org/abs/2111.02275) as

```bibtex
@article{jesson2021causal,
  title={Causal-BALD: Deep Bayesian Active Learning of Outcomes to Infer Treatment-Effects from Observational Data},
  author={Jesson, Andrew and Tigas, Panagiotis and van Amersfoort, Joost and Kirsch, Andreas and Shalit, Uri and Gal, Yarin},
  journal={Advances in Neural Information Processing Systems},
  volume={35},
  year={2021}
}
```

## Package Installation:

Testing Environment: 24GB NVIDIA RTX-3090 on Ubuntu 22.04 LTS platform with 12th Gen Intel i7-12700K 12-Core 20-Thread CPU.

```.sh
$ conda create --name FCCM python=3.9
$ conda activate FCCM
$ pip install -r requirements.txt  # install the BMDAL baselines for benchmarking
$ pip install .  # install the Causal-Bald baselines for benchmarking
$ pip install --upgrade torch==2.1.1 torchvision==0.16.1 -f https://download.pytorch.org/whl/cu118/torch_stable.html
```
