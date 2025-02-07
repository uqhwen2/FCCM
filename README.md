# FCCM

## Baseline Implementations and Treatment Effect Estimators:
The valuable contributions of the broader community in advancing open-source AI are acknowledged. The folders 'bmdal_reg' and 'causal_bald' serve as baselines (e.g., LCMD, Causal-BALD) and shared estimators (e.g., DUE-DNN), referencing existing literature as follows:

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

## Mapping Convention for the Method (in Paper) and Its Alias (in Code Script):

The following aliases are used in the shell file, e.g., ```DeepGPR_TrueSim-DUE.sh```, to call the corresponded methods:

| Method          | In Script       |
|-----------------|-----------------|
| LCMD            | lcmd            |
| BADGE           | kmeanspp        |
| BAIT            | bait            |
| QHTE            | qhte            |
| $\mu$ BALD      | mu              |
| $\rho$ BALD     | rho             |
| $\mu\rho$ BALD  | murho           |
| MACAL           | macal           |
| FCCM            | fccm            |

## TOY: 

The simulation of the one-dimensional dataset is detailed on appendix, the code implemtation for such simulation is under function ```def generation()``` in fccm.py

#### Using the .sh file to run 10 simulations, examples on LCMD, $\mu\rho$ BALD, and FCCM are given as follows:
```.sh
$ cd TOY
$ bash lcmd.sh 
$ bash murho.sh
$ bash fccm.sh
$ bash xxx.sh for the other method
```

#### Visualization on the Risk Metric -- PEHE:
```.sh
$ cd text_results
```
Then, run the ```toy_all_plots.ipynb``` which imports all text reuslts from csv file, e.g., from fccm/, averging 10 seeds, then save the generated figures with error bar in ```TOY_PEHE.pdf```.

#### Tunining the FCCM:

Run ```fccm_radius_tuning.sh```, the tuning resulta are saved under ```text_results/coverage_visuals_{}```, then run ```delta_tuning.ipynb``` for the 95% threshold identification for the radius.

#### Ablation Stduy Without the Counterfactual Covering:

Move the results of FCCM ```text_results/fccm``` to folder ```ablation_study``` then run ```abaltion_study.sh```, get the comparisons by running ```ablation_study.ipynb```.







