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

## Mapping Convention for the Baselines (in Paper) and Its Alias (in Code Script):

The following aliases are used in the shell file, e.g., ```DeepGPR_TrueSim-DUE.sh```, to call the corresponded methods:

| Baselines       | In Code Script  |
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

The simulation of the one-dimensional dataset is detailed on Paper Appendix, the code implemtation for such simulation is under function ```def generation()``` in fccm.py

#### Using the .sh file to run 10 simulations, examples on LCMD, $\mu\rho$ BALD, and FCCM are given as follows:
```.sh
$ cd TOY
$ bash lcmd.sh 
$ bash murho.sh
$ bash fccm.sh
$ bash xxx.sh # for the other methods
```

#### Visualization on the Risk Metric -- PEHE:
```.sh
$ cd text_results
```
Run the ```toy_all_plots.ipynb``` which imports all text reuslts from csv file, e.g., from ```text_results/fccm/```, averging 10 seeds, then save the generated figures with error bar in ```TOY_PEHE.pdf```.

#### Tunining the Radius for FCCM:

Under folder ```TOY```, run ```fccm_radius_tuning.sh```, the tuning results are saved under ```text_results/coverage_visuals_{}```, then run ```delta_tuning.ipynb``` for the 95% threshold identification for the radius.

#### Ablation Stduy Without the Counterfactual Covering:

Under folder ```TOY```, copy the test results of FCCM from ```text_results/fccm``` to folder ```ablation_study```, then run ```abaltion_study.sh``` to get the results without the counterfactual covering, and finally compare FCCM and FCCM- by running ```ablation_study.ipynb```.

## IBM: 

:exclamation::exclamation::exclamation: Due to uploading file limit, to test the IBM dataset, which is a public available asset from [here](https://github.com/IBM-HRL-MLHLS/IBM-Causal-Inference-Benchmarking-Framework/blob/master/README.md). Please download the file from [the anonymous link](https://drive.google.com/drive/folders/1fKNN-IaizwpEVUuNLtsNGOI0utahN2Hr), and place the ```ibm_train.npz``` and ```ibm_test.npz``` files under the ```IBM/dataset/ibm/```:exclamation::exclamation::exclamation:

#### Using the .sh file to run 10 simulations, examples on LCMD, $\mu\rho$ BALD, and FCCM are given as follows:
```.sh
$ cd IBM
$ bash lcmd.sh 
$ bash murho.sh
$ bash fccm.sh
$ bash xxx.sh # for the other methods
```

#### Visualization on the Risk Metric -- PEHE:
```.sh
$ cd text_results
```
Run the ```ibm_all_plots.ipynb``` which imports all text reuslts from csv file, e.g., from ```text_results/fccm/```, averging 10 seeds, then save the generated figures with error bar in ```IBM_PEHE.pdf```.

#### Tunining the Radius for FCCM:

Under folder ```IBM```, run ```fccm_radius_tuning.sh```, the tuning results are saved under ```text_results/coverage_visuals_{}```, then run ```delta_tuning.ipynb``` for the 95% threshold identification for the radius.

#### Ablation Stduy Without the Counterfactual Covering:

Under folder ```IBM```, copy the test results of FCCM from ```text_results/fccm``` to folder ```ablation_study```, then run ```abaltion_study.sh``` to get the results without the counterfactual covering, and finally compare FCCM and FCCM- by running ```ablation_study.ipynb```.


## CMNIST:

The initial execution will download the MNIST dataset from public available source, please enable Internet on your server, otherwise will run into error.

#### Using the .sh file to run 10 simulations, examples on LCMD, $\mu\rho$ BALD, and FCCM are given as follows:
```.sh
$ cd CMNIST
$ bash lcmd.sh 
$ bash murho.sh
$ bash fccm.sh
$ bash xxx.sh # for the other methods
```

#### Visualization on the Risk Metric -- PEHE:
```.sh
$ cd text_results
```
Run the ```cmnist_all_plots.ipynb``` which imports all text reuslts from csv file, e.g., from ```text_results/fccm/```, averging 10 seeds, then save the generated figures with error bar in ```CMNIST_PEHE.pdf```.

#### Tunining the Radius for FCCM:

Under folder ```CMNIST```, run ```fccm_radius_tuning.sh```, the tuning results are saved under ```text_results/coverage_visuals_{}```, then run ```delta_tuning.ipynb``` for the 95% threshold identification for the radius.

#### Ablation Stduy Without the Counterfactual Covering:

Under folder ```CMNIST```, copy the test results of FCCM from ```text_results/fccm``` to folder ```ablation_study```, then run ```abaltion_study.sh``` to get the results without the counterfactual covering, and finally compare FCCM and FCCM- by running ```ablation_study.ipynb```.




