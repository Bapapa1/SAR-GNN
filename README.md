# Saliency-Aware Regularized Graph Neural Network(SAR-GNN)


![Architecture of the Saliency-Aware Regularized Graph Neural Network (SARGNN).](https://github.com/Bapapa1/SAR-GNN/blob/main/framwork.jpg?raw=true)
## Overview

Here we provide the implementation of a SAR-GCN layer in Pytorch, along with a execution example (on the MUTAG dataset). The repository is organised as follows:

+ ```GNN_models/``` contains the implementation of the backbone network, e.g GCN.py (the code of SAR-GCN)
+ ```K_fold/``` contains the [evaluation framwork](https://arxiv.org/abs/1912.09893)
+ ```hyper_config/``` contains:
  - Adjusting the hyperparameter search range for 'Weighted sum' (```GNN_hyper_model.yml```);
  - Adjusting the hyperparameter search range for ' Scaling regularization' (```GNN_mutl_hyper_model.yml```);
+ ```models/``` contains:
  - Early-stop training mechanism (```EarlyStopper.py```)
  - Data pre-processing program （```data_splits.py```,```mask_data.py```,```social_degree.py```）
  - Utilities (```utils.py```)
  - The implementation of the Graph Neural Memory layer (```M_layers.py```)
  - The End-to-end training program (```Experiment.py```)
  


## RUN
Run ```main.py```, the main script file used for training and/or testing the model. The following options are supported:
```
python main.py [--cuda] [--seed] [--data_config] [--detail] [--log_every]
               [--heads] [--outer_folds] [--Integration_method] [--GNN_models]
               [--repeat] [--dataset_name] [--local_rank]
```

## Cite
Please cite our paper if you use this code in your own work:
```
@article{pei2024saliency,
  title={Saliency-Aware Regularized Graph Neural Network},
  author={Pei, Wenjie and Xu, Weina and Wu, Zongze and Li, Weichao and Wang, Jinfan and Lu, Guangming and Wang, Xiangrong},
  journal={Artificial Intelligence (AIJ)},
  year={2024}
}
```

