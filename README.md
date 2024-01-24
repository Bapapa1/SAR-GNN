# Saliency-Aware Regularized Graph Neural Network(SAR-GNN)
## Overview

Here we provide the implementation of a SAR-GCN layer in Pytorch, along with a execution example (on the MUTAG dataset). The repository is organised as follows:

+ ```GNN_models/```contains the implementation of the backbone network, e.g GCN.py (the code of SAR-GCN)
+ ```K_fold/``` contains the [evaluation framwork](https://arxiv.org/abs/1912.09893). 
+ ```hyper_config/```
+ ```models/```
  



## File
GNN_hyper_model.yml : Adjusting the hyperparameter search range for 'Weighted sum' 

GNN_mutl_hyper_model.yml: Adjusting the hyperparameter search range for ' Scaling regularization'

## RUN
```
python main.py
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

