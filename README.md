### [Learning Calibrated Medical Image Segmentation via Multi-rater Agreement Modeling](https://openaccess.thecvf.com/content/CVPR2021/papers/Ji_Learning_Calibrated_Medical_Image_Segmentation_via_Multi-Rater_Agreement_Modeling_CVPR_2021_paper.pdf) accepted by CVPR 2021.
------
![avatar](https://github.com/jiwei0921/MRNet/blob/main/Introduction.png) 

------

### Introduction for Calibrated Medical Image Segmentation

As depicted in the figure above, in medical image analysis, it is typical to collect multiple annotations, each from a different clinical expert or rater, in the expectation that possible diagnostic errors could be mitigated. Meanwhile, from the computer vision practitioner viewpoint, it has been a common practice to adopt the ground-truth labels obtained via either the majority-vote or simply one annotation from a preferred rater. This process, however, tends to overlook the rich information of agreement or disagreement ingrained in the raw multirater annotations. To address this issue, we propose to explicitly model the multi-rater (dis-)agreement, i.e., **MRNet**, which effectively improves the calibrated performance for generic medical image segmentation tasks. 

------

## MRNet Code

### > Requirment
+ pytorch 1.0.0+
+ torchvision
+ PIL
+ numpy
+ tensorboard==1.7.0
+ tensorboardX==2.0


### > Usage

Notes for parameters in ```demo.py```:
```
- set 'max_iteration' as **Number**         # e.g. 2000000, which means the maximum of iteration
- set ‘spshot’        as **Number**         # e.g. 20000, which means saving checkpoint every 20000 iterations
- set 'nclass'        as **Number**         # e.g. 2, which means binary tasks. i.e. model output
- set 'b_size'        as **Number**         # e.g. 2, whcih means batch size for training
- set 'sshow'         as **Number**         # e.g. 20, which means showing the training loss every 20 iterations
- set '--phase'       as **Train or Test**
- set '--param'       as **True or False**  # whether load checkpoint or not
- set '--dataset'     as **test_name**      # set test or val dataset
- set '--snap_num'    as **Number**         # e.g. 80000, load 80000th checkpoint
- set 'gpu_ids'       as **String**         # e.g. '0,1', which means running on 0 and 1 GPUs
- You need to set "train_data" and "test_data" path in demo.py
More details in demo.py
```


### Run MRNet Code

1. train

```python demo.py```                              # set '--phase' as train

2. test

```python demo.py```                              # set '--phase' as test

3. Load the log file

```cat ./Out/log/_*_.log```

4. Load the training details 

```tensorboard --logdir=/YourComputer/model_template/runs/_*_.local/```


### Dataset
1. RIGA benchmark: you can access to this download [link](https://pan.baidu.com/s/1PdvsVUOduuaJ7l4yxyZbew), with fetch code (**urob**). 
2. QUBIQ challenge: the formal web link is [here](https://qubiq.grand-challenge.org/participation/). 

### Bibtex
```
@InProceedings{Ji_2021_MRNet,
    author    = {Ji, Wei and Yu, Shuang and Wu, Junde and Ma, Kai and Bian, Cheng and Bi, Qi and Li, Jingjing and Liu, Hanruo and Cheng, Li and Zheng, Yefeng},
    title     = {Learning Calibrated Medical Image Segmentation via Multi-Rater Agreement Modeling},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {12341-12351}
}
```

### Contact Us
If you have any questions, please contact us ( wji3@ualberta.ca ).
