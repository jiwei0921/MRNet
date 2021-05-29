# MRNet Code


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

### Contact Us
If you have any questions, please contact us ( wji3@ualberta.ca ).
