# MRNet


```
###########目录结构描述
├── Readme.md                   // help
├── demo.py                     // 主运行代码(train or test)
├── utils                       // 数据加载,以及常用Tools
│   ├── dataset_loader.py       // 数据加载
│   ├── utils.py                // Logger, Save等常用接口
│   ├── generate_csv.py         // 数据集CSV文件生成
│   ├── visualize_img.py        // 特征图可视化
│   └── evaluateFM.py           // 评估函数(F-measure, MAE) for binary
├── models                      
│   ├── Unet                    // 加载常用Unet结构,包括ResNet18等
│   ├── VGGnet                  // 加载VGG model
│   └── ...                     // 自定义model
├── trainer                     
│   ├── trainer.py              // train类,train & val
│   └── ...                     // 自定义train方法
├── runs                        // tensorboard events文件
├── Out                         
│   ├── snapshot                // checkpoint存放目录
│   ├── results                 // prediction存放目录
│   ├── val                     // validation存放目录
│   └── log                     // train or test日志存放目录
├── losses.py                   // 常用loss函数以及自定义loss
├── optimizers.py               // 常用优化函数 e.g. SGD, Adam
└── config.py                   // 模型超参
```
 
```
###########数据结构描述
Dataset
│
│
├── train_data
│   │
│   ├── train_your_name1             
│   │   ├── train_images
│   │   │   ├──1.jpg
│   │   │   ├──2.jpg
│   │   │   └──...
│   │   ├── train_masks
│   │   │   ├──1.png
│   │   │   ├──2.png
│   │   │   └──...
│   │   └── train_your_name1.csv// 以DataFrame结构存储数据
│   └── train_your_name2        // 自定义训练集,结构同上
│       └── ...                    
│
└── test_data
    │
    ├── test_your_name1
    │   ├── test_images
    │   │   ├──1.jpg
    │   │   ├──2.jpg
    │   │   └──...
    │   ├── test_masks
    │   │   ├──1.png
    │   │   ├──2.png
    │   │   └──...
    │   └── test_your_name1.csv // 以DataFrame结构存储数据
    └── test_your_name2
        └── ...
```


###########V1.0.0 运行与调试

Note:
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
- You need to set **train_data** and **test_data** path in *demo.py*
More details in demo.py

1. train
python demo.py                              # set '--phase' as train

2. test
python demo.py                              # set '--phase' as test

3. 查看log文件
cat ./Out/log/_*_.log

4. 使用tensorboard,查看训练细节
tensorboard --logdir=/YourComputer/model_template/runs/_*_.local/
点击出现的网址即可查看






#### FAQs
Q1: You might get 'Error: Illegal postfix!' in generate_csv.py 
A1: './DS_store' might exist in your data files. And you need to check.
