# XDULoong_IESD_Contest
Team XDULoong's Solution Repository for 2024 IESD Contest 
## 目录结构
Repo中提供了 `pytorch` 的模型设计及训练代码。`pytorch` 目录下为使用pytorch搭建的网络原型、模型文件、网络权重文件以及训练代码
```shell
pytorch
│  help_code_demo.py  # 读取标签文件, 实现自定义数据集
│  train.py # 模型训练
│
├─ models
│      senet.py # 网络原型代码
│
├─ saved_models_senet
│      senet_best_0.958.pth # 导出的模型文件，包括网络结构信息
│
└─ saved_senet_sd
        senet_sd_0.958_light.pth # 导出的权重文件，state_dict
```
