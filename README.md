# XDULoong_IESD_Contest
Team XDULoong's Solution Repository for 2024 IESD Contest 
## 目录结构
Repo中提供了 `pytorch` 的模型设计及训练代码。`pytorch` 目录下为使用pytorch搭建的网络原型、模型文件、网络权重文件以及训练代码
```shell
pytorch
├─first # 第一次提交作品训练与模型代码
│  │  help_code_demo.py
│  │  train.py           # 训练代码
│  │
│  ├─models
│  │      senet.py       # 模型代码
│  │
│  ├─saved_models_senet
│  │      senet_best_0.958.pth  # 导出模型
│  │
│  └─saved_senet_sd
│          senet_sd_0.958_light.pth # 导出模型权重
│
└─second # 第二次提交作品训练与模型代码
    │  training_save_deep_models.py # 训练代码
    │
    ├─models
    │      model_1.py # 模型代码
    │
    └─saved_models
            ECG_net_state_dict.pkl # 导出模型权重
```
