## 使用说明

1. 用 `download_data.py` 下载 k 线的原始数据，默认是保存在 `/tmp/stock` 里的，可以通过参数改
2. 用 `convert_to_tfrecords.py` 生成 TFRecords （包含了图像、标签、股票代码、买入日期、卖出日期）
3. 使用 `simple_main.py` 训练、验证

目前没有预测功能。

## `simple_main.py`

我自己写的入口程序，可以实现训练和预测，只能使用单 CPU/GPU，可用于模型测试。

## `adv_main.py`

从 CIFAR-10 的 Estimator 官方程序改的，有很多的高级功能（开发中）
