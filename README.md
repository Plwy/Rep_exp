
### 重参数化验证实验

1.conv_bn 融合
将conv和bn层融合成一个卷积层
```
python conv_bn_fusion.py
```

2.分支相加
两个卷积分支相加, 使用一个卷积替代
```
python conv_test_add.py
```

3.卷积序列融合
1x1卷积接3x3卷积, 使用一个3x3卷积替代
```
python conv_fusion.py
```

4.分组卷积的融合
多个卷积结果concat, 使用一个卷积替代.方法:将多个卷积核权重在输出通道维度上进行拼接
```
python conv_cat.py
```

5.平均池化层转卷积
平均池化层是可以等价一个固定权重的卷积层
```
python avg_pool_tran.py
```
6.多尺度卷积融合
1x1卷积转为3x3卷积 .1x1 卷积核周围补0
```
python conv_tran.py
```
yoyo
