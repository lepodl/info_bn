# 验证ntP大模型训练中的instability相关问题
## MLP在ntP下的transfer情况，lr敏感率等等
step1： 完成代码，先跑mlp的在宽度n下的敏感度。
**ntk deriving**
两个目标：
* 保证每层激活都是O(1)的
* 保证每一层关于当前参数的ntk贡献量是O(1)的
在MLP架构下，上述两点满足后，应该能直接推导出ntk parmetrization，即满足标准ntk标准结果。


step2: 纵向比较不同宽度下的最优参数迁移情况及敏感度情况。