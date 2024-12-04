# 论述题：CUDA程序调优

MNIST（Modified National Institute of Standards and Technology）数据集是一个广泛使用的机器学习和计算机视觉数据集，主要用于手写数字的识别任务。它包含了大量手写数字的图像，每张图像是28×28像素的灰度图，表示一个数字（从0到9）。MNIST数据集分为训练集和测试集：

- 训练集：包含60,000张图像。
- 测试集：包含10,000张图像。

每张图像对应一个标签（0到9），表示该图像所表示的数字。

以下是一个CUDA程序，用于执行矩阵乘法和向量加法。程序中加载了MNIST数据集并初始化了一些矩阵和向量，然后在GPU上执行多次矩阵乘法和向量加法操作。您的任务是对该程序进行分析和调优，以提高其在GPU上的执行效率。

**你可以选择以下任务中的若干项进行论述和完成，你的总体完成度、单个题目的深入程度、实验报告完整性都会影响评分**

分析任务：

- 代码优化（30分）
- 内存优化 （30分）
- 并行度优化 （30分）
- 性能分析 （30分）

额外赋分：

- 使用Latex编写报告 (10分)
- 对每项任务都做了一定完成度的分析（10分）


**运行方法**

- 本程序执行需要约440MiB的显存，如无Nvidia显卡机器或您的显存不充足，您可以考虑使用Google Colab运行, 注意，它竟然是免费的
- 安装CUDA Tool Kit
- 在文件夹下执行 `make`
- 运行 ./mnist_cuda 查看结果

**报告要求**

- 您的报告应该是一份zip压缩包，至少包含一份pdf格式的报告和一份优化后的代码
- 您的报告内应该至少包含一份使用工具的性能分析报告，性能分析报告就像一张地图，会给我们指明优化方向
- 您的报告应该至少包括两张运行截图和两份代码，分别是原始程序和优化后程序的程序本身与运行输出截图

**评分提示**

- 考虑到不同同学的设备条件不一致，因此您无需追求绝对的高速，我们更希望看到您对这个题目本身的探索
- 你可以修改程序进行优化，但不应该破坏逻辑上的正确性，除破坏逻辑正确性外的改动都是允许的