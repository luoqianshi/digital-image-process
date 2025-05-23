# 数字图像处理作业3

本目录包含了数字图像处理相关的Python脚本，主要用于图像灰度转换、直方图分析和阈值分割等操作。

## 文件结构

```
homework3-t1/
├── gray.py          # 图像灰度转换
├── analysis.py      # 灰度直方图分析
├── segment.py       # 多种阈值分割方法实现
├── imgs/            # 输入图像目录
└── results/         # 结果输出目录
```

## 脚本功能说明

### 1. gray.py
- **功能**：将彩色图像转换为灰度图像
- **输入**：彩色图像（默认：`imgs/sugarcane_seedling.jpg`）
- **输出**：灰度图像（保存为：`imgs/gray_sugarcane_seedling.jpg`）
- **实现方法**：使用OpenCV的`cvtColor`函数进行颜色空间转换

### 2. analysis.py
- **功能**：生成灰度图像的直方图
- **输入**：灰度图像（默认：`imgs/gray_sugarcane_seedling.jpg`）
- **输出**：直方图图像（保存为：`results/histogram.jpg`）
- **实现方法**：使用matplotlib绘制灰度直方图，展示像素值分布情况

### 3. segment.py
- **功能**：实现多种阈值分割方法
- **输入**：灰度图像
- **输出**：不同方法的分割结果（保存在以图像名命名的子目录中）

#### 阈值分割方法说明

##### 3.1 自适应迭代阈值分割（Adaptive Iterative Thresholding）
- **原理**：
  1. 初始化阈值为图像的平均灰度值
  2. 根据当前阈值将图像分为前景和背景
  3. 计算前景和背景的平均灰度值
  4. 将新的阈值设为前景和背景平均值的中间值
  5. 重复步骤2-4，直到阈值收敛或达到最大迭代次数
- **参数**：
  - `max_iter`：最大迭代次数（默认：1000）
  - `tolerance`：收敛阈值（默认：1）
- **输出**：`adaptive_threshold.jpg`

##### 3.2 OTSU方法（大津法）
- **原理**：
  1. 计算图像的灰度直方图
  2. 遍历所有可能的阈值（0-255）
  3. 对每个阈值计算类间方差
  4. 选择使类间方差最大的阈值作为最优阈值
- **特点**：
  - 自动计算最优阈值
  - 适用于双峰图像
  - 基于类间方差最大化
- **输出**：`otsu_threshold.jpg`

##### 3.3 简单全局阈值
- **原理**：
  - 使用固定阈值（默认：127）进行二值化
  - 像素值大于阈值的设为255（白色）
  - 像素值小于等于阈值的设为0（黑色）
- **特点**：
  - 实现简单
  - 适用于对比度较高的图像
- **输出**：`global_threshold.jpg`

## 使用说明

1. 确保已安装必要的Python库：
```bash
pip install opencv-python numpy matplotlib
```

2. 运行各个脚本：
```bash
python gray.py      # 灰度转换
python analysis.py  # 直方图分析
python segment.py   # 阈值分割
```

3. 查看结果：
- 灰度图像保存在`imgs/`目录
- 直方图和分割结果保存在`results/`目录
- 每个图像的分割结果保存在以其名称命名的子目录中

## 注意事项

1. 运行脚本前请确保输入图像存在于正确的目录
2. 可以根据需要修改输入图像路径
3. 分割结果会自动创建对应的子目录
4. 所有输出图像都使用JPG格式保存 