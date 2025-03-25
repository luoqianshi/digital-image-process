# 图像增强与去噪实验

本项目实现了多种图像增强和去噪方法，用于处理含噪声的图像。主要包含以下功能模块：

## 项目结构

```
homework2-t3/
├── img/                    # 原始图像目录
├── denoise_imgs/          # 去噪结果输出目录
├── add_noise.py           # 添加噪声的实现
├── enhance.py             # 基础图像增强方法
├── other_enhance.py       # 其他增强方法
├── ref_enhance.py         # 参考增强与去噪方法
└── README.md              # 项目说明文档
```

## 功能模块说明

### 1. 添加噪声 (add_noise.py)
- 实现了向图像添加不同类型的噪声
- 支持添加椒盐噪声
- 支持添加高斯随机噪声
- 输出：`noise_lena.png`

### 2. 基础图像增强 (enhance.py)
- 实现了多种边缘检测和增强方法
- 中值滤波去噪
- Roberts边缘检测
- Sobel边缘检测
- Laplacian边缘增强
- 输出多个处理阶段的结果：
  - `lena_1.png`: 中值滤波结果
  - `lena_2_roberts.png`: Roberts边缘检测
  - `lena_2_sobel.png`: Sobel边缘检测
  - `lena_2_laplacian.png`: Laplacian边缘增强
  - `lena_3_*.png`: 各种方法的最终增强结果
  - `lena_4_*.png`: 亮度调整后的结果

### 3. 其它增强方法 (other_enhance.py)
实现了四种不同的图像去噪方法：
1. 形态学滤波
   - 使用椭圆形结构元素
   - 结合开运算和闭运算
   - 保护图像细节

2. 小波变换去噪
   - 支持彩色图像处理
   - 使用db4小波基函数
   - 多层次小波分解
   - 自适应阈值处理

3. K近邻滤波
   - 自适应窗口处理
   - 加权平均策略
   - 边缘保护

4. 自适应中值滤波
   - 动态窗口大小
   - 结合双边滤波
   - 更好的边缘保护效果

输出结果保存在`denoise_imgs`目录下：
- `lena_morph.png`: 形态学滤波结果
- `lena_wavelet.png`: 小波变换去噪结果
- `lena_knn.png`: K近邻滤波结果
- `lena_adaptive_median.png`: 自适应中值滤波结果

## 使用说明

1. 添加噪声：
```python
python add_noise.py
```

2. 基础图像增强：
```python
python enhance.py
```

3. 参考增强方法：
```python
python ref_enhance.py
```

## 依赖库
- numpy
- opencv-python
- pywavelets (PyWavelets)
- scipy

## 注意事项
1. 运行代码前请确保已安装所有依赖库
2. 确保输入图像存在于正确的路径
3. 建议先运行add_noise.py生成带噪声的图像
4. 不同的增强方法可能需要根据具体图像调整参数

## 结果比较
各种方法的优缺点：
- 形态学滤波：适合去除椒盐噪声，但可能会轻微模糊边缘
- 小波变换：较好地保持了图像细节，对高斯噪声效果好
- K近邻滤波：边缘保护效果好，但计算量较大
- 自适应中值滤波：综合效果较好，特别是对混合噪声 