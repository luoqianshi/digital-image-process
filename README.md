# 数字图像处理实验项目

本项目包含一系列数字图像处理实验，主要涵盖图像去噪、图像增强和图像分析等主题。每个实验位于单独的文件夹中，拥有各自的实现和说明文档。

## 项目结构

```
digital-image-process/
│
├── homework2-t1/     # 实验一：不同滤波方法的图像去噪对比分析
├── homework2-t2/     # 实验二：二值图像生成与均值滤波处理
├── homework2-t3/     # 实验三：多种图像增强与去噪方法实现
└── README.md        # 项目主说明文档
```

## 实验内容

### 实验一：不同滤波方法的图像去噪对比分析 (homework2-t1)

该实验比较了不同滤波方法对噪声图像的去噪效果，并分析了模板大小对处理速度的影响以及不同方法之间的时间效率。

**主要内容：**
- 实现均值滤波、中值滤波、高斯滤波和双边滤波四种方法
- 使用MSE、PSNR和SSIM三种指标评估去噪效果
- 分析不同滤波方法的性能和效果
- 可视化处理结果和性能指标对比

**关键文件：**
- `noise_reduction_comparison.py`：主程序文件
- `/noise images`：噪声图像目录
- `/results`：处理结果输出目录

### 实验二：二值图像生成与均值滤波处理 (homework2-t2)

该实验包含二值图像生成和均值滤波处理两部分，并分析滤波前后的直方图变化。

**主要内容：**
- 生成左白右黑的二值图像和黑白相间的棋盘格图像
- 对图像进行3x3均值滤波处理
- 计算并分析滤波前后的灰度直方图变化

**关键文件：**
- `image_gen.py`：二值图像生成脚本
- `filter_and_histogram.py`：均值滤波和直方图处理脚本
- `/results`：处理结果存储目录

### 实验三：多种图像增强与去噪方法实现 (homework2-t3)

该实验实现了多种图像增强和去噪方法，用于处理含噪声的图像。

**主要内容：**
- 实现椒盐噪声和高斯噪声的添加
- 实现多种边缘检测和增强方法（Roberts、Sobel、Laplacian等）
- 实现多种去噪方法（形态学滤波、小波变换、K近邻滤波、自适应中值滤波）
- 调整图像亮度和对比度
- 比较不同去噪和增强方法的效果

**关键文件：**
- `add_noise.py`：添加噪声的实现
- `enhance.py`：基础图像增强方法
- `other_enhance.py`：其它增强与去噪方法
- `/img`：原始图像目录
- `/denoise_imgs`：去噪结果输出目录

## 使用说明

每个实验都有其独立的依赖和使用方法，请参考各个实验目录下的README.md文件获取详细信息。

### 共同依赖库

大多数实验需要以下依赖库：
- NumPy
- OpenCV (cv2)
- Matplotlib
- SciPy

对于特定实验，可能需要额外的库，如：
- scikit-image：用于计算SSIM指标（实验一）
- PyWavelets：用于小波变换去噪（实验三）

### 安装依赖

```bash
pip install numpy opencv-python matplotlib scipy scikit-image pywavelets
```

## 注意事项

1. 请确保按照各实验的README中的步骤顺序运行脚本
2. 某些实验可能需要先生成中间结果，然后再进行后续处理
3. 处理大图像时，某些算法可能需要较长时间运行
4. 所有实验的结果都会保存在各自的输出目录中

## 作者

骆谦实

## 许可

本项目用于教育目的，遵循[MIT许可证](https://opensource.org/licenses/MIT)。 
