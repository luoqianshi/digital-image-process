# 数字图像处理作业二 - 图像生成与滤波处理

本项目包含两个主要部分：二值图像生成和均值滤波处理。

## 项目结构

```
homework2-t2/
│
├── image_gen.py          # 二值图像生成脚本
├── filter_and_histogram.py  # 均值滤波和直方图处理脚本
├── bi_left.png          # 生成的左侧二值图像
├── bi_right.png         # 生成的右侧二值图像
├── results/             # 处理结果存储目录
│   ├── avg_filter_*.png     # 均值滤波后的图像
│   ├── before_histogram_*.png  # 滤波前的直方图
│   └── after_histogram_*.png   # 滤波后的直方图
└── README.md            # 项目说明文档
```

## 功能说明

### 1. 二值图像生成 (image_gen.py)

- 生成左白右黑的二值图像 (bi_left.png)
- 生成黑白相间的棋盘格图像 (bi_right.png)
- 图像尺寸默认为256x256像素
- 棋盘格的方块大小默认为32x32像素

### 2. 图像滤波与直方图分析 (filter_and_histogram.py)

- 对输入图像进行3x3均值滤波处理
- 计算并保存滤波前后的灰度直方图
- 计算直方图变化量
- 所有处理结果保存在results目录下

## 使用方法

1. 生成二值图像：
```bash
python image_gen.py
```

2. 进行滤波处理和直方图分析：
```bash
python filter_and_histogram.py
```

## 依赖库

- OpenCV (cv2)
- NumPy
- Matplotlib

## 注意事项

- 确保系统安装了支持中文的字体（如SimHei）
- 运行滤波处理前需要先生成二值图像
- 所有处理结果会自动保存在results目录下 