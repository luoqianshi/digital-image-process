# 图像分割方法实现

本实验实现了两类图像分割方法：
1. 基于区域的图像分割
2. 基于边缘的图像分割

## 环境要求

- Python 3.x
- OpenCV
- NumPy

## 文件说明

- `area_based_segment.py`: 基于区域的图像分割方法实现
- `edge_based_segment.py`: 基于边缘的图像分割方法实现
- `imgs/`: 输入图像目录
- `results/`: 分割结果保存目录
  - `area_based_segment/`: 区域分割结果
  - `edge_based_segment/`: 边缘分割结果

## 基于区域的图像分割

### 1. 简单区域生长法 (simple_region_growing)

```python
def simple_region_growing(image, seed_points, threshold=20)
```

参数说明：
- `threshold`: 生长阈值，控制像素值差异的容忍度
  - 默认值：20
  - 建议范围：10-50
  - 调参经验：
    - 值越小，分割越精细，但可能导致过度分割
    - 值越大，分割越粗糙，但可能导致欠分割
    - 对于对比度较高的图像，建议使用较小的阈值（10-20）
    - 对于对比度较低的图像，建议使用较大的阈值（30-50）

### 2. 质心区域生长法 (centroid_region_growing)

```python
def centroid_region_growing(image, seed_points, threshold=20, max_region_size=None)
```

参数说明：
- `threshold`: 生长阈值，控制像素值与区域质心值的差异容忍度
  - 默认值：20
  - 建议范围：10-50
  - 调参经验：
    - 与简单区域生长法类似，但效果通常更好
    - 对于纹理丰富的图像，建议使用较小的阈值
    - 对于平滑区域，可以使用较大的阈值
- `max_region_size`: 最大区域大小限制
  - 默认值：图像面积的20%
  - 建议范围：10%-30%
  - 调参经验：
    - 值太小会导致区域生长不充分
    - 值太大会导致不同区域合并
    - 对于复杂图像，建议使用较小的值
    - 对于简单图像，可以使用较大的值

### 种子点选择

当前实现中使用了5个种子点：
- 左上角点 (width//4, height//4)
- 左下角点 (width//4, 3*height//4)
- 右上角点 (3*width//4, height//4)
- 右下角点 (3*width//4, 3*height//4)
- 中心点 (width//2, height//2)

调参经验：
- 种子点数量：
  - 太少：可能导致覆盖不完整
  - 太多：可能导致计算开销大
  - 建议：根据图像复杂度选择3-7个种子点
- 种子点位置：
  - 建议选择图像中具有代表性的区域
  - 避免选择噪声点或边缘点
  - 对于特定目标，可以手动选择种子点

## 基于边缘的图像分割

实现了以下边缘检测算子：

### 1. Roberts算子
- 特点：最简单的边缘检测算子之一
- 适用场景：对噪声敏感，适合检测明显的边缘
- 优点：计算简单，速度快
- 缺点：对噪声敏感，边缘检测效果一般

### 2. Prewitt算子
- 特点：3x3的卷积核，考虑了8邻域
- 适用场景：对噪声有一定的抑制能力
- 优点：比Roberts算子更稳定
- 缺点：对噪声仍然比较敏感

### 3. Sobel算子
- 特点：3x3的卷积核，考虑了像素距离的影响
- 适用场景：最常用的边缘检测算子之一
- 优点：对噪声有较好的抑制能力
- 缺点：可能会产生较粗的边缘

### 4. Laplacian算子
- 特点：二阶微分算子，对边缘更敏感
- 适用场景：需要精确定位边缘位置的场景
- 优点：对边缘的定位更准确
- 缺点：对噪声非常敏感

### 5. LOG算子（Laplacian of Gaussian）
- 特点：先进行高斯滤波，再进行Laplacian边缘检测
- 适用场景：需要抑制噪声的场景
- 优点：对噪声有很好的抑制能力
- 缺点：计算量较大

### 6. Canny算子
- 特点：多步骤的边缘检测算法
- 参数：
  - `threshold1`: 低阈值，默认100
  - `threshold2`: 高阈值，默认200
- 适用场景：需要高质量边缘检测结果的场景
- 优点：边缘检测效果最好
- 缺点：计算量最大

## 使用示例

```python
# 处理单张图片
python area_based_segment.py  # 运行区域分割
python edge_based_segment.py  # 运行边缘分割
```

## 结果说明

程序会在 `results/` 目录下生成以下文件：

区域分割结果：
- `area_based_segment/simple_growing_[image_name].jpg`
- `area_based_segment/centroid_growing_[image_name].jpg`

边缘分割结果：
- `edge_based_segment/roberts_[image_name].jpg`
- `edge_based_segment/prewitt_[image_name].jpg`
- `edge_based_segment/sobel_[image_name].jpg`
- `edge_based_segment/laplacian_[image_name].jpg`
- `edge_based_segment/log_[image_name].jpg`
- `edge_based_segment/canny_[image_name].jpg`

## 注意事项

1. 输入图像会自动转换为灰度图进行处理
2. 确保输入图像清晰，噪声较少
3. 对于复杂图像，可能需要调整参数以获得更好的分割效果
4. 建议先用小图像测试参数效果，再处理大图像
5. 边缘检测方法的选择：
   - 如果图像噪声较小，可以使用Roberts、Prewitt或Sobel算子
   - 如果图像噪声较大，建议使用LOG或Canny算子
   - 如果需要精确定位边缘，可以使用Laplacian算子
6. 区域分割方法的选择：
   - 如果图像对比度较高，可以使用简单区域生长法
   - 如果图像对比度较低，建议使用质心区域生长法
   - 对于复杂图像，可能需要结合多种方法 