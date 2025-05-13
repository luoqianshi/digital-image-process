import cv2
import matplotlib.pyplot as plt
import os

'''
- 功能：生成灰度图片的直方图
- 输入：灰度图片路径
- 输出：直方图图片
- 备注：使用matplotlib库绘制直方图，用于分析图片的灰度分布情况
'''

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 读取灰度图片
# input_path = 'imgs/highlight15.png'
# input_path = 'imgs/sugarcane_seedling.jpg'
# input_path = 'imgs/gray_sugarcane_seedling.jpg'
input_path = 'imgs/CT-image.jpg'

# 获取图片文件名（不含扩展名）
image_name = os.path.splitext(os.path.basename(input_path))[0]

# 确保输入文件存在
if not os.path.exists(input_path):
    print(f"错误：找不到输入文件 {input_path}")
    exit(1)

# 读取图片
img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

# 创建results文件夹（如果不存在）
if not os.path.exists('results'):
    os.makedirs('results')

# 创建图形
plt.figure(figsize=(10, 6))

# 绘制直方图
plt.hist(img.ravel(), bins=256, range=[0, 256], color='gray', alpha=0.7)
plt.title('灰度直方图')
plt.xlabel('灰度值')
plt.ylabel('像素数量')

# 设置x轴范围
plt.xlim([0, 256])

# 添加网格线
plt.grid(True, linestyle='--', alpha=0.7)

# 保存直方图
output_path = f'results/{image_name}_histogram.jpg'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"直方图已保存到：{output_path}") 