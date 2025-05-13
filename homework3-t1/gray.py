import cv2
import os

'''
- 功能：将彩色图片转换为灰度图片
- 输入：原始图片路径
- 输出：灰度图片路径
- 备注：使用OpenCV库进行图片处理，进行实验数据的初步准备和预处理。
'''

# 读取原始图片
input_path = 'imgs/sugarcane_seedling.jpg'
output_path = 'imgs/gray_sugarcane_seedling.jpg'


# 确保输入文件存在
if not os.path.exists(input_path):
    print(f"错误：找不到输入文件 {input_path}")
    exit(1)

# 读取图片
img = cv2.imread(input_path)

# 将图片转换为灰度图
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 保存灰度图片
cv2.imwrite(output_path, gray_img)

print(f"灰度图片已保存到：{output_path}") 