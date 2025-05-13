import cv2
import numpy as np
import os

'''
- 功能：实现多种阈值分割方法
- 输入：灰度图片路径
- 输出：不同方法的分割结果
- 备注：包含自适应迭代阈值和OTSU方法
'''

def adaptive_iterative_threshold(image, max_iter=1000, tolerance=1):
    """
    自适应迭代阈值分割
    :param image: 输入图像
    :param max_iter: 最大迭代次数
    :param tolerance: 收敛阈值
    :return: 分割阈值
    """
    # 初始化阈值为图像的平均灰度值
    threshold = np.mean(image)
    
    for _ in range(max_iter):
        # 根据当前阈值将图像分为前景和背景
        foreground = image[image > threshold]
        background = image[image <= threshold]
        
        # 计算新的阈值
        new_threshold = (np.mean(foreground) + np.mean(background)) / 2
        
        # 检查是否收敛
        if abs(new_threshold - threshold) < tolerance:
            break
            
        threshold = new_threshold
    
    return threshold

def apply_threshold(image, threshold):
    """
    应用阈值进行分割
    :param image: 输入图像
    :param threshold: 阈值
    :return: 分割后的二值图像
    """
    return (image > threshold).astype(np.uint8) * 255

def main():
    # 读取灰度图片
    input_path = 'imgs/CT-image.jpg'
    # input_path = 'imgs/highlight15.png'
    # input_path = 'imgs/sugarcane_seedling.jpg'
    # input_path = 'imgs/gray_sugarcane_seedling.jpg'
    
    # 确保输入文件存在
    if not os.path.exists(input_path):
        print(f"错误：找不到输入文件 {input_path}")
        return
    
    # 获取图片文件名（不含扩展名）
    image_name = os.path.splitext(os.path.basename(input_path))[0]
    
    # 创建results文件夹（如果不存在）
    if not os.path.exists('results'):
        os.makedirs('results')
    
    # 为当前图片创建结果目录
    result_dir = os.path.join('results', image_name)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    # 读取图片
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    
    # 1. 自适应迭代阈值分割
    adaptive_threshold = adaptive_iterative_threshold(img)
    adaptive_result = apply_threshold(img, adaptive_threshold)
    adaptive_path = os.path.join(result_dir, 'adaptive_threshold.jpg')
    cv2.imwrite(adaptive_path, adaptive_result)
    print(f"自适应迭代阈值分割结果已保存到：{adaptive_path}")
    print(f"阈值为：{adaptive_threshold:.2f}")
    
    # 2. OTSU方法
    otsu_threshold, otsu_result = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu_path = os.path.join(result_dir, 'otsu_threshold.jpg')
    cv2.imwrite(otsu_path, otsu_result)
    print(f"OTSU分割结果已保存到：{otsu_path}")
    print(f"阈值为：{otsu_threshold}")
    
    # 3. 简单全局阈值（作为对比）
    _, global_result = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    global_path = os.path.join(result_dir, 'global_threshold.jpg')
    cv2.imwrite(global_path, global_result)
    print(f"全局阈值分割结果已保存到：{global_path}")

if __name__ == "__main__":
    main() 