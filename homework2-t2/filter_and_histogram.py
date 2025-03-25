import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def apply_mean_filter(image_path, output_path):
    """
    对图像应用3x3均值滤波器并保存结果
    """
    # 读取图像
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # 应用3x3均值滤波
    filtered_img = cv2.blur(img, (3, 3))
    
    # 保存滤波后的图像
    cv2.imwrite(output_path, filtered_img)
    
    return img, filtered_img

def calculate_histogram(image):
    """
    计算图像灰度直方图
    """
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    return hist

def plot_single_histogram(hist, title, save_path):
    """
    绘制并保存单个直方图
    """
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.xlabel('灰度值')
    plt.ylabel('像素数量')
    
    plt.plot(hist, color='blue')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def process_image(image_name):
    """
    处理单个图像：应用滤波、计算直方图、绘制对比图
    """
    # 确保results目录存在
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # 设置输入输出路径
    input_path = f"{image_name}.png"
    output_path = os.path.join(results_dir, f"avg_filter_{image_name}.png")
    before_hist_path = os.path.join(results_dir, f"before_histogram_{image_name}.png")
    after_hist_path = os.path.join(results_dir, f"after_histogram_{image_name}.png")
    
    print(f"处理图像: {input_path}")
    
    # 应用滤波
    original, filtered = apply_mean_filter(input_path, output_path)
    
    # 计算直方图
    original_hist = calculate_histogram(original)
    filtered_hist = calculate_histogram(filtered)
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    
    # 分别绘制并保存直方图
    plot_single_histogram(original_hist, 
                         f"{image_name}原始图像灰度直方图", 
                         before_hist_path)
    
    plot_single_histogram(filtered_hist, 
                         f"{image_name}均值滤波后灰度直方图", 
                         after_hist_path)
    
    # 计算直方图变化
    hist_diff = np.sum(np.abs(original_hist - filtered_hist))
    print(f"{image_name}直方图变化量: {hist_diff}")
    
    return original_hist, filtered_hist

# 主程序
if __name__ == "__main__":
    # 处理左侧图像
    process_image("bi_left")
    
    # 处理右侧图像
    process_image("bi_right")
    
    print("处理完成！所有结果已保存到results文件夹中。") 