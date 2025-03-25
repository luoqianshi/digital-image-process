import cv2
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def apply_mean_filter(image, kernel_size):
    """应用均值滤波"""
    start_time = time.time()
    result = cv2.blur(image, (kernel_size, kernel_size))
    end_time = time.time()
    return result, end_time - start_time

def apply_median_filter(image, kernel_size):
    """应用中值滤波"""
    start_time = time.time()
    result = cv2.medianBlur(image, kernel_size)
    end_time = time.time()
    return result, end_time - start_time

def apply_gaussian_filter(image, kernel_size, sigma=0):
    """应用高斯滤波"""
    start_time = time.time()
    result = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    end_time = time.time()
    return result, end_time - start_time

def apply_bilateral_filter(image, d, sigma_color, sigma_space):
    """应用双边滤波"""
    start_time = time.time()
    result = cv2.bilateralFilter(image, d, sigma_color, sigma_space)
    end_time = time.time()
    return result, end_time - start_time

def calculate_metrics(original, processed):
    """计算评价指标"""
    mse_value = mse(original, processed)
    psnr_value = psnr(original, processed)
    ssim_value = ssim(original, processed, multichannel=True if len(original.shape) > 2 else False)
    
    return {
        'MSE': mse_value,
        'PSNR': psnr_value,
        'SSIM': ssim_value
    }

def process_image(image_path, save_dir):
    """处理单张图像并分析"""
    # 创建保存结果的目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return
    
    # 转换为灰度图像进行处理（如果是彩色图像）
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image
    
    # 图像文件名（不含路径和扩展名）
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # 用于存储结果的字典
    results = {
        'kernel_size': [],
        'mean_time': [],
        'median_time': [],
        'gaussian_time': [],
        'bilateral_time': [],
        'metrics': {}
    }
    
    # 测试不同的模板大小
    kernel_sizes = [3, 5, 7, 9, 11]
    
    # 保存原始图像
    original_save_path = os.path.join(save_dir, f"{image_name}_original.png")
    cv2.imwrite(original_save_path, gray_image)
    
    for kernel_size in kernel_sizes:
        print(f"处理内核大小: {kernel_size}x{kernel_size}")
        results['kernel_size'].append(kernel_size)
        
        # 应用不同的滤波方法
        mean_result, mean_time = apply_mean_filter(gray_image, kernel_size)
        median_result, median_time = apply_median_filter(gray_image, kernel_size)
        gaussian_result, gaussian_time = apply_gaussian_filter(gray_image, kernel_size)
        bilateral_result, bilateral_time = apply_bilateral_filter(gray_image, kernel_size, 75, 75)
        
        results['mean_time'].append(mean_time)
        results['median_time'].append(median_time)
        results['gaussian_time'].append(gaussian_time)
        results['bilateral_time'].append(bilateral_time)
        
        # 计算评价指标
        mean_metrics = calculate_metrics(gray_image, mean_result)
        median_metrics = calculate_metrics(gray_image, median_result)
        gaussian_metrics = calculate_metrics(gray_image, gaussian_result)
        bilateral_metrics = calculate_metrics(gray_image, bilateral_result)
        
        # 保存指标结果
        results['metrics'][kernel_size] = {
            'mean': mean_metrics,
            'median': median_metrics,
            'gaussian': gaussian_metrics,
            'bilateral': bilateral_metrics
        }
        
        # 保存处理后的图像
        if kernel_size == 5:  # 只保存中间大小的内核结果，以免生成太多图像
            cv2.imwrite(os.path.join(save_dir, f"{image_name}_mean_{kernel_size}.png"), mean_result)
            cv2.imwrite(os.path.join(save_dir, f"{image_name}_median_{kernel_size}.png"), median_result)
            cv2.imwrite(os.path.join(save_dir, f"{image_name}_gaussian_{kernel_size}.png"), gaussian_result)
            cv2.imwrite(os.path.join(save_dir, f"{image_name}_bilateral_{kernel_size}.png"), bilateral_result)
    
    return results

def plot_time_comparison(results, image_name, save_dir):
    """绘制时间效率对比图"""
    plt.figure(figsize=(12, 6))
    
    plt.plot(results['kernel_size'], results['mean_time'], 'o-', label='均值滤波')
    plt.plot(results['kernel_size'], results['median_time'], 's-', label='中值滤波')
    plt.plot(results['kernel_size'], results['gaussian_time'], '^-', label='高斯滤波')
    plt.plot(results['kernel_size'], results['bilateral_time'], 'v-', label='双边滤波')
    
    plt.xlabel('内核大小')
    plt.ylabel('处理时间 (秒)')
    plt.title(f'{image_name} - 不同滤波方法的时间效率对比')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(os.path.join(save_dir, f"{image_name}_time_comparison.png"))
    plt.close()

def plot_metrics_comparison(results, image_name, save_dir):
    """绘制评价指标对比图"""
    kernel_sizes = results['kernel_size']
    metrics = ['MSE', 'PSNR', 'SSIM']
    methods = ['mean', 'median', 'gaussian', 'bilateral']
    method_labels = ['均值滤波', '中值滤波', '高斯滤波', '双边滤波']
    
    for metric in metrics:
        plt.figure(figsize=(12, 6))
        
        for i, method in enumerate(methods):
            values = []
            for kernel_size in kernel_sizes:
                values.append(results['metrics'][kernel_size][method][metric])
            
            plt.plot(kernel_sizes, values, 'o-', label=method_labels[i])
        
        plt.xlabel('内核大小')
        plt.ylabel(metric)
        plt.title(f'{image_name} - {metric} 对比')
        plt.legend()
        plt.grid(True)
        
        plt.savefig(os.path.join(save_dir, f"{image_name}_{metric}_comparison.png"))
        plt.close()

def main():
    # 源图像目录和结果保存目录
    input_dir = "../noise images"
    output_dir = "./results"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 处理目录中的所有图像
    for filename in os.listdir(input_dir):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            image_path = os.path.join(input_dir, filename)
            image_name = os.path.splitext(filename)[0]
            
            print(f"正在处理图像: {filename}")
            
            # 创建该图像的结果目录
            image_output_dir = os.path.join(output_dir, image_name)
            
            # 处理图像并获取结果
            results = process_image(image_path, image_output_dir)
            
            if results:
                # 绘制时间效率对比图
                plot_time_comparison(results, image_name, image_output_dir)
                
                # 绘制评价指标对比图
                plot_metrics_comparison(results, image_name, image_output_dir)
                
                # 保存结果表格
                with open(os.path.join(image_output_dir, f"{image_name}_results.txt"), 'w') as f:
                    f.write(f"图像: {filename}\n\n")
                    
                    f.write("模板大小对处理速度的影响:\n")
                    f.write("=" * 80 + "\n")
                    f.write("内核大小 | 均值滤波 (秒) | 中值滤波 (秒) | 高斯滤波 (秒) | 双边滤波 (秒)\n")
                    f.write("-" * 80 + "\n")
                    
                    for i, kernel_size in enumerate(results['kernel_size']):
                        f.write(f"{kernel_size:^8} | {results['mean_time'][i]:^13.6f} | {results['median_time'][i]:^13.6f} | "
                                f"{results['gaussian_time'][i]:^13.6f} | {results['bilateral_time'][i]:^13.6f}\n")
                    
                    f.write("\n\n评价指标对比:\n")
                    
                    for kernel_size in results['kernel_size']:
                        f.write(f"\n内核大小: {kernel_size}x{kernel_size}\n")
                        f.write("=" * 80 + "\n")
                        f.write("滤波方法 | MSE | PSNR | SSIM\n")
                        f.write("-" * 80 + "\n")
                        
                        methods = ['mean', 'median', 'gaussian', 'bilateral']
                        method_labels = ['均值滤波', '中值滤波', '高斯滤波', '双边滤波']
                        
                        for method, label in zip(methods, method_labels):
                            metrics = results['metrics'][kernel_size][method]
                            f.write(f"{label:^8} | {metrics['MSE']:^10.2f} | {metrics['PSNR']:^10.2f} | "
                                    f"{metrics['SSIM']:^10.4f}\n")
    
    print("所有图像处理完成!")

if __name__ == "__main__":
    main() 