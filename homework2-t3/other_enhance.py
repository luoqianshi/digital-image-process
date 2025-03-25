import numpy as np
import cv2
import pywt
from scipy import ndimage
from scipy.spatial import cKDTree
import os

def morphological_filter(image, kernel_size=5):
    """
    形态学滤波去噪
    :param image: 输入图像
    :param kernel_size: 结构元素大小
    :return: 去噪后的图像
    """
    # 创建椭圆形结构元素，对图像细节保护更好
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # 先进行闭运算去除暗噪声
    closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    
    # 再进行开运算去除亮噪声
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
    
    # 最后进行中值滤波平滑处理
    result = cv2.medianBlur(opening, 3)
    
    return result

def wavelet_denoising(image, wavelet='db4', level=2):
    """
    小波变换去噪（支持彩色图像）
    :param image: 输入图像
    :param wavelet: 小波基函数
    :param level: 分解层数
    :return: 去噪后的图像
    """
    # 对每个通道分别进行小波去噪
    if len(image.shape) == 3:
        result = np.zeros_like(image)
        for i in range(3):
            # 对每个通道进行小波变换
            coeffs = pywt.wavedec2(image[:,:,i], wavelet, level=level)
            
            # 自适应阈值处理
            threshold = np.std(coeffs[0]) * 0.4  # 降低阈值以保留更多细节
            coeffs = list(coeffs)
            
            # 对每个尺度进行不同程度的阈值处理
            for j in range(1, len(coeffs)):
                scale_threshold = threshold * (0.8 ** j)  # 高频部分使用更小的阈值
                coeffs[j] = tuple(pywt.threshold(c, scale_threshold, mode='soft') for c in coeffs[j])
            
            # 小波重构
            denoised = pywt.waverec2(coeffs, wavelet)
            
            # 确保尺寸一致
            if denoised.shape != image[:,:,i].shape:
                denoised = cv2.resize(denoised, (image.shape[1], image.shape[0]))
            
            result[:,:,i] = np.uint8(np.clip(denoised, 0, 255))
        
        return result
    else:
        return np.uint8(wavelet_denoising(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)))

def k_nearest_neighbor_filter(image, k=7, window_size=7):
    """
    K近邻滤波
    :param image: 输入图像
    :param k: 近邻数量
    :param window_size: 搜索窗口大小
    :return: 去噪后的图像
    """
    if len(image.shape) == 3:
        # 对彩色图像分别处理每个通道
        result = np.zeros_like(image)
        for i in range(3):
            result[:,:,i] = k_nearest_neighbor_filter(image[:,:,i], k, window_size)
        # 使用双边滤波进行最终的平滑处理
        result = cv2.bilateralFilter(result, 5, 75, 75)
        return result
    
    # 转换为灰度图
    gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 创建结果数组
    result = np.zeros_like(gray)
    height, width = gray.shape
    
    # 对每个像素进行处理
    for i in range(height):
        for j in range(width):
            # 获取搜索窗口
            start_i = max(0, i - window_size//2)
            end_i = min(height, i + window_size//2 + 1)
            start_j = max(0, j - window_size//2)
            end_j = min(width, j + window_size//2 + 1)
            
            # 获取窗口内的像素值
            window = gray[start_i:end_i, start_j:end_j].flatten()
            
            # 计算与中心像素的差异
            center_value = gray[i, j]
            differences = np.abs(window - center_value)
            
            # 获取k个最接近的值，使用加权平均
            k_nearest_indices = np.argsort(differences)[:k]
            k_nearest_values = window[k_nearest_indices]
            weights = 1 / (differences[k_nearest_indices] + 1e-6)
            result[i, j] = np.average(k_nearest_values, weights=weights)
    
    return np.uint8(result)

def adaptive_median_filter(image, min_window_size=3, max_window_size=11):
    """
    自适应中值滤波
    :param image: 输入图像
    :param min_window_size: 最小窗口大小
    :param max_window_size: 最大窗口大小
    :return: 去噪后的图像
    """
    if len(image.shape) == 3:
        # 对彩色图像分别处理每个通道
        result = np.zeros_like(image)
        for i in range(3):
            result[:,:,i] = adaptive_median_filter(image[:,:,i], min_window_size, max_window_size)
        # 使用双边滤波进行边缘保护
        result = cv2.bilateralFilter(result, 5, 75, 75)
        return result
    
    # 转换为灰度图
    gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 创建结果数组
    result = np.zeros_like(gray)
    height, width = gray.shape
    
    # 对每个像素进行处理
    for i in range(height):
        for j in range(width):
            window_size = min_window_size
            while window_size <= max_window_size:
                # 获取当前窗口
                start_i = max(0, i - window_size//2)
                end_i = min(height, i + window_size//2 + 1)
                start_j = max(0, j - window_size//2)
                end_j = min(width, j + window_size//2 + 1)
                
                window = gray[start_i:end_i, start_j:end_j].flatten()
                median = np.median(window)
                min_val = np.min(window)
                max_val = np.max(window)
                
                # 判断是否需要增大窗口
                if min_val < median < max_val:
                    if min_val < gray[i, j] < max_val:
                        result[i, j] = gray[i, j]  # 保留原始值
                    else:
                        result[i, j] = median  # 使用中值替换
                    break
                else:
                    if window_size < max_window_size:
                        window_size += 2
                    else:
                        result[i, j] = median
                        break
    
    return np.uint8(result)

def main():
    # 读取带噪声的图像
    img_path = './homework2-t3/noise_lena.png'
    img = cv2.imread(img_path)
    
    if img is None:
        print(f"无法读取图像: {img_path}")
        return
    
    # 创建denoise_imgs目录（如果不存在）
    if not os.path.exists('denoise_imgs'):
        os.makedirs('denoise_imgs')
    
    # 1. 形态学滤波去噪
    morph_result = morphological_filter(img)
    cv2.imwrite('denoise_imgs/lena_morph.png', morph_result)
    
    # 2. 小波变换去噪（现在支持彩色图像）
    wavelet_result = wavelet_denoising(img)
    cv2.imwrite('denoise_imgs/lena_wavelet.png', wavelet_result)
    
    # 3. K近邻滤波去噪
    knn_result = k_nearest_neighbor_filter(img)
    cv2.imwrite('denoise_imgs/lena_knn.png', knn_result)
    
    # 4. 自适应中值滤波去噪
    adaptive_median_result = adaptive_median_filter(img)
    cv2.imwrite('denoise_imgs/lena_adaptive_median.png', adaptive_median_result)
    
    print("图像去噪完成！")
    print("已生成以下文件：")
    print("1. denoise_imgs/lena_morph.png - 形态学滤波去噪结果")
    print("2. denoise_imgs/lena_wavelet.png - 小波变换去噪结果（彩色）")
    print("3. denoise_imgs/lena_knn.png - K近邻滤波去噪结果")
    print("4. denoise_imgs/lena_adaptive_median.png - 自适应中值滤波去噪结果")

if __name__ == "__main__":
    main() 