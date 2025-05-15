import cv2
import numpy as np
import os

'''
基于边缘的图像分割方法实现
包含以下算子：
1. Roberts算子
2. Prewitt算子
3. Sobel算子
4. Laplacian算子
5. LOG算子
6. Canny算子
'''

def safe_normalize(edge):
    """
    安全地归一化边缘检测结果
    :param edge: 边缘检测结果
    :return: 归一化后的结果
    """
    # 处理无效值
    edge = np.nan_to_num(edge, nan=0.0, posinf=255.0, neginf=0.0)
    
    # 确保数据类型正确
    edge = edge.astype(np.float32)
    
    # 获取有效值的范围
    min_val = np.min(edge)
    max_val = np.max(edge)
    
    # 如果所有值都相同，直接返回
    if min_val == max_val:
        return np.zeros_like(edge, dtype=np.uint8)
    
    # 归一化到0-255
    edge = ((edge - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    return edge

def roberts_edge_detection(image):
    """
    Roberts算子边缘检测
    :param image: 输入图像
    :return: 边缘检测结果
    """
    # Roberts算子
    roberts_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
    roberts_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)
    
    # 计算x和y方向的梯度
    grad_x = cv2.filter2D(image, -1, roberts_x)
    grad_y = cv2.filter2D(image, -1, roberts_y)
    
    # 计算梯度幅值
    edge = np.sqrt(np.square(grad_x) + np.square(grad_y))
    
    # 安全归一化
    return safe_normalize(edge)

def prewitt_edge_detection(image):
    """
    Prewitt算子边缘检测
    :param image: 输入图像
    :return: 边缘检测结果
    """
    # Prewitt算子
    prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
    prewitt_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
    
    # 计算x和y方向的梯度
    grad_x = cv2.filter2D(image, -1, prewitt_x)
    grad_y = cv2.filter2D(image, -1, prewitt_y)
    
    # 计算梯度幅值
    edge = np.sqrt(np.square(grad_x) + np.square(grad_y))
    
    # 安全归一化
    return safe_normalize(edge)

def sobel_edge_detection(image):
    """
    Sobel算子边缘检测
    :param image: 输入图像
    :return: 边缘检测结果
    """
    # 使用OpenCV的Sobel函数
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    
    # 计算梯度幅值
    edge = np.sqrt(np.square(grad_x) + np.square(grad_y))
    
    # 安全归一化
    return safe_normalize(edge)

def laplacian_edge_detection(image):
    """
    Laplacian算子边缘检测
    :param image: 输入图像
    :return: 边缘检测结果
    """
    # 使用OpenCV的Laplacian函数
    edge = cv2.Laplacian(image, cv2.CV_64F)
    
    # 取绝对值
    edge = np.absolute(edge)
    
    # 安全归一化
    return safe_normalize(edge)

def log_edge_detection(image):
    """
    LOG（Laplacian of Gaussian）算子边缘检测
    :param image: 输入图像
    :return: 边缘检测结果
    """
    # 先进行高斯模糊
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    
    # 再进行Laplacian边缘检测
    edge = cv2.Laplacian(blurred, cv2.CV_64F)
    
    # 取绝对值
    edge = np.absolute(edge)
    
    # 安全归一化
    return safe_normalize(edge)

def canny_edge_detection(image):
    """
    Canny算子边缘检测
    :param image: 输入图像
    :return: 边缘检测结果
    """
    # 使用OpenCV的Canny函数
    edge = cv2.Canny(image, 100, 200)
    return edge

def process_image(image_path):
    """
    处理单张图片，应用所有边缘检测方法
    :param image_path: 输入图片路径
    """
    # 读取图片并转换为灰度图
    image = cv2.imread(image_path)
    if image is None:
        print(f"错误：无法读取图片 {image_path}")
        return
    
    # 转换为灰度图
    if len(image.shape) == 3:  # 如果是彩色图像
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 获取图片文件名（不含扩展名）
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # 创建结果保存路径
    result_dir = os.path.join('results', 'edge_based_segment')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    # 应用各种边缘检测方法
    methods = {
        'roberts': roberts_edge_detection,
        'prewitt': prewitt_edge_detection,
        'sobel': sobel_edge_detection,
        'laplacian': laplacian_edge_detection,
        'log': log_edge_detection,
        'canny': canny_edge_detection
    }
    
    # 处理并保存结果
    for method_name, method_func in methods.items():
        try:
            result = method_func(image)
            output_path = os.path.join(result_dir, f"{method_name}_{image_name}.jpg")
            cv2.imwrite(output_path, result)
            print(f"{method_name}边缘检测结果已保存到：{output_path}")
        except Exception as e:
            print(f"处理{method_name}方法时出错：{str(e)}")

def main():
    # 获取imgs目录下的所有图片
    img_dir = 'imgs'
    if not os.path.exists(img_dir):
        print(f"错误：找不到图片目录 {img_dir}")
        return
    
    # 处理所有图片
    for filename in os.listdir(img_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
            image_path = os.path.join(img_dir, filename)
            print(f"\n处理图片：{filename}")
            process_image(image_path)

if __name__ == "__main__":
    main() 