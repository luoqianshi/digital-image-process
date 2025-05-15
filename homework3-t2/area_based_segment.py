import cv2
import numpy as np
import os
from collections import deque

'''
基于区域的图像分割方法实现
包含以下方法：
1. 简单区域生长法
2. 质心区域生长法
3. 分裂合并法
'''

def simple_region_growing(image, seed_points, threshold=20):
    """
    简单区域生长法
    :param image: 输入图像
    :param seed_points: 种子点列表，每个元素为(x, y)坐标
    :param threshold: 生长阈值
    :return: 分割结果
    """
    height, width = image.shape
    result = np.zeros_like(image)
    
    # 8邻域方向
    directions = [(-1, -1), (-1, 0), (-1, 1),
                 (0, -1),          (0, 1),
                 (1, -1),  (1, 0),  (1, 1)]
    
    # 对每个种子点进行区域生长
    for seed_x, seed_y in seed_points:
        if result[seed_y, seed_x] != 0:
            continue
            
        # 使用队列进行区域生长
        queue = deque([(seed_x, seed_y)])
        seed_value = image[seed_y, seed_x]
        result[seed_y, seed_x] = 255
        
        while queue:
            x, y = queue.popleft()
            
            # 检查8邻域
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                
                # 检查边界
                if (0 <= nx < width and 0 <= ny < height and 
                    result[ny, nx] == 0 and 
                    abs(int(image[ny, nx]) - int(seed_value)) <= threshold):
                    result[ny, nx] = 255
                    queue.append((nx, ny))
    
    return result

def centroid_region_growing(image, seed_points, threshold=30, max_region_size=None):
    """
    质心区域生长法
    :param image: 输入图像
    :param seed_points: 种子点列表，每个元素为(x, y)坐标
    :param threshold: 生长阈值，默认值降低到30
    :param max_region_size: 最大区域大小，None表示不限制
    :return: 分割结果
    """
    height, width = image.shape
    result = np.zeros_like(image)
    
    # 8邻域方向
    directions = [(-1, -1), (-1, 0), (-1, 1),
                 (0, -1),          (0, 1),
                 (1, -1),  (1, 0),  (1, 1)]
    
    # 对每个种子点进行区域生长
    for seed_x, seed_y in seed_points:
        if result[seed_y, seed_x] != 0:
            continue
            
        # 使用队列进行区域生长
        queue = deque([(seed_x, seed_y)])
        result[seed_y, seed_x] = 255
        
        # 使用累积和来优化质心计算，使用float64类型避免溢出
        sum_x = float(seed_x)
        sum_y = float(seed_y)
        sum_value = float(image[seed_y, seed_x])
        pixel_count = 1
        
        while queue:
            x, y = queue.popleft()
            
            # 计算当前区域的质心
            centroid_x = sum_x / pixel_count
            centroid_y = sum_y / pixel_count
            centroid_value = sum_value / pixel_count
            
            # 检查是否达到最大区域大小
            if max_region_size is not None and pixel_count >= max_region_size:
                continue
            
            # 检查8邻域
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                
                # 检查边界和是否已访问
                if (0 <= nx < width and 0 <= ny < height and result[ny, nx] == 0):
                    # 使用更宽松的生长条件
                    current_value = float(image[ny, nx])
                    if abs(current_value - centroid_value) <= threshold:
                        result[ny, nx] = 255
                        queue.append((nx, ny))
                        
                        # 更新累积和
                        sum_x += float(nx)
                        sum_y += float(ny)
                        sum_value += current_value
                        pixel_count += 1
    
    return result

def process_image(image_path):
    """
    处理单张图片，应用所有区域分割方法
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
    result_dir = os.path.join('results', 'area_based_segment')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    # 选择多个种子点
    height, width = image.shape
    seed_points = [
        (width//4, height//4),      # 左上
        (width//4, 3*height//4),    # 左下
        (3*width//4, height//4),    # 右上
        (3*width//4, 3*height//4),  # 右下
        (width//2, height//2)       # 中心
    ]
    
    # 计算最大区域大小（图像面积的20%）
    max_region_size = int(height * width * 0.2)
    
    # 应用各种区域分割方法
    methods = {
        'simple_growing': lambda img: simple_region_growing(img, seed_points),
        'centroid_growing': lambda img: centroid_region_growing(img, seed_points, threshold=20, max_region_size=max_region_size),
    }
    
    # 处理并保存结果
    for method_name, method_func in methods.items():
        try:
            result = method_func(image)
            output_path = os.path.join(result_dir, f"{method_name}_{image_name}.jpg")
            cv2.imwrite(output_path, result)
            print(f"{method_name}分割结果已保存到：{output_path}")
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