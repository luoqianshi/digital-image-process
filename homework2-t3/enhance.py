import numpy as np
import cv2
import os

def adjust_brightness_contrast(image, alpha=1.2, beta=30):
    """
    调整图像的亮度和对比度
    :param image: 输入图像
    :param alpha: 对比度调整因子（>1增加对比度，<1减少对比度）
    :param beta: 亮度调整因子（>0增加亮度，<0减少亮度）
    :return: 调整后的图像
    """
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted

def median_filter(image, kernel_size=3):
    """
    对图像进行中值滤波
    :param image: 输入图像
    :param kernel_size: 核大小
    :return: 中值滤波后的图像
    """
    return cv2.medianBlur(image, kernel_size)

def roberts_edge_detection(image):
    """
    使用Roberts算子进行边缘检测
    :param image: 输入图像
    :return: 边缘检测结果
    """
    # 转换为灰度图
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Roberts算子
    roberts_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
    roberts_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)
    
    # 计算x和y方向的梯度
    grad_x = cv2.filter2D(gray, -1, roberts_x)
    grad_y = cv2.filter2D(gray, -1, roberts_y)
    
    # 计算梯度幅值
    gradient = np.sqrt(grad_x**2 + grad_y**2)
    
    # 归一化到0-255
    gradient = np.uint8(gradient * 255 / np.max(gradient))
    
    return gradient

def sobel_edge_detection(image):
    """
    使用Sobel算子进行边缘检测
    :param image: 输入图像
    :return: 边缘检测结果
    """
    # 转换为灰度图
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Sobel算子
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # 计算梯度幅值
    gradient = np.sqrt(sobelx**2 + sobely**2)
    
    # 归一化到0-255
    gradient = np.uint8(gradient * 255 / np.max(gradient))
    
    return gradient

def laplacian_edge_enhancement(image):
    """
    使用Laplacian算子进行边缘增强
    :param image: 输入图像
    :return: 边缘增强后的图像
    """
    # 转换为灰度图
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Laplacian算子
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    
    # 归一化到0-255
    laplacian = np.uint8(np.absolute(laplacian) * 255 / np.max(np.absolute(laplacian)))
    
    return laplacian

def unsharp_masking(image, amount=1.5, radius=1):
    """
    使用Unsharp Masking进行图像锐化
    :param image: 输入图像
    :param amount: 锐化强度
    :param radius: 高斯核半径
    :return: 锐化后的图像
    """
    # 创建高斯模糊
    blurred = cv2.GaussianBlur(image, (0, 0), radius)
    
    # 计算锐化掩码
    sharpened = cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)
    
    return sharpened

def enhance_image(image, edge_weight=0.3):
    """
    图像增强主函数
    :param image: 输入图像
    :param edge_weight: 边缘信息的权重
    :return: 增强后的图像
    """
    # 1. 中值滤波
    median_img = median_filter(image)
    cv2.imwrite('lena_1.png', median_img)
    
    # 2. Roberts边缘检测
    roberts_edge = roberts_edge_detection(median_img)
    cv2.imwrite('lena_2_roberts.png', roberts_edge)
    
    # 3. Sobel边缘检测
    sobel_edge = sobel_edge_detection(median_img)
    cv2.imwrite('lena_2_sobel.png', sobel_edge)
    
    # 4. Laplacian边缘增强
    laplacian_edge = laplacian_edge_enhancement(median_img)
    cv2.imwrite('lena_2_laplacian.png', laplacian_edge)
    
    # 5. Unsharp Masking锐化
    sharpened = unsharp_masking(median_img)
    cv2.imwrite('lena_2_sharpened.png', sharpened)
    
    # 6. 生成各种边缘增强方法的最终结果
    if len(image.shape) == 3:
        # 将边缘图转换为3通道
        roberts_edge = cv2.cvtColor(roberts_edge, cv2.COLOR_GRAY2BGR)
        sobel_edge = cv2.cvtColor(sobel_edge, cv2.COLOR_GRAY2BGR)
        laplacian_edge = cv2.cvtColor(laplacian_edge, cv2.COLOR_GRAY2BGR)
    
    # Roberts边缘增强结果
    enhanced_roberts = cv2.addWeighted(median_img, 1-edge_weight, roberts_edge, edge_weight, 0)
    cv2.imwrite('lena_3_roberts.png', enhanced_roberts)
    
    # Sobel边缘增强结果
    enhanced_sobel = cv2.addWeighted(median_img, 1-edge_weight, sobel_edge, edge_weight, 0)
    cv2.imwrite('lena_3_sobel.png', enhanced_sobel)
    
    # Laplacian边缘增强结果
    enhanced_laplacian = cv2.addWeighted(median_img, 1-edge_weight, laplacian_edge, edge_weight, 0)
    cv2.imwrite('lena_3_laplacian.png', enhanced_laplacian)
    
    # 7. 调整亮度和对比度
    # Sobel边缘增强结果的亮度调整
    enhanced_sobel_bright = adjust_brightness_contrast(enhanced_sobel, alpha=1.1, beta=20)
    cv2.imwrite('lena_4_sobel.png', enhanced_sobel_bright)
    
    # Laplacian边缘增强结果的亮度调整
    enhanced_laplacian_bright = adjust_brightness_contrast(enhanced_laplacian, alpha=1.1, beta=20)
    cv2.imwrite('lena_4_laplacian.png', enhanced_laplacian_bright)
    
    return enhanced_sobel  # 默认返回Sobel增强结果

def main():
    # 读取带噪声的图像
    img_path = './homework2-t3/noise_lena.png'
    img = cv2.imread(img_path)
    
    if img is None:
        print(f"无法读取图像: {img_path}")
        return
    
    # 进行图像增强
    enhanced_img = enhance_image(img)
    print("图像增强完成！")
    print("已生成以下文件：")
    print("1. lena_1.png - 中值滤波结果")
    print("2. lena_2_roberts.png - Roberts边缘检测结果")
    print("3. lena_2_sobel.png - Sobel边缘检测结果")
    print("4. lena_2_laplacian.png - Laplacian边缘增强结果")
    print("5. lena_2_sharpened.png - Unsharp Masking锐化结果")
    print("6. lena_3_roberts.png - Roberts边缘增强最终结果")
    print("7. lena_3_sobel.png - Sobel边缘增强最终结果")
    print("8. lena_3_laplacian.png - Laplacian边缘增强最终结果")
    print("9. lena_3_sharpened.png - Unsharp Masking增强最终结果")
    print("10. lena_4_sobel.png - Sobel边缘增强亮度调整结果")
    print("11. lena_4_laplacian.png - Laplacian边缘增强亮度调整结果")

if __name__ == "__main__":
    main() 