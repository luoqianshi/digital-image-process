import numpy as np
import cv2
import os

def add_salt_pepper_noise(image, salt_prob=0.01, pepper_prob=0.01):
    """
    给图像添加椒盐噪声
    :param image: 输入图像
    :param salt_prob: 盐噪声概率
    :param pepper_prob: 椒噪声概率
    :return: 添加椒盐噪声后的图像
    """
    noisy_image = np.copy(image)
    # 添加盐噪声（白点）
    salt_mask = np.random.random(image.shape[:2]) < salt_prob
    if len(image.shape) == 3:  # 彩色图像
        noisy_image[salt_mask, :] = 255
    else:  # 灰度图像
        noisy_image[salt_mask] = 255
    
    # 添加椒噪声（黑点）
    pepper_mask = np.random.random(image.shape[:2]) < pepper_prob
    if len(image.shape) == 3:  # 彩色图像
        noisy_image[pepper_mask, :] = 0
    else:  # 灰度图像
        noisy_image[pepper_mask] = 0
    
    return noisy_image

def add_gaussian_noise(image, mean=0, sigma=25):
    """
    给图像添加高斯随机噪声
    :param image: 输入图像
    :param mean: 噪声均值
    :param sigma: 噪声标准差
    :return: 添加高斯噪声后的图像
    """
    gaussian_noise = np.random.normal(mean, sigma, image.shape)
    noisy_image = np.clip(image + gaussian_noise, 0, 255).astype(np.uint8)
    return noisy_image

def main():
    # 读取图像（彩色）
    img_path = os.path.join('img', 'lena.png')
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    
    if img is None:
        print(f"无法读取图像: {img_path}")
        return
    
    # 添加椒盐噪声
    sp_noisy_img = add_salt_pepper_noise(img, salt_prob=0.02, pepper_prob=0.02)
    
    # 添加高斯噪声
    final_noisy_img = add_gaussian_noise(sp_noisy_img, mean=0, sigma=15)
    
    # 保存带噪声的图像
    output_path = './homework2-t3/noise_lena.png'
    cv2.imwrite(output_path, final_noisy_img)
    print(f"已生成带噪声的彩色图像: {output_path}")

if __name__ == "__main__":
    main() 