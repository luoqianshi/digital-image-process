import numpy as np
from PIL import Image

def generate_half_image(size=(1024, 1024)):
    """
    生成左白右黑的二值图像
    """
    # 创建白色背景
    img = np.ones(size, dtype=np.uint8) * 255
    
    # 右半部分设为黑色
    half_width = size[1] // 2
    img[:, half_width:] = 0
    
    return img

def generate_checkerboard(size=(1024, 1024), block_size=128):
    """
    生成棋盘格二值图像
    """
    # 计算行列数
    rows, cols = size[0] // block_size, size[1] // block_size
    
    # 创建棋盘格图案
    img = np.ones(size, dtype=np.uint8) * 255
    
    for i in range(rows):
        for j in range(cols):
            if (i + j) % 2 == 1:  # 修改为1，确保左上角(0,0)为白色
                row_start, row_end = i * block_size, (i + 1) * block_size
                col_start, col_end = j * block_size, (j + 1) * block_size
                img[row_start:row_end, col_start:col_end] = 0
    
    return img

# 生成左白右黑图像
half_img = generate_half_image()
half_img_pil = Image.fromarray(half_img)
half_img_pil.save('bi_left.png')

# 生成棋盘格图像
checkerboard_img = generate_checkerboard()
checkerboard_img_pil = Image.fromarray(checkerboard_img)
checkerboard_img_pil.save('bi_right.png')

print("图像已成功生成！")
