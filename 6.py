import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Dense, concatenate, Input, Concatenate
from keras.optimizers import Adam
from scipy.signal import convolve2d


img = np.asarray(Image.open('cat.png'))
pad_img = np.pad(img, ((50, 50), (50, 50), (0, 0)), mode='constant', constant_values=0)

kernel_z = np.zeros((3, 3))
kernel_o = np.ones((3, 3))
kernel_r = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

kernel_r = np.random.random((3, 3))
print(kernel_r)


def conv(image, kernel):

    height, width, channels = image.shape
    k_height, k_width = kernel.shape

    pad_h = k_height // 2
    pad_w = k_width // 2
    
    output = np.zeros_like(image)
    
    for i in range(pad_h, height - pad_h):
        for j in range(pad_w, width - pad_w):
            for c in range(channels):

                sum_val = 0
                for ki in range(k_height):
                    for kj in range(k_width):
                        sum_val += image[i + ki - pad_h, j + kj - pad_w, c] * kernel[ki, kj]
                output[i, j, c] = sum_val
    
    return output


def pool(image, kernel_size=2):

    height, width, channels = image.shape
    
    out_height = height // kernel_size
    out_width = width // kernel_size
    output = np.zeros((out_height, out_width, channels), dtype=image.dtype)

    for i in range(out_height):
        for j in range(out_width):
            for c in range(channels):

                block = image[i*kernel_size:(i+1)*kernel_size, j*kernel_size:(j+1)*kernel_size, c]

                output[i, j, c] = np.min(block)
    
    return output

res_z = conv(pad_img, kernel_z)
res_o = conv(pad_img, kernel_o)
res_r = conv(pad_img, kernel_r)
pooled_img = pool(img)

plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.title('Оригинальное изображение')
plt.imshow(pad_img)
plt.axis('off')

plt.subplot(2, 3, 2)
plt.title('Свёртка с нулями')
plt.imshow(res_z.astype(np.uint8))
plt.axis('off')

plt.subplot(2, 3, 3)
plt.title('Свёртка с единицами')
plt.imshow(res_o.astype(np.uint8))
plt.axis('off')

plt.subplot(2, 3, 4)
plt.title('Свертка произвольными')
plt.imshow(res_r.astype(np.uint8))
plt.axis('off')

plt.subplot(2, 3, 5)
plt.title('Пуллинг')
plt.imshow(pooled_img.astype(np.uint8))
plt.axis('off')

plt.tight_layout()
plt.show()


