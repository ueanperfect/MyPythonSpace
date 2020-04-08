import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tensorflow.keras import datasets, layers,Sequential,losses,optimizers

def read_image(image_number,path):
    image_value = [0 for i in range(image_number)]
    file_path = [0 for i in range(image_number)]
    for i in range(114):
        file_path[i]=path.format(str(i+1))
        image_value[i] =plt.imread(file_path[i])
    return image_value

path1 = '/Users/faguangnanhai/Desktop/k/0{}.bmp'

k = read_image(114,path1)

y = np.random.normal(size=[1231,1792,3])

plt.imshow()
plt.show()

