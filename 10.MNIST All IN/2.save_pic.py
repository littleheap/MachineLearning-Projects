import os
import scipy.misc
from tensorflow.examples.tutorials.mnist import input_data

# 读取MNIST数据集
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 把原始图片保存在MNIST_data/raw/文件夹下
save_dir = 'MNIST_data/raw/'

# 不存在该文件夹就创建该文件夹
if os.path.exists(save_dir) is False:
    os.makedirs(save_dir)

# 保存前20张图片
for i in range(20):
    # mnist.train.images[i, :]就表示第i张图片，序号从0开始
    image_array = mnist.train.images[i, :]
    # 图片是一个784维的向量，重新还原为28x28维的图像。
    image_array = image_array.reshape(28, 28)
    # 保存文件的格式为mnist_train_0.jpg ~ mnist_train_19.jpg
    filename = save_dir + 'mnist_train_%d.jpg' % i
    # 将image_array保存为图片
    # 先用scipy.misc.toimage转换为图像，再调用save直接保存
    scipy.misc.toimage(image_array, cmin=0.0, cmax=1.0).save(filename)

print('Please check: %s ' % save_dir)  # Please check: MNIST_data/raw/
