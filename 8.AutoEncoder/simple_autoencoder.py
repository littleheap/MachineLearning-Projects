import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

# 读取数字识别数据集
mnist = input_data.read_data_sets('MNIST_data', validation_size=0)

# 显示一个目标数字
img = mnist.train.images[2]

plt.imshow(img.reshape((28, 28)), cmap='Greys_r')

plt.show()

# 模型构建
# 编码层神经元个数
encoding_dim = 32

# 输入图形尺寸
image_size = mnist.train.images.shape[1]

# 输入输出保持同级别大小
inputs_ = tf.placeholder(tf.float32, (None, image_size), name='inputs')
targets_ = tf.placeholder(tf.float32, (None, image_size), name='targets')

# 输出层
encoded = tf.layers.dense(inputs_, encoding_dim, activation=tf.nn.relu)

# 输出层logits
logits = tf.layers.dense(encoded, image_size, activation=None)

# Sigmoid获取输出
decoded = tf.nn.sigmoid(logits, name='output')

# 损失函数
loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=targets_, logits=logits)

# 代价计算
cost = tf.reduce_mean(loss)

# 优化器定义
opt = tf.train.AdamOptimizer(0.001).minimize(cost)

# 训练
sess = tf.Session()

epochs = 20
batch_size = 200
sess.run(tf.global_variables_initializer())
for e in range(epochs):
    for ii in range(mnist.train.num_examples // batch_size):
        batch = mnist.train.next_batch(batch_size)
        feed = {inputs_: batch[0], targets_: batch[0]}
        batch_cost, _ = sess.run([cost, opt], feed_dict=feed)

        print("Epoch: {}/{}...".format(e + 1, epochs),
              "Training loss: {:.4f}".format(batch_cost))

'''
    ......
    ......
    ......
    Epoch: 20/20... Training loss: 0.0905
    Epoch: 20/20... Training loss: 0.0940
    Epoch: 20/20... Training loss: 0.0926
    Epoch: 20/20... Training loss: 0.0941
    Epoch: 20/20... Training loss: 0.0978
    Epoch: 20/20... Training loss: 0.0935
    Epoch: 20/20... Training loss: 0.0938
    Epoch: 20/20... Training loss: 0.0941
    Epoch: 20/20... Training loss: 0.0965
    Epoch: 20/20... Training loss: 0.0960
    Epoch: 20/20... Training loss: 0.0923
    Epoch: 20/20... Training loss: 0.0935
    Epoch: 20/20... Training loss: 0.0911
'''


# 测试
fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(20, 4))
in_imgs = mnist.test.images[:10]
reconstructed, compressed = sess.run([decoded, encoded], feed_dict={inputs_: in_imgs})

for images, row in zip([in_imgs, reconstructed], axes):
    for img, ax in zip(images, row):
        ax.imshow(img.reshape((28, 28)), cmap='Greys_r')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

fig.tight_layout(pad=0.1)
plt.show()

sess.close()
