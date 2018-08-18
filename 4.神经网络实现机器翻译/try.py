import tensorflow as tf

x = tf.constant([1, 1])
y = tf.constant([2, 2])
z = tf.constant([3, 3])

m = tf.stack([x, y, z], axis=0)

a = tf.constant([4, 4])
b = tf.constant([5, 5])
c = tf.constant([6, 6])

n = tf.stack([a, b, c], axis=0)

input_m = tf.strided_slice(m, [0, 0], [2, 2])

input_n = tf.strided_slice(n, [1, 0], [3, 3])

output = n = tf.stack([input_m, input_n], axis=0)

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    print('m:')
    print(input_m.eval())
    print(input_m.shape)
    print('n:')
    print(input_n.eval())
    print(input_n.shape)
    print('result:')
    print(output.eval())
    print(output.shape)
