import tensorflow as tf

x = tf.constant(2,shape=[2,2])

tf.InteractiveSession()

print(x)

x.eval()

# 定义矩阵
tf.zeros([2,3],tf.int32)
