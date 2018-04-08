import tensorflow as tf
import numpy as np

#定义scopescope
with tf.name_scope('sum') as scope:
     a = tf.add(2.0,3.0)

#获取session
sess = tf.Session()
writer = tf.summary.FileWriter("train", sess.graph)
#添加用于初始化变量的节点
init = tf.global_variables_initializer()
#然后，加载模型，初始化所有变量
sess.run(init)
