import tensorflow as tf

#定义图
a = tf.add(2.0,3.0)

#获取session
sess = tf.Session()

#输出图到当前路径下到train目录
writer = tf.summary.FileWriter("train", sess.graph)

#然后，加载模型，初始化所有变量
sess.run(a)
