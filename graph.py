import tensorflow as tf

# 创建新图
g = tf.Graph()

# 将新图设置成默认图，并指定模型
with g.as_default():
    x = tf.add(3,5)

# 创建指定图的session
sess = tf.Session(graph=g)

# 执行模型
sess.run(x)
