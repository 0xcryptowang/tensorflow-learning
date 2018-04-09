import tensorflow as tf

# 获取默认graph
g1 = tf.get_default_graph()

# 创建一个新graph
g2 = tf.Graph()

# 指定图1的模型
with g1.as_default():
    a = tf.constant(3)

# 指定图2的模型
with g2.as_default():
    b = tf.constant(5)


# 获取graph1的session，并输出设备日志
with tf.Session(graph=g1,config=tf.ConfigProto(log_device_placement=True)) as sess1:
    # 定义模型
    x1 = tf.add(3,5)
    # 输出图到当前路径下到train目录
    writer1 = tf.summary.FileWriter("train", sess1.graph)
    # 加载模型
    sess1.run(x1)

    # 关闭写入
    writer1.close()

# 获取graph2的session，并输出设备日志
with tf.Session(graph=g2,config=tf.ConfigProto(log_device_placement=True)) as sess2:
    # 定义模型
    x2 = tf.multiply(4,6)
    # 输出图到当前路径下到train目录
    writer2 = tf.summary.FileWriter("train", sess2.graph)
    # 加载模型
    sess2.run(x2)

    # 关闭写入
    writer2.close()
