import tensorflow as tf

# 定义模型
a = tf.add(2.0,3.0)

# 获取session,配置输出设备日志
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

#加载模型
sess.run(a)
