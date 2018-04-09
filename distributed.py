import tensorflow as tf

# 指定运行操作的设备，/cpu:0其中的0表示设备号，TF不区分CPU的设备号，设置为0即可。
# 设置cpu运行时Tensor是储存在内存里的，而非显存里。gpu快的原因也是因此，
# 减少了数据从内存传输的时间，直接在显存中完成。
with tf.device('/cpu:0'):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    c = tf.matmul(a, b)

# 获取session,配置输出设备日志
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

#加载模型
print(sess.run(c))
