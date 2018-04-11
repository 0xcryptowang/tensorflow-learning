import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time

# 定义模型参数
# 学习率
learning_rate = 0.01
# 批量大小
batch_size = 128
#
n_epochs = 30

# 读取数据集并进行one hot编码
# TFLearn（tf的一个简单接口）有一个让你可以从Yan Lecun个人网站加载MNIST数据集的脚本，并且把它分为训练集，验证集和测试集
mnist = input_data.read_data_sets("data/mnist", one_hot=True)

# 定义批量占位符
# 每个图片28*28像素，拉伸为1维张量长度为784（28*28 = 784）
# 每个图像有10个类，对应于数字0 - 9
X = tf.placeholder(tf.float32, [batch_size, 784], name="image")
Y = tf.placeholder(tf.float32, [batch_size, 10], name="label")

# 定义变量
w = tf.Variable(tf.random_normal(shape=[784, 10], stddev=0.01), name='weights')
b = tf.Variable(tf.zeros([1, 10]), name="bias")

# 矩阵相乘再加上偏移
logits = tf.matmul(X, w) + b

# softmax模型(注意版本)
entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y, name='loss')

# 求所有样本平均值
loss = tf.reduce_mean(entropy)

# 使用实现梯度下降算法的优化器，初始学习率为0.001，最小化损失函数
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

with tf.Session() as sess:
    # 输出图到当前路径下到train目录
    writer = tf.summary.FileWriter('train', sess.graph)

    # 获取时间
    start_time = time.time()

    # 初始化全局变量（比如:w和b）
    sess.run(tf.global_variables_initializer())

    # 计算训练集总批次 （总训练样本/批量输入大小）
    n_batches = int(mnist.train.num_examples / batch_size)

    # 训练模型
    for i in range(n_epochs):
        # 初始化总损失
        total_loss = 0

        # 循环批次
        for _ in range(n_batches):
            # 批量导入训练集数据
            X_batch, Y_batch = mnist.train.next_batch(batch_size)

            # 获取损失函数的值（sess.run执行了optimizer, loss 2个函数，_, lost表示分别赋值函数执行结果，_表示忽略结果）
            _, loss_batch = sess.run([optimizer, loss], feed_dict={X: X_batch, Y: Y_batch})

            # 累加总损失函数的值
            total_loss += loss_batch

        # 输出平均损失(总损失／总批次)
        print('Average loss epoch {0}: {1}'.format(i, total_loss / n_batches))

    # 输出训练模型总耗时
    print('Total time: {0} seconds'.format(time.time() - start_time))

    # 输出优化完成
    print('Optimization Finished!')

    # 测试模型
    # 计算测试集总批次 （总测试样本/批量输入大小）
    n_batches = int(mnist.test.num_examples / batch_size)

    # 总预测
    total_correct_preds = 0

    # 循环批次测试
    for i in range(n_batches):
        # 批量导入测试集数据
        X_batch, Y_batch = mnist.test.next_batch(batch_size)

        # 运行优化器，损失函数，
        _, loss_batch, logits_batch = sess.run([optimizer, loss, logits], feed_dict={X: X_batch, Y: Y_batch})

        # 数学模型
        preds = tf.nn.softmax(logits_batch)
        correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(Y_batch, 1))
        accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))

        # 累加总预测
        total_correct_preds += sess.run(accuracy)

    # 输出准确率
    print('Accuracy {0}'.format(total_correct_preds / mnist.test.num_examples))

    # 关闭输出
    writer.close()
