import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import xlrd

# 火灾数据集文件路径
DATA_FILE = 'data/fire_theft.xls'

# 读取xls文件
book = xlrd.open_workbook(DATA_FILE, encoding_override="utf-8")

# 获得第一个sheet
sheet = book.sheet_by_index(0)

# 取出每一行的值构建成数组
# 1.for i in range(1, sheet.nrows)遍历sheet的所有行
# 2.sheet.row_values(i)取出一行的值
# 3.np.asarray(...）构建成数组
data = np.asarray([sheet.row_values(i) for i in range(1, sheet.nrows)])

# 样本数量
n_samples = sheet.nrows - 1

# 定义占位符，并起名
X = tf.placeholder(tf.float32, name='X')
Y = tf.placeholder(tf.float32, name='Y')

# 定义变量，初始化为0，并起名
w = tf.Variable(0.0, name='weights')
b = tf.Variable(0.0, name='bias')

# 创建模型预测Y Y=wX+b
Y_predicted = X * w + b

# 使用平方误差构建损失函数（Y - Y_predicted)²
loss = tf.square(Y - Y_predicted, name='loss')

# 使用实现梯度下降算法的优化器，初始学习率为0.001，最小化损失函数
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

# 获取session
with tf.Session() as sess:
    # 初始化变量（比如:w和b）
    sess.run(tf.global_variables_initializer())
    # 输出图到当前路径下到train目录
    writer = tf.summary.FileWriter('train', sess.graph)

    # 训练模型100次
    for i in range(100):
        # 初始化损失函数为0
        total_loss = 0
        for x, y in data:
            # 获取损失函数的值
            _, l = sess.run([optimizer, loss], feed_dict={X: x, Y:y})
            # 累加总损失函数的值
            total_loss += l
        # 输出每次的训练平均损失值 （截止到当次循环完成的总损失值/总样本数）
        print ('Epoch {0}: {1}'.format(i, total_loss/n_samples))

    # 关闭输出
    writer.close()

    # 执行模型并输出模型结果
    w_value, b_value = sess.run([w, b])


# 取出值
X, Y = data.T[0], data.T[1]

# 真实值图形显示
plt.plot(X, Y, 'bo', label='Real data')

# 预测数据图形显示
plt.plot(X, X * w_value + b_value, 'r', label='Predicted data')

# 添加图例的标注
plt.legend()

# 展示图形
plt.show()
