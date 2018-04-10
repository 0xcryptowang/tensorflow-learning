# tensorflow-learning
tensorflow学习  

1. 定义模型
```python
import tensorflow as tf
import numpy as np

# 定义模型
a = tf.add(2,3)

# 输出
print(a)
```
输出:
```  
Tensor("Add:0", shape=(), dtype=int32)  
```

2. 定义模型并执行结果
```python  
import tensorflow as tf
import numpy as np

# 定义模型
a = tf.add(2,3)

# 开启session
sess = tf.Session()

# 输出
print(sess.run(a))

# 关闭session
sess.close()
```  
输出:
```  
2018-04-08 13:00:17.952598: I tensorflow/core/platform/cpu_feature_guard.cc:140] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
5
```

3. 使用tensorBoard
```python
import tensorflow as tf

# 定义模型
a = tf.add(2.0,3.0)

# 获取session
sess = tf.Session()

# 输出图到当前路径下到train目录
writer = tf.summary.FileWriter("train", sess.graph)

# 加载模型
sess.run(a)

# 关闭session
sess.close()

# 关闭写入
writer.close()
```  

3.1 命令行启动tensorBoard(进入程序路径，指定tensorBoard可视化日志目录)
```
tensorboard --logdir=train
```  

3.2 web访问tensorBoard
```
http://localhost:6006/
```
3.3 访问结果  
![sum](images/sum.png)  

4. 查看tensorflow版本(cpu还是gpu版本)
```python

import tensorflow as tf

# 定义模型
a = tf.add(2.0,3.0)

# 获取session,配置输出设备日志
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

# 加载模型
sess.run(a)

# 关闭session
sess.close()
```  
控制台输出内容，可以看出是cpu还是gpu工作:  
```
2018-04-09 10:05:04.258885: I tensorflow/core/platform/cpu_feature_guard.cc:140] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
Device mapping: no known devices.
2018-04-09 10:05:04.259206: I tensorflow/core/common_runtime/direct_session.cc:297] Device mapping:

Add: (Add): /job:localhost/replica:0/task:0/device:CPU:0
2018-04-09 10:05:04.260529: I tensorflow/core/common_runtime/placer.cc:875] Add: (Add)/job:localhost/replica:0/task:0/device:CPU:0
Add/y: (Const): /job:localhost/replica:0/task:0/device:CPU:0
2018-04-09 10:05:04.260546: I tensorflow/core/common_runtime/placer.cc:875] Add/y: (Const)/job:localhost/replica:0/task:0/device:CPU:0
Add/x: (Const): /job:localhost/replica:0/task:0/device:CPU:0
2018-04-09 10:05:04.260624: I tensorflow/core/common_runtime/placer.cc:875] Add/x: (Const)/job:localhost/replica:0/task:0/device:CPU:0
```  

5. 指定设备进行操作  
```python
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

# 加载模型
print(sess.run(c))
```  

6. 指定图进行操作  
```python
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

# 关闭session
sess.close()
```  

7. 指定多个图进行操作（这里深入的区别还是没看出来，可能是因为我mac本没有gpu的原因，暂时先这么理解。）  
```python
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
```  

8. 线性回归
```
我们经常听说保险公司使用例如一个社区的火灾和盗贼去衡量一个社区的安全程度，我的问题是，这是不是多余的，
火灾和盗贼在一个社区里是否是相关的，如果相关，那么我们能不能找到他们的关系

换句话说，我们能不能找到一个函数f，如果X是火灾数并且Y是盗贼数，是否存在Y=f(X)？

给出这个关系，如果我们有特定社区的火灾数我们能不能预测这一区域的盗贼数

我们有 the U.S. Commission on Civil Rights, courtesy of C engage Learning 的一
个数据集(data目录下的fire_theft.xls文件)

数据集描述：
名称：芝加哥的火灾和盗贼
X=每1000住房单元的火灾数
Y=每1000人口的盗贼数
每对数据取自地区码不同的芝加哥的42个区域

方案：
首先假设火灾数和盗贼数成线性关系：Y=wX+b
我们需要找到参数w和b，通过平方差误差作为的损失函数：（Y - Y_predicted)²
```  
实现代码如下：
```python
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
    # 初始化全局变量（比如:w和b）
    sess.run(tf.global_variables_initializer())

    # 输出图到当前路径下到train目录
    writer = tf.summary.FileWriter('train', sess.graph)

    # 关闭输出
    writer.close()

    # 训练模型100次
    for i in range(100):
        # 初始化损失函数为0
        total_loss = 0
        for x, y in data:
            # 获取损失函数的值（sess.run执行了optimizer, loss 2个函数，_, lost表示分别赋值函数执行结果，_表示忽略结果）
            _, lost = sess.run([optimizer, loss], feed_dict={X: x, Y:y})
            # 累加总损失函数的值
            total_loss += lost
        # 输出每次的训练平均损失值 （截止到当次循环完成的总损失值/总样本数）
        print ('Epoch {0}: {1}'.format(i, total_loss/n_samples))

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
```  
图形结果：  
![lineRegression](images/lineRegression.png)  

tensorBoard结果：
![lineRegressionGraph](images/lineRegressionGraph.png)

9. 


