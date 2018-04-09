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
