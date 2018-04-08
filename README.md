# tensorflow-learning
tensorflow学习  

1. 定义图
```python
import tensorflow as tf
import numpy as np

a = tf.add(2,3)
print(a)
```
输出:
```  
Tensor("Add:0", shape=(), dtype=int32)  
```

2. 定义图并执行结果
```python  
import tensorflow as tf
import numpy as np

a = tf.add(2,3)
sess = tf.Session()
print(sess.run(a))
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
import numpy as np

#定义图
a = tf.add(2.0,3.0)

#获取session
sess = tf.Session()

#输出图到当前路径下到train目录
writer = tf.summary.FileWriter("train", sess.graph)

#加载模型
sess.run(a)
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
