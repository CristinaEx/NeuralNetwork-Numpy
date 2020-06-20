## NeuralNetwork-Numpy
### 模式识别大作业 使用numpy进行深度学习

### 使用方法
```
from net import Net
net = Net()
net.addData(x)
net.addConvLayout([1,1,2,1],bias = True,padding='VAILD',st_func='SIGMOID',init_type='RANDOM')
```
- addData()：添加数据
- addConvLayout()：在当前网络最后面添加一层网络
- regress()：回归
- count()：计算各层输出
### test.py的可视化输出

![pic_1](1.JPG)

- print(net) 输出网络结构

![pic_2](2.JPG)