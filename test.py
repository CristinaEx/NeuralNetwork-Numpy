import numpy
import matplotlib.pyplot as plt
import math
from net import Net

# 生成符合正态分布的初始数据
# batch_size = 100,shape = (2,)
batch_size = 100
x = numpy.random.normal(size=(batch_size,1,1,2))
w0 = numpy.random.normal(size=(2,1))
b0 = numpy.random.normal(size=(1,))
y0 = (numpy.dot(x,w0)+b0 > 0) + 0
# 数据可视化
plt.figure(1)
for i in range(batch_size):
    color = '.r'
    if y0[i] == 0:
        color = '.b'
    plt.plot(x[i,0,0,0],x[i,0,0,1],color)
plt.show()
plt.figure(2)
net = Net()
net.addData(x)
net.addConvLayout([1,1,2,1],bias = True,padding='VAILD',st_func='SIGMOID',init_type='RANDOM')
print(net)
# print(net.count()[:,0,0,0])
# 学习
learning_rate = 3
for i in range(1000): 
    net.regress(learning_rate,y0)
print(sum(abs(net.count()[:,0,0,0]-y0[:,0,0,0]))/batch_size)
for i in range(batch_size):
    color = '.r'
    if y0[i] == 0:
        color = '.b'
    plt.plot(x[i,0,0,0],x[i,0,0,1],color)
x_ = numpy.array(list(range(-200,200)))/100
y = (0.5-(net.conv_filter[0][0,0,0,0]*x_+1*net.conv_bias[0]))/net.conv_filter[0][0,0,1,0]
y = numpy.reshape(y,(400,))
plt.plot(x_,y,'g')
plt.show()