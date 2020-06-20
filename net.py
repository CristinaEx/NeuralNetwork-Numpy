import numpy
from conv import conv2d,deconv2d
from math import floor,ceil

class MainNet:
    def __init__(self,net):
        """
        net:第一个网络
        """
        nets = [net] # 网络
        pass

    def addNet(self,net,lastNetPos=[0,0]):
        """
        该网络在上一级网络的位置,比如net的输入[w,h,mod]在目前nets最后进入的网络的[w_pos,h_pos],默认为[0,0]
        """
        pass

class Net:
    def __init__(self):
        self.data = None # 输入[batch_size,w,h,mod]
        self.conv_layout = [] # 卷积层
        self.conv_filter = [] # 每层卷积层对应卷积核
        self.conv_bias = [] # 每个卷积核对应偏置项
        self.st_func = [] # 每层网络对应激活函数
        self.padding = [] # 每层网络对应padding方式
        self.strides = [] # 每层网络卷积核对应步长
        pass

    def addData(self,data):
        """
        添加数据，数据的shape只能改变batch_size
        """
        self.data = data

    def addConvLayout(self,filter_shape,bias=None,strides=[1,1,1,1],init_type='ZERO',padding='SAME',st_func='NONE'):
        """
        使用前self.data必须构建初始值，或者0值填充
        filter_shape:[w,h,input,output]
        bias:None->默认偏置项,True:有偏置,False:无偏置
        strides:步长
        init_type='ZERO' or 'RANDOM' -> 初始置零或高斯分布(推荐RANDOM)
        padding:VAILD or SAME
        st_func:激活函数 NONE(√) SIGMOID(√) RELU LEAKY_RELU_(alpha值) TANH
        """ 
        if bias == None:
            self.conv_bias.append([])
        if init_type == 'ZERO':
            self.conv_filter.append(numpy.zeros(filter_shape))
            if not bias == None:
                self.conv_bias.append(numpy.zeros(filter_shape[3]))
        else:
            self.conv_filter.append(numpy.random.standard_normal(filter_shape)/(filter_shape[0]*filter_shape[1]*filter_shape[2]))
            if not bias == None:
                self.conv_bias.append(numpy.random.standard_normal(filter_shape[3]))
        if self.conv_layout:
            data = self.conv_layout[-1]
        else:
            data = self.data
        layout = conv2d(data,self.conv_filter[-1],strides,padding)
        if not bias == None:
            for channel in range(filter_shape[3]):
                layout[:,:,:,channel] += self.conv_bias[-1][channel]
        if st_func == 'SIGMOID':
            layout = self.__sigmoid(layout)
        self.st_func.append(st_func)
        self.padding.append(padding)
        self.strides.append(strides)
        self.conv_layout.append(layout)

    def __sigmoid(self,x):
        """
        sigmoid
        return y = 1/(1+exp(-x))
        """
        return 1/(1+numpy.exp(-x))

    def __sigmoid_loss(self,y):
        """
        y = 1/(1+exp(-x))
        sigmoid 导数
        返回dy/dx
        """
        return y*(1-y)

    def load(self,fileName):
        """
        读取
        """
        pass

    def save(self,filename):
        """
        保存
        """
        pass

    def regress(self,learning_rate,label,regress_type='SGD'):
        """
        后向传播，仅支持随机梯度下降
        learing_rate:学习率
        label:最后一层的标准输出
        权值更新速率正比于learning_rate
        """
        self.count() # 更新权值
        if regress_type == 'SGD':
            now_layout = self.conv_layout[-1] # 当前layout
            loss = now_layout-label # label = now_layout - loss
            for i in range(1,len(self.conv_layout)+1):
                now_layout = self.conv_layout[-i] # 当前layout
                if i == len(self.conv_layout):
                    last_layout = self.data
                else:
                    last_layout = self.conv_layout[-i-1]
                batch_size,h,w,channel = last_layout.shape
                fh,fw,channel_in,channel_out = self.conv_filter[-i].shape # 当前filter
                # true_layout = st_func^(-1)(label)
                # 经过激活函数传播loss
                if self.st_func[-i] == 'SIGMOID':
                    loss = loss*self.__sigmoid_loss(now_layout)
                # 进行随机梯度下降更新权值
                data = last_layout
                if self.padding[-i] == 'SAME':
                    data = numpy.pad(data,((0,0),(floor(fh/2),floor(fh/2)),(floor(fw/2),floor(fw/2)),(0,0)),'constant')
                # conv(data,filter[-1],strides,'VAILD') = now_layout
                # filter[-1] = filter[-1]-sum(```在batch_size维度上累加```learning_rate*loss*data[batch_size,x0:x0+fw,y0:y0+fh,:]/filter_size)
                SGD_K = 0.1 # 随机抽取其中的0.1倍进行回归
                [batch_size,m,n,channel_out] = loss.shape
                SGD_NUM = ceil(m*n*SGD_K) # 每个batch抽取SGD_NUM个区域进行回归,每个区域大小为[fh,fw,channel_in]
                # [fh,fw,channel_in] .* [fh,fw,channel_in,channel_out] = [1,1,channel_out] 
                # [fh,fw,channel_in,cho] = [fh,fw,channel_in] * [1,1,cho] .* learning_rate / filter_size
                filter_loss = numpy.zeros(self.conv_filter[-i].shape)
                bias = not len(self.conv_bias[-i]) == 0
                if bias:
                    bias_loss = numpy.zeros(self.conv_bias[-i].shape)
                [lb,lh,lw,lc] = loss.shape
                x0 = numpy.random.randint(lh,size=SGD_NUM)
                y0 = numpy.random.randint(lw,size=SGD_NUM)
                filter_size = fh*fw*channel_in
                Kb = learning_rate/SGD_NUM/batch_size
                Kf = Kb/filter_size
                for b in range(batch_size):
                    for j in range(SGD_NUM):
                        x = x0[j]*self.strides[-i][0]
                        y = y0[j]*self.strides[-i][0]
                        for ch in range(channel_out):
                            filter_loss[:,:,:,ch] = filter_loss[:,:,:,ch] + loss[b,x,y,ch] * data[b,x:x+fh,y:y+fw,:]*Kf
                            if bias:
                                bias_loss[ch] = loss[b,x,y,ch]*Kb
                loss = deconv2d(loss,self.conv_filter[-i],self.strides[-i],self.padding[-i]) # 更新loss
                # 更新后的卷积算子权值
                self.conv_filter[-i] = self.conv_filter[-i] - filter_loss
                if bias:
                    self.conv_bias[-i] = self.conv_bias[-i] - bias_loss

    def count(self):
        """
        前向传播，得出结果，输出最后一层
        """
        last_layout = self.data
        for i in range(len(self.conv_layout)):
            layout = conv2d(last_layout,self.conv_filter[i],self.strides[i],self.padding[i])
            if not len(self.conv_bias[i]) == 0:
                filter_shape = self.conv_filter[i].shape
                for channel in range(filter_shape[3]):
                    layout[:,:,:,channel] += self.conv_bias[i][channel]
            if self.st_func[i] == 'SIGMOID':
                layout = 1/(1+numpy.exp(-layout));
            self.conv_layout[i] = layout
            last_layout = layout
        return last_layout

    def __str__(self):
        """
        返回网络参数
        """
        result = 'input_data:'
        result += str(self.data.shape) + '\n'
        for i in range(len(self.conv_layout)):
            result += 'filter:'
            result += str(self.conv_filter[i].shape) + '      '
            result += 'bias:'
            if not len(self.conv_bias[i]) == 0:
                result += str(self.conv_bias[i].shape) + '      \n'
            else:
                result += 'None' + '      \n'
            result += 'st_func:' + self.st_func[i] + '\n'
            result += ('layout_' + str(i) + ':')
            result += str(self.conv_layout[i].shape) + '      \n'         
        return result

if __name__ == '__main__':
    net = Net()
    net.addData(numpy.ones([10,28,28,1]))
    net.addConvLayout([3,3,1,8],bias = None,init_type='RANDOM')
    net.addConvLayout([3,3,8,16],bias = None,init_type='RANDOM')
    net.addConvLayout([5,5,16,32],bias = True,init_type='RANDOM')
    net.addConvLayout([28,28,32,10],bias = True,padding='VAILD',st_func='SIGMOID',init_type='RANDOM')
    print(net)
    print(net.count()[0,:,:,:])
    print(net.conv_filter[3][0:10,0,0,0])
    net.regress(0.001,numpy.ones(net.count().shape))
    print(net.count()[0,:,:,:])
    print(net.conv_filter[3][0:10,0,0,0])