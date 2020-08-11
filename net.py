import numpy
from conv import conv2d,deconv2d
from math import floor,ceil
from os import mkdir
from os.path import isdir,dirname,isfile

class MainNet:
    # 多重网络
    def __init__(self,net):
        """
        net:第一个网络
        """
        self.nets = [net] # 网络
        pass

    def addNet(self,net):
        """
        添加网络，接入当前网络最后一级
        """
        self.nets.append(net)

    def addData(self,data):
        """
        向所有网络添加数据
        """
        for net in self.nets:
            net.addData(data)
            data = net.count()

    def count(self):
        """
        返回最后一层输出
        """
        return self.nets[-1].count()

class BNNet:
    # Batch Normalization
    def __init__(self):
        self.data = None # 输入[batch_size,w,h,mod]
        self.gamma = 1
        self.beta = 0
        self.min_error = 1e-8
        pass

    def addData(self,data):
        """
        添加数据[batch_size,w,h,mod]
        """
        self.data = data
        self.u = None
        self.o = None

    def count(self):
        # self.data = [batch_size,w,h,mod]
        batch_size,w,h,mod = self.data.shape
        self.u = numpy.mean(self.data,axis=0) # shape:[w,h,mod]
        self.o = numpy.var(self.data,axis=0) # shape:[batch_size,w,h,mod]
        self.norm = (self.data-self.u)/self.o
        return self.norm*self.gamma+self.beta

    def backward(self,loss,learning_rate): 
        batch_size,w,h,mod = numpy.shape(self.data)

        X_mu = self.data-self.u
        std_inv = 1/numpy.sqrt(self.o+self.min_error)
        dX_norm = loss*self.gamma
        dvar = numpy.sum(dX_norm*X_mu,axis=0)*(-0.5)*std_inv**3
        dmu = numpy.sum(dX_norm*-std_inv,axis=0)+dvar*numpy.mean(-2*X_mu,axis=0)

        dX = dX_norm*std_inv+dvar*2*X_mu/batch_size+dmu/batch_size
        dgamma = numpy.sum(loss*self.norm,axis=0)
        dbeta = numpy.sum(loss,axis=0)

        return dX,dgamma,dbeta

class RMSProp:
    def __init__(self):
        self.r = [] # 累积平方梯度
        self.ro = 0.5 # 衰减速率
        self.min_error = 0.000001 # 小常数
        self.layout_index = 0 # 从后到前的layout
        # filter <- filter - v
        # filter_loss = v

    def regress(self,loss,data,conv_filter,conv_bias,strides,padding,learning_rate):
        """
        进行梯度下降
        loss:经过激活函数传递上来的损失值
        data:上一层参数
        conv_filter:算子
        conv_bias:偏置项
        st_func(conv2d(data,filter)) = now_layout
        strides:[dy,dx]步长
        learning_rate:学习率
        返回filter_loss,bias_loss
        """
        fh,fw,channel_in,channel_out = conv_filter.shape # 当前filter
        [batch_size,m,n,channel_out] = loss.shape

        if len(self.r) <= self.layout_index:
            # 添加初始累积平方梯度
            self.r.append(numpy.zeros([fh,fw,channel_in,channel_out]))
        # [fh,fw,channel_in] .* [fh,fw,channel_in,channel_out] = [1,1,channel_out] 
        # [fh,fw,channel_in,cho] = [fh,fw,channel_in] * [1,1,cho] .* learning_rate / filter_size
        filter_loss = numpy.zeros(conv_filter.shape)
        bias_loss = numpy.zeros(conv_bias.shape)
        [lb,lh,lw,lc] = loss.shape
        K = learning_rate/batch_size
        for b in range(batch_size):
            for x in range(lh):
                x = x*strides[0]
                for y in range(lw):
                    y = y*strides[1]
                    for ch in range(channel_out):
                        filter_loss[:,:,:,ch] = filter_loss[:,:,:,ch] + loss[b,x,y,ch] * data[b,x:x+fh,y:y+fw,:]*K
                        bias_loss[ch] = loss[b,x,y,ch]*K
        # 更新累积梯度
        self.r[self.layout_index] = self.r[self.layout_index] * self.ro + (1-self.ro)*filter_loss*filter_loss;    
        loss = deconv2d(loss,conv_filter,strides,padding) # 更新loss           
        filter_loss = filter_loss/numpy.sqrt(self.r[self.layout_index]+self.min_error)
        self.layout_index = self.layout_index + 1 # 下一层
        return filter_loss,bias_loss,loss

    def __str__(self):
        return 'RMSPROP'

    def reinit(self):
        """
        每次优化时初始化
        """
        self.layout_index = 0

class Nesterov:
    def __init__(self):
        self.v = [] # 初始速度 为0
        self.alpha = 0.5 # 动量参数 v<-alpha*v+Loss
        self.layout_index = 0 # 从后到前的layout
        # filter <- filter - v
        # filter_loss = v

    def regress(self,loss,data,conv_filter,conv_bias,strides,padding,learning_rate):
        """
        进行梯度下降
        loss:经过激活函数传递上来的损失值
        data:上一层参数
        conv_filter:算子
        conv_bias:偏置项
        st_func(conv2d(data,filter)) = now_layout
        strides:[dy,dx]步长
        learning_rate:学习率
        返回filter_loss,bias_loss
        """
        fh,fw,channel_in,channel_out = conv_filter.shape # 当前filter
        [batch_size,m,n,channel_out] = loss.shape

        if len(self.v) <= self.layout_index:
            # 添加初始动量
            self.v.append(numpy.zeros([fh,fw,channel_in,channel_out]))
        conv_filter_new = conv_filter - self.alpha * self.v[self.layout_index]
        loss_new = deconv2d(loss,conv_filter_new,strides,padding) # 针对data的loss
        data_new = data - loss_new
        # [fh,fw,channel_in] .* [fh,fw,channel_in,channel_out] = [1,1,channel_out] 
        # [fh,fw,channel_in,cho] = [fh,fw,channel_in] * [1,1,cho] .* learning_rate / filter_size
        filter_loss = numpy.zeros(conv_filter.shape)
        bias_loss = numpy.zeros(conv_bias.shape)
        [lb,lh,lw,lc] = loss.shape
        K = learning_rate/batch_size
        for b in range(batch_size):
            for x in range(lh):
                x = x*strides[0]
                for y in range(lw):
                    y = y*strides[1]
                    for ch in range(channel_out):
                        filter_loss[:,:,:,ch] = filter_loss[:,:,:,ch] + loss[b,x,y,ch] * data_new[b,x:x+fh,y:y+fw,:]*K
                        bias_loss[ch] = loss[b,x,y,ch]*K
        self.v[self.layout_index] = self.v[self.layout_index]*self.alpha+filter_loss               
        filter_loss = self.v[self.layout_index]
        self.layout_index = self.layout_index + 1 # 下一层
        return filter_loss,bias_loss,loss_new

    def __str__(self):
        return 'NESTEROV'

    def reinit(self):
        """
        每次优化时初始化
        """
        self.layout_index = 0
        

class SGD:
    def __init__(self):
        self.SGD_K = 0.1 # 随机抽取其中的0.1倍进行回归

    def regress(self,loss,data,conv_filter,conv_bias,strides,padding,learning_rate):
        """
        进行随机梯度下降
        loss:经过激活函数传递上来的损失值
        data:上一层参数
        conv_filter:算子
        conv_bias:偏置项
        st_func(conv2d(data,filter)) = now_layout
        strides:[dy,dx]步长
        learning_rate:学习率
        padding:补位方式
        返回filter_loss,bias_loss
        """
        fh,fw,channel_in,channel_out = conv_filter.shape # 当前filter
        [batch_size,m,n,channel_out] = loss.shape
        SGD_NUM = ceil(m*n*self.SGD_K) # 每个batch抽取SGD_NUM个区域进行回归,每个区域大小为[fh,fw,channel_in]
        # [fh,fw,channel_in] .* [fh,fw,channel_in,channel_out] = [1,1,channel_out] 
        # [fh,fw,channel_in,cho] = [fh,fw,channel_in] * [1,1,cho] .* learning_rate / filter_size
        filter_loss = numpy.zeros(conv_filter.shape)
        bias_loss = numpy.zeros(conv_bias.shape)
        [lb,lh,lw,lc] = loss.shape
        x0 = numpy.random.randint(lh,size=SGD_NUM)
        y0 = numpy.random.randint(lw,size=SGD_NUM)
        K = learning_rate/SGD_NUM/batch_size
        for b in range(batch_size):
            for j in range(SGD_NUM):
                x = x0[j]*strides[0]
                y = y0[j]*strides[1]
                for ch in range(channel_out):
                    filter_loss[:,:,:,ch] = filter_loss[:,:,:,ch] + loss[b,x,y,ch] * data[b,x:x+fh,y:y+fw,:]*K
                    bias_loss[ch] = loss[b,x,y,ch]*K
        loss = deconv2d(loss,conv_filter,strides,padding) # 更新loss
        return filter_loss,bias_loss,loss

    def __str__(self):
        return 'SGD'

    def reinit(self):
        pass

class ConvNet:
    # 全卷积神经网络
    def __init__(self):
        self.data = None # 输入[batch_size,w,h,mod]
        self.conv_layout = [] # 卷积层
        self.conv_filter = [] # 每层卷积层对应卷积核
        self.conv_bias = [] # 每个卷积核对应偏置项
        self.st_func = [] # 每层网络对应激活函数
        self.padding = [] # 每层网络对应padding方式
        self.strides = [] # 每层网络卷积核对应步长
        self.regress_tool = SGD() # 默认SGD
        

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
            self.conv_bias.append(numpy.array([]))
        if init_type == 'ZERO':
            self.conv_filter.append(numpy.zeros(filter_shape))
            if not bias == None:
                self.conv_bias.append(numpy.zeros(filter_shape[3]))
        else:
            self.conv_filter.append(numpy.random.standard_normal(filter_shape)/(filter_shape[0]*filter_shape[1]))
            if not bias == None:
                self.conv_bias.append(numpy.random.standard_normal(filter_shape[3]))
        # if self.conv_layout:
        #     data = self.conv_layout[-1]
        # else:
        #     data = self.data
        # layout = conv2d(data,self.conv_filter[-1],strides,padding)
        # if not bias == None:
        #     for channel in range(filter_shape[3]):
        #         layout[:,:,:,channel] += self.conv_bias[-1][channel]
        # if st_func == 'SIGMOID':
        #     layout = self.__sigmoid(layout)
        self.st_func.append(st_func)
        self.padding.append(padding)
        self.strides.append(strides)
        self.conv_layout.append(None)

    def __sigmoid(self,x):
        """
        sigmoid
        return y = 1/(1+exp(-x))
        """
        return 1/(1+numpy.exp(-x))

    def __leakyRelu(self,x,alpha):
        """
        leaky relu = (x<0)*alpha+(x>0)*1
        """
        return ((x<0)*alpha+(x>0))*x

    def __sigmoid_loss(self,y):
        """
        y = 1/(1+exp(-x))
        sigmoid 导数
        返回dy/dx
        """
        return y*(1-y)

    def __leakyRelu_loss(self,y,alpha):
        return (y<0)*alpha+(y>=0)

    def load(self,fileName):
        """
        读取
        """
        if not isfile(fileName):
            return False
        self.__init__() # 初始化
        with open(fileName) as f:
            layout_num = int(f.readline())
            for i in range(layout_num):
                filter_size = [int(x) for x in list(f.readline()[1:-2].split(','))]
                b = False
                if len(list(f.readline())) >= 2:
                    b = True
                strides = [int(x) for x in list(f.readline()[1:-2].split(','))]
                padding = f.readline()[0:-1]
                st_func = f.readline()[0:-1]
                self.addConvLayout(filter_size,bias = b,strides = strides,padding = padding,st_func = st_func)
        for i in range(len(self.conv_filter)):
            data = numpy.load(dirname(fileName) + '\\layout_'+str(i)+'.npz')
            self.conv_filter[i]=data['x']
            self.conv_bias[i]=data['y']
        return True

    def save(self,filename):
        """
        保存
        """
        if not isdir(dirname(filename)):
            mkdir(dirname(filename))
        with open(filename,'w+') as f:
            f.write(str(len(self.conv_filter)))
            f.write('\n')
            for i in range(len(self.conv_filter)):
                f.write(str(self.conv_filter[i].shape))
                f.write('\n')
                f.write(str(self.conv_bias[i].shape))
                f.write('\n')
                f.write(str(self.strides[i]))
                f.write('\n')
                f.write(str(self.padding[i]))
                f.write('\n')
                f.write(str(self.st_func[i]))
                f.write('\n')
        for i in range(len(self.conv_filter)):
            numpy.savez(dirname(filename) + '\\layout_'+str(i),x=self.conv_filter[i],y=self.conv_bias[i])
        return True

    def regress(self,learning_rate,label,regress_type='SGD',loss_type='MSE'):
        """
        后向传播
        learing_rate:学习率
        label:最后一层的标准输出
        regress_type:回归类型(SGD:随机梯度下降...)
        loss_type:MSE or CE(交叉熵)
        返回第一层的loss(反向传播到第一层)
        """
        self.count() # 更新权值
        # 更换优化器
        if not str(self.regress_tool) == regress_type:
            # print('CHANGE REGRESS TYPE')
            if regress_type == 'SGD':
                self.regress_tool = SGD()
            elif regress_type == 'NESTEROV':
                self.regress_tool = Nesterov()
            elif regress_type == 'RMSPROP':
                self.regress_tool = RMSProp()
        self.regress_tool.reinit()

        now_layout = self.conv_layout[-1] # 当前layout
        # -------------------------经过损失函数传播loss---------------------------------
        if loss_type == 'CE':
            min_error = 0.000000001
            loss = (-label/(now_layout+min_error)+(1-label)/(1-now_layout+min_error))
        else:
            # MSE
            loss = 2*(now_layout-label) # loss = d_loss/
        # -------------------------------------------------------------------------------
        for i in range(1,len(self.conv_layout)+1):
            now_layout = self.conv_layout[-i] # 当前layout
            if i == len(self.conv_layout):
                last_layout = self.data
            else:
                last_layout = self.conv_layout[-i-1]
            batch_size,h,w,channel = last_layout.shape
            fh,fw,channel_in,channel_out = self.conv_filter[-i].shape # 当前filter
            # true_layout = st_func^(-1)(label)
            # ---------------------经过激活函数传播loss------------------------
            if self.st_func[-i] == 'SIGMOID':
                loss = loss*self.__sigmoid_loss(now_layout)# loss = (d_loss/d_st_func_out_put) * (d_st_func_out_put/d_hidden_output) = d_loss/d_hidden_output
            elif self.st_func[-i][0:10] == 'LEAKY_RELU':
                alpha = float(self.st_func[-i][11:])
                loss = loss*self.__leakyRelu_loss(now_layout,alpha)
            # -----------------------------------------------------------------
            # 注:d_hidden_output/dw = data
            # 进行随机梯度下降更新权值
            data = last_layout
            if self.padding[-i] == 'SAME':
                data = numpy.pad(data,((0,0),(floor(fh/2),floor(fh/2)),(floor(fw/2),floor(fw/2)),(0,0)),'constant')
            # conv(data,filter[-1],strides,'VAILD') = now_layout
            # filter[-1] = filter[-1]-sum(```在batch_size维度上累加```learning_rate*loss*data[batch_size,x0:x0+fw,y0:y0+fh,:]/filter_size)
            # --------------------------梯度下降------------------------------
            filter_loss,bias_loss,loss = self.regress_tool.regress(loss,data,self.conv_filter[-i],self.conv_bias[-i],self.strides[-i],self.padding[-i],learning_rate)              
            # ------------------------------------------------------------------
            # 更新后的卷积算子权值
            self.conv_filter[-i] = self.conv_filter[-i] - filter_loss
            if not len(self.conv_bias[-i]) == 0:
                self.conv_bias[-i] = self.conv_bias[-i] - bias_loss
        return loss

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
                layout = self.__sigmoid(layout);
            elif self.st_func[i][0:10] == 'LEAKY_RELU':
                alpha = float(self.st_func[i][11:])
                layout = self.__leakyRelu(layout,alpha)
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
    net = ConvNet()
    net.addData(numpy.ones([10,28,28,1]))
    net.addConvLayout([3,3,1,8],bias = None,init_type='RANDOM',st_func='LEAKY_RELU_0.02')
    net.addConvLayout([3,3,8,16],bias = None,init_type='RANDOM',st_func='LEAKY_RELU_0.02')
    net.addConvLayout([5,5,16,32],bias = True,init_type='RANDOM',st_func='LEAKY_RELU_0.02')
    net.addConvLayout([28,28,32,10],bias = True,padding='VAILD',st_func='SIGMOID',init_type='RANDOM')
    net.count()
    print(net)
    print(net.conv_filter[0][:,:,0,0])
    net.regress(0.001,numpy.ones(net.count().shape))
    print(net.conv_filter[0][:,:,0,0])

    