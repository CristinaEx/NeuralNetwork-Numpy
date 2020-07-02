from mnist_load import *
from net import *
import random

MODEL_PATH = 'model\\model.dat'

def num2oneHot(nums,len_):
    """
    nums:列表
    len_:one-hot向量长度
    return one-hot 向量[0:len-1]
    """
    result = []
    for i in range(len(nums)):
        result.append([])
        num = int(nums[i])
        for j in range(num):
            result[i].append(0)
        result[i].append(1)
        for j in range(len_-num-1):
            result[i].append(0)
    return result

def train():
    train_images = load_train_images()
    train_labels = load_train_labels()
    
    batch_size = 30
    learning_rate = 0.0001
    train_num = 1 # 训练次数
    it_num = 10 # 每次迭代次数
    init_type = 'RANDOM'

    net = ConvNet()
    if not net.load(MODEL_PATH):      
        net.addConvLayout([3,3,1,4],bias = True,padding='VAILD',init_type=init_type,st_func='LEAKY_RELU_0.01')
        net.addConvLayout([3,3,4,8],bias = True,padding='VAILD',init_type=init_type,st_func='LEAKY_RELU_0.01')
        net.addConvLayout([5,5,8,16],bias = True,padding='VAILD',init_type=init_type,st_func='LEAKY_RELU_0.01')
        net.addConvLayout([5,5,16,32],bias = True,padding='VAILD',init_type=init_type,st_func='LEAKY_RELU_0.01')
        net.addConvLayout([16,16,32,64],bias = True,padding='VAILD',st_func='SIGMOID',init_type=init_type)
        net.addConvLayout([1,1,64,10],bias = True,padding='VAILD',st_func='SIGMOID',init_type=init_type)

    bn_net = BNNet()
    main_net = MainNet(bn_net)
    main_net.addNet(net)

    for j in range(train_num):
        index = random.randint(1,len(train_images)-batch_size-1)
        # index = 0
        data = train_images[index:index+batch_size]
        data = numpy.reshape(data,[batch_size,28,28,1])
        label = train_labels[index:index+batch_size]
        # 改为one_hot向量
        label = num2oneHot(label,10)
        label = numpy.reshape(label,[batch_size,1,1,10])
        main_net.addData(data)
        for i in range(it_num):
            loss = sum(sum(sum(sum(abs(net.count()-label)))))/batch_size
            # print(net.conv_filter[0][:,:,0,0])
            print(loss)
            net.regress(learning_rate,label,regress_type='SGD',loss_type = 'CE')  # 交叉熵
        print('saving...')
        net.save(MODEL_PATH)
        print('finish!') 
        learning_rate = 0.995*learning_rate
        
    loss = sum(sum(sum(sum(abs(net.count()-label)))))/batch_size
    print(loss)

    net.save(MODEL_PATH)

def test():
    test_images = load_test_images()
    test_labels = load_test_labels()

    test_num = 10

    net = ConvNet()
    net.load(MODEL_PATH)

    bn_net = BNNet()
    main_net = MainNet(bn_net)
    main_net.addNet(net)

    for i in range(test_num):
        index = random.randint(1,len(test_images)-2)
        # index = i
        data = test_images[index]
        data = numpy.reshape(data,[1,28,28,1])
        label = test_labels[index]
        print('correct result:'+str(int(label)))
        main_net.addData(data)
        result = main_net.count()[0,0,0,:]
        j = 0
        for i in range(len(result)):
            if result[i] > result[j]:
                j = i
        print('net result:'+str(j))


if __name__ == '__main__':
    train()
    test()
    