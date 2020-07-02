from mnist_load import *
from net import *
from mnist_train import MODEL_PATH
import random
import numpy

if __name__ == '__main__':
    test_images = load_test_images()
    test_labels = load_test_labels()
    net = ConvNet()
    net.load(MODEL_PATH)
    batch_num = 100
    correct_num = 0
    error_num = 0
    for i in range(int(len(test_images)/batch_num)):
        data = test_images[i:i+batch_num]
        data = numpy.reshape(data,[batch_num,28,28,1])
        label = test_labels[i:i+batch_num]
        # print(label)
        net.addData(data)
        result = net.count()[:,0,0,:]
        for j in range(batch_num):
            m = 0
            for i in range(len(result[j])):
                if result[j][i] > result[j][m]:
                    m = i
            if int(label[j]) == m:
                correct_num += 1
            else:
                error_num += 1
        print('correct rate:'+str(correct_num/(correct_num+error_num)))