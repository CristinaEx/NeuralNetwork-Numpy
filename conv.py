import numpy
from math import floor,ceil

def conv2d(data,filter,strides=[1,1,1,1],padding='SAME'):
    """
    data:数据[batch_size,h,w,channel]
    filter:卷积算子[h,w,channel_in,channel_out]
    strides:步长[dh,dw]
    padding:'VAILD' or 'SAME'
    """
    batch_size,h,w,channel = data.shape
    fh,fw,channel_in,channel_out = filter.shape
    if padding == 'VAILD':
        result = numpy.zeros((batch_size,floor((h-fh)/strides[0])+1,floor((w-fw)/strides[1])+1,channel_out))
        for b in range(batch_size):
            for m in range(floor((h-fh)/strides[0])+1):
                x = int(m*strides[0])
                for n in range(floor((w-fw)/strides[1])+1):
                    y = int(n*strides[1])
                    for c in range(channel_out):
                        result[b,m,n,c]=sum(sum(sum(data[b,x:x+fh,y:y+fw,:]*filter[:,:,:,c])));
    else:
        data0 = numpy.pad(data,((0,0),(floor(fh/2),floor(fh/2)),(floor(fw/2),floor(fw/2)),(0,0)),'constant')
        result0 = conv2d(data0,filter,strides,padding='VAILD')
        result = result0[:,0:h,0:w,:]
    return result

if __name__ == '__main__':
    A = numpy.ones((3,2,2,1))
    B = numpy.ones((4,4,1,1))
    # C = conv2d(A,B,[1,1],'VAILD')
    D = conv2d(A,B,[1,1],'SAME')
    # print(C[0,:,:,0])
    print(D[0,:,:,0])