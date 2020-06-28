import numpy
from math import floor,ceil

def conv2d(data,filter,strides=[1,1],padding='SAME'):
    """
    data:数据[batch_size,h,w,channel]
    filter:卷积算子[h,w,channel_in,channel_out]
    filter的w,h维度是颠倒的，可以修改，但没必要，还省时间
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

def deconv2d(loss,filter,strides=[1,1],padding='SAME'):
    """
    反卷积
    计算loss传播
    目前仅支持步长为1
    """
    batch_size,h,w,channel = loss.shape
    fh,fw,channel_in,channel_out = filter.shape
    new_filter = filter.transpose(0,1,3,2) # output_channel转换
    new_filter = numpy.flip(new_filter,0)
    new_filter = numpy.flip(new_filter,1) # 算子上下左右翻转
    if padding == 'VAILD':
        data0 = numpy.pad(loss,((0,0),(fh-1,fh-1),(fw-1,fw-1),(0,0)),'constant')
    else:
        data0 = numpy.pad(loss,((0,0),(floor(fh/2),floor(fh/2)),(floor(fw/2),floor(fw/2)),(0,0)),'constant')
    return conv2d(data0,new_filter,strides,'VAILD')

if __name__ == '__main__':
    A = numpy.array(
        [
            [[[3]],[[4]]],
            [[[5]],[[6]]]
        ]
    )
    B = numpy.array(
        [
            [[[-1]],[[1]]],
            [[[2]],[[-2]]]
        ]
    )
    B = numpy.reshape(B,[1,2,2,1])
    print(A.shape)
    print(B.shape)
    print(deconv2d(B,A,padding='VAILD')[0,:,:,0])