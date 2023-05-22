import matplotlib.pyplot as plt
import builtins
import datetime
import numpy as np
import math

old_print = builtins.print
def hook_print():
    def my_print(*args, **kwargs):
        old_print(datetime.datetime.now(), end=" ")
        old_print(*args, **kwargs)
    builtins.print = my_print
def unhook_print():
    builtins.print = old_print
hook_print()

# usage: save_plts(2,2,os.path.join(args.path,"./result.jpg"),img1,img2,img3,img4)
def save_plts(row,col,fileName,*args,gray=False,**kwargs):
    plt.figure(dpi=300,figsize=(12,8))
    for i,item in enumerate(args):
        plt.subplot(row,col,i+1)
        if gray == True:
            plt.imshow(item,'gray')
            continue
        plt.imshow(item)
    plt.savefig(fileName) 
    plt.close('all')

def padding_operation(input,padding):
    channel, input_h , input_w = input.shape
    for i in range(padding):
        input = np.insert(input,0,0,axis=1)
        input = np.insert(input,input_h+i+1,0,axis=1)
    for i in range(padding):
        input = np.insert(input,0,0,axis=2)
        input = np.insert(input,input_w+i+1,0,axis = 2)
    return input

# old version of implement of conv, less efficient
def matrix_multiplication_for_conv2d_old(input:np.ndarray, kernel:np.ndarray,bias = 0, stride=1, padding=0):
    if padding > 0:
        input = padding_operation(input, padding)
    # 计算输出大小
    channel, input_h , input_w = input.shape
    out_channel, in_channel,  kernel_h, kernel_w = kernel.shape

    output_h = (math.floor((input_h - kernel_h) / stride) + 1)
    output_w = (math.floor((input_w - kernel_w) / stride) + 1)

    output = [[[0.0 for i in range(output_w)] for i in range(output_h)] for i in range(out_channel)]
	
    # 不考虑性能
    for oc in range(out_channel):
        for ic in range(in_channel):
            for i in range(0, input_h - kernel_h + 1, stride):
                for j in range(0, input_w - kernel_w + 1, stride):
                    region = input[ic, i:i + kernel_h, j: j + kernel_w]
                    output[oc][int(i / stride)][int(j / stride)] += np.sum(region * kernel[oc][ic])
        # output[oc] += bias[oc]

    return np.array(output)