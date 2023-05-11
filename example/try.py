
from ipcl_python import PaillierKeypair, context, hybridControl, hybridMode
import pickle as pkl
import timeit
import sys
import numpy as np
from RED.REDCNN import *
from collections import OrderedDict
import torch
import scipy.io as scio
import torch.nn.functional as F
from matplotlib import image as image, pyplot as plt
import math
import time
import copy

import builtins
import datetime
old_print = builtins.print
 
def hook_print():
    def my_print(*args, **kwargs):
        old_print(datetime.datetime.now(), end=" ")
        old_print(*args, **kwargs)
    builtins.print = my_print
def unhook_print():
    builtins.print = old_print
hook_print()



def padding_operation(input,padding):
    channel, input_h , input_w = input.shape
    for i in range(padding):
        input = np.insert(input,0,0,axis=1)
        input = np.insert(input,input_h+i+1,0,axis=1)
    for i in range(padding):
        input = np.insert(input,0,0,axis=2)
        input = np.insert(input,input_w+i+1,0,axis = 2)
    return input


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

def img2col(x, ksize, stride):
    c,h,w = x.shape
    img_col = []
    for i in range(0, h-ksize+1, stride):
        for j in range(0, w-ksize+1, stride):
            col = x[:, i:i+ksize, j:j+ksize].reshape(-1) 
            img_col.append(col)
    return np.array(img_col)

# @profile
def matrix_multiplication_for_conv2d(input:np.ndarray, kernel:np.ndarray,bias = 0, stride=1, padding=0):
    if padding > 0:
        input = padding_operation(input, padding)
    channel, input_h , input_w = input.shape
    out_channel, in_channel,  kernel_h, kernel_w = kernel.shape

    kernel = kernel.reshape(out_channel, -1)
    output_h = (math.floor((input_h - kernel_h) / stride) + 1)
    output_w = (math.floor((input_w - kernel_w) / stride) + 1)
    # output = [[[0.0 for i in range(output_w)] for i in range(output_h)] for i in range(out_channel)]
    output = np.zeros((out_channel,output_h,output_w))
    input = img2col(input,kernel_h,stride)
    # out = np.dot(kernel, input.T).reshape(out_channel,-1)
    out = np.dot(kernel, input.T).reshape(out_channel,-1)
    output = np.reshape(out, output.shape) 
    return output

def encrypt_matrix(input,pk):
    output = []
    for batch in input:
        batch_output = []
        for data in batch:
            items = []
            for item in data:
                items.append(pk.encrypt(item.astype(float)))
            batch_output.append(items)
        output.append(batch_output)
    output = np.array(output)
    return output

def decrypt_matrix(input,sk):
    output = []
    for batch in input:
        batch_output = []
        for data in batch:
            items = []
            for item in data:
                items.append(sk.decrypt(item))
            batch_output.append(items)
        output.append(batch_output)
    output = np.array(output)
    return output

def encrypt_conv(x,module):
    in_channels = module.in_channels
    out_channels = module.out_channels
    padding,_ = module.padding
    stride,_ = module.stride
    kernel = module.weight.cpu().detach().numpy()
    bias = 0
    print(in_channels,out_channels,padding,stride,kernel.shape)
    x = matrix_multiplication_for_conv2d(x,kernel,bias,stride,padding)
    return x

if __name__ == "__main__":
    # generate Paillier scheme key pair
    pk, sk = PaillierKeypair.generate_keypair(200)
    # Acquire QAT engine control
    context.initializeContext("QAT")

    # net = RED_CNN()
    # net.load_state_dict(torch.load('./example/checkpoint/epoch_49_loss_0.017239.pth'))

    # # validate in torch
    # path = './example/data_0001.mat'
    # img_data = scio.loadmat(path)['data']
    # img_data = np.random.random((1,64,64))
    # input_data = torch.FloatTensor(img_data).unsqueeze_(0)
    # print("start to calc using pytorch")
    # out = net(input_data).cpu().detach().numpy()

    # #validate in encrypted 
    # ct_img_data = encrypt_matrix(img_data,pk)
    # # ct_img_data = img_data
    # print("start to calc using encrypt")
    # res = net.forward_encrypt(ct_img_data,encrypt_conv)
    # print("calc using encrypt finished")

    # res = decrypt_matrix(res,sk)
    # print("result the same using two ways:",np.allclose(out,res))

    haha = np.random.random((2,20,20))
    kernel = np.random.random((2,2,3,3))
    enc_haha = encrypt_matrix(haha,pk)
    res1 = matrix_multiplication_for_conv2d(enc_haha,kernel,0,1,2)
    res = decrypt_matrix(res1,sk)
    
    haha = torch.tensor(haha)
    haha = haha.unsqueeze(0)
    kernel = torch.tensor(kernel)
    res2 = F.conv2d(haha,kernel,padding=2)
    # Release QAT engine control
    print(np.allclose(res,res2))

    context.terminateContext()