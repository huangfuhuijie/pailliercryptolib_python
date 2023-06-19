
from ipcl_python import PaillierKeypair, context, hybridControl, hybridMode,PaillierEncryptedNumber
import pickle as pkl
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
import os
import multiprocessing as mp
mp.set_start_method("spawn",force=True)
import cv2 as cv

from utils import *


def visual():
    # generate Paillier scheme key pair
    pk, sk = PaillierKeypair.generate_keypair(200)
    # Acquire QAT engine control
    context.initializeContext("QAT")
    # net = RED_CNN()
    # net.load_state_dict(torch.load('./example/checkpoint/epoch_99_loss_0.008091.pth'))
    # net = RED_CNN_last_relu()
    # net.load_state_dict(torch.load('./example/checkpoint/best_param_last_relu.pth'))
    net = RED_CNN_all_relu()
    net.load_state_dict(torch.load('./example/checkpoint/best_param_all_relu.pth'))

    # data preparation
    path = './example/RED/data_0036.mat'
    img_data = scio.loadmat(path)['data']
    img_data = np.expand_dims(img_data, axis=0)
    # img_data = np.random.random((1,4,4))
    ori_shape = img_data.shape

    # validate in torch
    input_data = torch.FloatTensor(img_data).unsqueeze_(0)
    print("start to calc using pytorch")
    out = net(input_data).cpu().detach().numpy().squeeze(0)
    del input_data

    #validate in encrypted 
    #Estimated completion time is 900 times the time of the first layer
    ct_img_data = encrypt_matrix(img_data,pk)
    tr_ct_img_data = np.array([[[float(data2.ciphertext().getTexts()[0].__str__())for data2 in data1]for data1 in data] for data in ct_img_data])
    tr_ct_img_data = tr_ct_img_data/np.max(tr_ct_img_data)
    ct_out_data = encrypt_matrix(out,pk)
    tr_ct_out_data = np.array([[[float(data2.ciphertext().getTexts()[0].__str__())for data2 in data1]for data1 in data] for data in ct_out_data])
    tr_ct_out_data = tr_ct_out_data/np.max(tr_ct_out_data)

    save_plts(2,2,"all_relu.png",img_data.transpose(1,2,0),out.transpose(1,2,0),tr_ct_img_data.transpose(1,2,0),tr_ct_out_data.transpose(1,2,0),gray=True)

    

    context.terminateContext()

if __name__ == "__main__":
    
    visual()