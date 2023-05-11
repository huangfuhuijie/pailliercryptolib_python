import numpy as np
from phe import paillier
from RED.REDCNN import *
import copy

def encrypt_vector(public_key, parameters):
    # parameters = parameters.flatten(0).cpu().numpy().tolist()
    parameters = parameters.tolist()
    # print(parameters)
    parameters = [public_key.encrypt(parameter) for parameter in parameters]
    return parameters

def decrypt_vector(private_key, parameters):
    parameters = [private_key.decrypt(parameter) for parameter in parameters]
    return parameters


public_key, private_key = paillier.generate_paillier_keypair(n_length=64)



import torch
import scipy.io as scio
from matplotlib import image as image, pyplot as plt
net = RED_CNN().double()
net.load_state_dict(torch.load('./checkpoint/epoch_49_loss_0.017239.pth'))

path = './data_0001.mat'
img_data = scio.loadmat(path)['data']

# input_data = torch.FloatTensor(img_data).unsqueeze_(0).unsqueeze_(0)
# out = net(input_data)
# img = out[0,0,:,:]
# plt.imshow(img.cpu().detach().numpy(),cmap='gray')

enc_img_data = encrypt_vector(public_key, img_data.flatten())

haha = np.array([1],dtype=int)
# haha = public_key.encrypt(haha)
haha = encrypt_vector(public_key,haha)
cc = haha * 5
haha2 = decrypt_vector(private_key,cc)

haha._EncryptedNumber__ciphertext *= 5
# haha._EncryptedNumber__ciphertext += 2
# haha._EncryptedNumber__ciphertext = int(haha._EncryptedNumber__ciphertext)
haha = private_key.decrypt(haha)

haha = np.array(haha,dtype=np.float64)
haha.resize((256,256))
haha = torch.DoubleTensor(haha)
haha.unsqueeze_(0)
haha.unsqueeze_(0)
# haha = haha.double()
out_ = net(haha)


out_2 = out_[0,0,:,:]
out_2 = out_2.detach().cpu().numpy()
out_2 = out_2.flatten()

enc_img_2 = copy.deepcopy(enc_img_data)
for i in range(enc_img_2.__len__()):
    enc_img_2[i]._EncryptedNumber__ciphertext=int(out_2[i])
# enc_img_2 = [data._EncryptedNumber__ciphertext ]
dec_img = decrypt_vector(private_key,enc_img_2)
aa = np.array(dec_img)
aa.resize((256,256))
plt.imshow(aa,cmap='gray')
plt.show()
print('1')
# datas = MayoDataSets()