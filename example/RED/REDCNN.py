import torch
import torch.nn as nn
# from torchsummary import summary

class RED_CNN(nn.Module):
    def __init__(self, out_ch=96):
        super(RED_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, out_ch, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv3 = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv4 = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv5 = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=2, bias=False)

        self.tconv1 = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=2, bias=False)
        self.tconv2 = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=2, bias=False)
        self.tconv3 = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=2, bias=False)
        self.tconv4 = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=2, bias=False)
        self.tconv5 = nn.Conv2d(out_ch, 1, kernel_size=5, stride=1, padding=2, bias=False)

        # self.relu = nn.ReLU()

    def forward(self, x):
        # encoder
        residual_1 = x
        # out = self.relu(self.conv1(x))
        # out = self.relu(self.conv2(out))
        out = self.conv1(x)
        out = self.conv2(out)

        residual_2 = out
        # out = self.relu(self.conv3(out))
        # out = self.relu(self.conv4(out))
        out = self.conv3(out)
        out = self.conv4(out)

        residual_3 = out
        # out = self.relu(self.conv5(out))
        out = self.conv5(out)
        # decoder
        ####这里改了，提取多个输出
        out = self.tconv1(out)
        out += residual_3
        # out = self.tconv2(self.relu(out))
        # out = self.tconv3(self.relu(out))
        out = self.tconv2(out)
        out = self.tconv3(out)
        out += residual_2
        # out = self.tconv4(self.relu(out))
        # out = self.tconv5(self.relu(out))
        out = self.tconv4(out)
        out = self.tconv5(out)
        out += residual_1
        # out = self.relu(out)
        return out

    def forward_encrypt(self, x,make_conv):
        with torch.no_grad():
            # encoder
            residual_1 = x
            # out = self.relu(self.conv1(x))
            # out = self.relu(self.conv2(out))
            out = make_conv(x,self.conv1)
            out = make_conv(out,self.conv2)

            residual_2 = out
            # out = self.relu(self.conv3(out))
            # out = self.relu(self.conv4(out))
            out = make_conv(out,self.conv3)
            out = make_conv(out,self.conv4)

            residual_3 = out
            # out = self.relu(self.conv5(out))
            out = make_conv(out,self.conv5)
            # decoder
            ####这里改了，提取多个输出
            out = make_conv(out,self.tconv1)
            out += residual_3
            # out = self.tconv2(self.relu(out))
            # out = self.tconv3(self.relu(out))
            out = make_conv(out,self.tconv2)
            out = make_conv(out,self.tconv3)
            out += residual_2
            # out = self.tconv4(self.relu(out))
            # out = self.tconv5(self.relu(out))
            out = make_conv(out,self.tconv4)
            out = make_conv(out,self.tconv5)
            out += residual_1
            # out = self.relu(out)
            return out


if __name__ == '__main__':
    red = RED_CNN()
    red.cuda()
    # summary(red, input_size=(1, 128, 128))