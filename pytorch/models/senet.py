import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super(SEBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)
    
    def forward(self, x):
        b, c, _, _ = x.size()
        # print(f'before pool: {x.size()}')
        y = self.global_avg_pool(x)
        # print(f'after pool: {y.size()}')
        y = y.view(b,c)
        # y = self.view(b,c)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(b, c, 1, 1)
        return x * y
    

class SENet(nn.Module):
    def __init__(self):
        super(SENet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(12,1), stride=(2,1), padding=(0,0))
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=(2,1), stride=(3,1))
        
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(7,1), stride=(2,1), padding=(0,0))
        
        self.se1 = SEBlock(channels=8)
        self.bn2 = nn.BatchNorm2d(8)
        self.pool2 = nn.MaxPool2d(kernel_size=(2,1), stride=(2,1))
        
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(10,1), stride=(1,1), padding=(0,0))
        
        self.se2 = SEBlock(channels=8)
        self.pool3 = nn.MaxPool2d(kernel_size=(2,1), stride=(2,1))
        
        
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=0.1)
        
        self.fc1 = nn.Linear(160,20)
        self.fc2 = nn.Linear(20, 10)
        self.output = nn.Linear(10, 2)


    def forward(self, x):
        x = self.conv1(x)
        # x = self.dpconv1(x)
        x = F.relu(x)
        x = self.bn1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        # x = self.dpconv2(x)
        x = F.relu(x)
        x = self.se1(x)
        x = self.bn2(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        # x = self.dpconv3(x)
        x = F.relu(x)
        x = self.se2(x)
        x = self.pool3(x)
        
        # x = x.permute(0, 2, 1)  # Permute to (batch, seq_len, features) for LSTM
        # x, _ = self.lstm(x)
        x = self.flatten(x)
        
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.output(x), dim=1)
        
        return x

# # 示例输入：batch_size=16, 信号通道数=1, 序列长度=1000
# ecg_input = torch.randn(32, 1, 1250, 1)
# model = SENet()
# output = model(ecg_input)
# print(output.shape)  # 输出：(16, 7)