from re import S
from telnetlib import SE
import torch
import torch.nn as nn
            
class score(nn.Module):
    def __init__(self, in_channel, out_channel, n):
        super(score, self).__init__()
        self.conv1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=5, stride=5),
            nn.Conv2d(in_channel, 64, 5),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(n, out_channel),
            # nn.BatchNorm1d(out_channel),
            # nn.ReLU(),
            # nn.Linear(128, out_channel),
        )
    def forward(self, inputs):
        b = inputs.shape[0]
        inputs = self.conv1(inputs)
        inputs_pool = inputs.reshape(b, -1)
        x = self.fc(inputs_pool)
        return x
        

class model(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(model, self).__init__()
        self.in_channel = in_channel
        self.out_channel = [out_channel] if not isinstance(out_channel, list) else out_channel

        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel[0], 3, padding=1),
            nn.BatchNorm2d(out_channel[0]),
            nn.ReLU()
        )

        self.stage2 = nn.Sequential(
            nn.Conv2d(out_channel[0], out_channel[1], 3, padding=1),
            nn.BatchNorm2d(out_channel[1]),
            nn.ReLU()
        )

        self.stage3 = nn.Sequential(
            nn.Conv2d(out_channel[1], out_channel[2], 3, padding=1),
        )
        
        self.score1 = score(2*out_channel[0], 64, 64*76*76)
        self.score2 = score(2*out_channel[1], 64, 64*76*76)
        self.score3 = score(2*out_channel[2], 64, 64*76*76)
        
        self.pre = nn.Sequential(
            nn.Linear(192, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 2),
            # nn.BatchNorm1d(64),
            # nn.ReLU(),
            # nn.Linear(64, 2)
        )
    def forward(self, inputs):
        B, C, M, N = inputs.shape # bitch size c=428
        inputs1, inputs2 = inputs[:,:C//2, :, :], inputs[:, C//2:, :, :]
        inputs = torch.cat([inputs1, inputs2], 0) # 2b, m, n, 214
     
        features1 = self.stage1(inputs)
        features2 = self.stage2(features1)
        features3 = self.stage3(features2)

        # distance
        distance = []
        f1, f2 = torch.chunk(features1, 2, 0)
        f = torch.cat([f1, f2], 1)
        dist = self.score1(f)
        distance.append(dist)
        
        f1, f2 = torch.chunk(features2, 2, 0)
        f = torch.cat([f1, f2], 1)
        dist = self.score2(f)
        distance.append(dist)
        
        f1, f2 = torch.chunk(features3, 2, 0)
        f = torch.cat([f1, f2], 1)
        dist = self.score3(f)
        distance.append(dist)
    
        distance = torch.cat(distance, 1)
        
        pre = self.pre(distance)
        return pre, features3.squeeze(1)
        
        
            
        
        
if __name__=='__main__':
    x = torch.rand([2, 400, 400, 428]).cuda()
    model = model(214, [1]).cuda()
    x = model(x)
            
        
            

            
        
            
        
        