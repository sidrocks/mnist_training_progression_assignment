import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Input Block 28  
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU()
        ) # output_size = 26 
        #RF=1+(3−1)×1=3
        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU()
        ) # output_size = 24. 
        #RF=3+(3−1)×1=5
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 12
        #RF=5+(2−1)⋅1=6
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=8, kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU()
        ) # output_size = 12 
        #RF=6+(1−1)⋅2=6

        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU()
        ) # output_size = 10 
        #RF=6+(3−1)⋅2=10
        # CONVOLUTION BLOCK 2
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU()
        ) # output_size = 8 
        #RF=10+(3−1)⋅2=14
        #self.pool2 = nn.MaxPool2d(2, 2) # output_size = 4

        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU()
        ) # output_size = 8 
        #RF=14+(1−1)⋅2=14

        # OUTPUT BLOCK
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(20),
            nn.ReLU()
        ) # output_size = 6
        #RF=16+(3−1)×2=20

        #Step: 7 Increase model capacity. Add additional layers.
        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        ) # output_size = 6 
        #RF=14+(3−1)⋅2=18
        #Step 6: Add GAP
        self.gap = nn.Sequential(
            nn.AvgPool2d(6)
        ) # output_size = 1 >>> 19
        #RF=18+(6−1)⋅2=28
        #Final RF = 28
        #STEP 5: Add Dropout
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.pool1(x)
        x = self.convblock3(x)
        x = self.dropout(x) #STEP 5: Add Dropout        
        x = self.convblock4(x)
        x = self.convblock5(x)
        #x = self.pool2(x)
        x = self.convblock6(x)
        x = self.dropout(x)  #STEP 5: Add Dropout
        x = self.convblock7(x)
        x = self.convblock8(x) #Step: 7 Increase model capacity. Add additional layers.
        x = self.gap(x) #Step 6: Add GAP
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)