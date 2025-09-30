import torch.nn as nn
import torch.nn.functional as F

dropout_value = 0.01
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Input Block 28  >>> 64
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            #nn.Dropout(dropout_value) #Step 8: Add Dropout at each layer
        ) # output_size = 26 >>> 62
        #RF=1+(3−1)×1=3
        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            #nn.Dropout(dropout_value) #Step 8: Add Dropout at each layer
        ) # output_size = 24. >>> 60
        #RF=3+(3−1)×1=5
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 12

         # TRANSITION BLOCK 1
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=8, kernel_size=(1, 1), padding=0, bias=False),
            #nn.BatchNorm2d(8),
            #nn.ReLU()
        ) # output_size = 12 >>> 58
        #RF=5+(3−1)×1=7
       
        
        #RF=7+(2−1)×1=8 (Stride of previous layer is 1)
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(dropout_value) #Step 8: Add Dropout at each layer
            
        ) # output_size = 10 >>> 29
        #RF=8+(1−1)×2=8 (Stride of pooling layer is 2)
        # CONVOLUTION BLOCK 2
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(dropout_value) #Step 8: Add Dropout at each layer
        ) # output_size = 8 >>> 27
        #RF=8+(3−1)×2=8+4=12
        self.pool2 = nn.MaxPool2d(2, 2) # output_size = 4

        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=8, kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Dropout(dropout_value) #Step 8: Add Dropout at each layer
        ) # output_size = 8 >>> 25
        #RF=12+(3−1)×2=12+4=16

        # OUTPUT BLOCK
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            #nn.Dropout(dropout_value) #Step 8: Add Dropout at each layer
        ) # output_size = 6 >>> 23
        #RF=16+(3−1)×2=20

        #Step 6: Add GAP
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1)
        ) # output_size = 1 >>> 19
        #RF = 20+(6-1)×2=  28

        #Step: 7 Increase model capacity. Add additional layers.
        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        ) # output_size = 6 >>> 21
        #RF=20+(1−1)×2=20

        #Final RF = 28
        
        #STEP 5: Add Dropout
        #self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.pool1(x)
        x = self.convblock3(x)
        #x = self.dropout(x) #STEP 5: Add Dropout        
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.pool2(x)
        x = self.convblock6(x)
        #x = self.dropout(x)  #STEP 5: Add Dropout
        x = self.convblock7(x)
        
        x = self.gap(x) #Step 6: Add GAP
        x = self.convblock8(x) #Step: 7 Increase model capacity. Add additional layers.
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=1)