import torch
import torch.nn as nn



class PatchGAN(nn.Module):
    def __init__(self,input_channels):
        super(PatchGAN,self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=input_channels*2,out_channels=64,kernel_size=4,stride=2,padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(in_channels=512,out_channels=1,kernel_size=1),
            nn.Sigmoid()
            )

    def forward(self,x,y):
        out = torch.cat((x,y),axis=1)
        return self.model(out)



class Generator(nn.Module):
    def __init__(self, in_channels ,out_channels):
        """
        Paper details:
        - Encoder: C64-C128-C256-C512-C512-C512-C512-C512
        - All convolutions are 4×4 spatial filters applied with stride 2
        - Convolutions in the encoder downsample by a factor of 2
        - Decoder: CD512-CD1024-CD1024-C1024-C1024-C512 -C256-C128
        """
        super(Generator,self).__init__()
        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels=3 ,out_channels=64 ,kernel_size=4 ,stride=2 ,padding=1 ,padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels=64 ,out_channels=64*2 ,kernel_size=4,stride=2,padding=1,padding_mode='reflect'),
            nn.BatchNorm2d(64*2),
            nn.LeakyReLU(0.2)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(in_channels=64*2 ,out_channels=64*4 ,kernel_size=4,stride=2,padding=1,padding_mode='reflect'),
            nn.BatchNorm2d(64*4),
            nn.LeakyReLU(0.2)
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(in_channels=64*4 ,out_channels=64*8 ,kernel_size=4,stride=2,padding=1,padding_mode='reflect'),
            nn.BatchNorm2d(64*8),
            nn.LeakyReLU(0.2)
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(in_channels=64*8 ,out_channels=64*8 ,kernel_size=4,stride=2,padding=1,padding_mode='reflect'),
            nn.BatchNorm2d(64*8),
            nn.LeakyReLU(0.2)
        )
        self.down5 = nn.Sequential(
            nn.Conv2d(in_channels=64*8 ,out_channels=64*8 ,kernel_size=4,stride=2,padding=1,padding_mode='reflect'),
            nn.BatchNorm2d(64*8),
            nn.LeakyReLU(0.2)
        )
        self.down6 = nn.Sequential(
            nn.Conv2d(in_channels=64*8 ,out_channels=64*8 ,kernel_size=4,stride=2,padding=1,padding_mode='reflect'),
            nn.BatchNorm2d(64*8),
            nn.LeakyReLU(0.2)
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(64*8 ,64*8 ,4,2,1),
            nn.ReLU()
        )

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(64*8 ,64*8 ,4,2,1,bias=False),
            nn.BatchNorm2d(64*8),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(64*8*2 ,64*8 ,4,2,1,bias=False),
            nn.BatchNorm2d(64*8),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )
        
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(64*8*2 ,64*8 ,4,2,1,bias=False),
            nn.BatchNorm2d(64*8),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(64*8*2 ,64*8 ,4,2,1,bias=False),
            nn.BatchNorm2d(64*8),
            nn.ReLU(),
        
        )
        self.up5 = nn.Sequential(
            nn.ConvTranspose2d(64*8*2 ,64*4 ,4,2,1,bias=False),
            nn.BatchNorm2d(64*4),
            nn.ReLU(),
        )
        self.up6 = nn.Sequential(
            nn.ConvTranspose2d(64*4*2 ,64*2 ,4,2,1,bias=False),
            nn.BatchNorm2d(64*2),
            nn.ReLU(),
        )
        self.up7 = nn.Sequential(
            nn.ConvTranspose2d(64*4 ,64 ,4,2,1,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(64*2 , 3,4,2,1),
            nn.Tanh()
        )
    def forward(self,x):
        d1 = self.initial_down(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        d6 = self.down5(d5)
        d7 = self.down6(d6)
        bottleneck = self.bottleneck(d7)
        up1 = self.up1(bottleneck)
        up2 = self.up2(torch.cat([up1,d7] ,1))
        up3 = self.up3(torch.cat([up2,d6] ,1))
        up4 = self.up4(torch.cat([up3,d5] ,1))
        up5 = self.up5(torch.cat([up4,d4] ,1))
        up6 = self.up6(torch.cat([up5,d3] ,1))
        up7 = self.up7(torch.cat([up6,d2] ,1))
        return self.final_up(torch.cat([up7,d1] ,1))
if __name__ == '__main__':
    obj = PatchGAN(3)
    dim = 256
    source = torch.randn((10,3,dim,dim))
    target = torch.randn((10,3,dim,dim))

    print(obj(source,target).shape)