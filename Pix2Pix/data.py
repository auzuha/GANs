import torch.nn as nn
import torch
import os
from PIL import Image
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt

class MyDataset(torch.utils.data.Dataset):
    def __init__(self,PATH,transform):
        super(MyDataset,self).__init__()
        self.root_dir = PATH
        self.list_files = os.listdir(self.root_dir)
        self.t = transform
    def __len__(self):
        return len(self.list_files)
    def __getitem__(self,index):
        data = self.list_files[index]
        img_path = os.path.join(self.root_dir ,data)
        img = Image.open(img_path)
        source = img.crop((0,0,600,600))
        target = img.crop((601,0,1200,600))
        source = source.resize((256,256))
        target = target.resize((256,256))
        source = self.t(source)
        target = self.t(target)
        return source ,target


if __name__=='__main__':
    o = MyDataset("C:/datasets/maps/train" , transforms.Compose([
        transforms.ToTensor()
    ]))

    loader = torch.utils.data.DataLoader(o,10)

    for x,y in loader:
        print(x.shape ,y.shape)
        plt.imshow(y[0].permute(1,2,0).detach().numpy())
        plt.show()
        break