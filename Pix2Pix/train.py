import torch.nn as nn
import torch
from torchvision import transforms
import torchvision
from data import MyDataset
from model import PatchGAN ,Generator
'''
FOR SOURCE_IMG , TARGET_IMG IN DATALOADER:
    disc_preds < -- SOURCE_IMG , TARGET_IMG
    disc_real_loss <-- disc_preds with labels 1
    fake_targets <-- gen(SOURCE_IMG)
    disc_fake_loss <-- disc(SOURCE_IMG , fake_targets) with labels 0
    some loss adjustments including L1 Loss 
    disc.update_weights

    disc_fake_preds <-- disc(SOURCE_IMG,fake_targets)
    gen_loss <-- disc_fake_preds BCE with labels 1
    some loss adjustments including L1 Loss 
    gen.update_weights



'''
def d(i):
    return i*0.5 + 0.5
## ADD L1 LOSS
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
print(device)

disc = PatchGAN(3)
gen = Generator(3,3)
disc.to(device)
gen.to(device)
obj = MyDataset("C:/datasets/maps/train" ,transforms.Compose([transforms.ToTensor() ,
transforms.Normalize((0.5,0.5,0.5) ,(0.5,0.5,0.5))
]))
print(len(obj))
loader = torch.utils.data.DataLoader(obj ,batch_size=5)




disc_optim = torch.optim.Adam(disc.parameters() ,2e-4,)
gen_optim = torch.optim.Adam(gen.parameters() ,2e-4 ,)

disc_optim = torch.optim.Adam(disc.parameters() ,2e-4,betas=(0.5,0.999))
gen_optim = torch.optim.Adam(gen.parameters() ,2e-4 , betas=(0.5,0.999))

criterion = torch.nn.BCEWithLogitsLoss()
l1 = torch.nn.L1Loss()

for epoch in range(100):
    for idx , (source, target) in enumerate(loader):
        disc_optim.zero_grad()
        source = source.to(device)
        target = d(target).to(device)
        real_preds = disc(source, target)
        disc_real_loss = criterion(real_preds ,torch.ones_like(real_preds))
        fake_imgs = gen(source).detach()
        disc_fake_preds = disc(source ,fake_imgs)
        disc_fake_loss = criterion(disc_fake_preds ,torch.zeros_like(disc_fake_preds))
        total_loss = (disc_real_loss + disc_fake_loss) / 2
        total_loss.backward()
        disc_optim.step()

        ##
        gen_optim.zero_grad()
        #fake_imgs = gen(source)
        disc_fake_preds = disc(source ,fake_imgs)
        l = l1(fake_imgs ,target)
        gen_loss = criterion(disc_fake_preds ,torch.ones_like(disc_fake_preds)) + l*10
        gen_loss.backward()
        gen_optim.step()
        print(total_loss , gen_loss)
        if (idx+1) % 20 == 0:
            with torch.no_grad():
                g = torchvision.utils.make_grid(d(fake_imgs),nrow=5)
                g2 = torchvision.utils.make_grid((target),nrow=5)
                torchvision.utils.save_image(g ,"pix2pix_img.png")
                torchvision.utils.save_image(g2 ,"pix2pix_img_true.png")
            