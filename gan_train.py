import os, sys
sys.path.append(os.getcwd())

import time

import libs as lib
import libs.plot
from tensorboardX import SummaryWriter

from models.wgan import *
from models.checkers import *
from config import InvNetConfig

import torch
import torchvision
from torch import nn
from torch import autograd
from torchvision import transforms, datasets
from timeit import default_timer as timer
from matscidata import MatSciDataset
from fixedcircledata import CircleDataset
import torch.nn.init as init

config = InvNetConfig()

DATA_DIR = config.trainset_path
VAL_DIR = config.validset_path

IMAGE_DATA_SET = config.dataset
torch.cuda.set_device(config.gpu)

def load_dim(path_to_folder):
    """

    :param path_to_folder: data path to read the dataset
    :return: dimension of width
    """
    if IMAGE_DATA_SET == 'circle':
        dataset = CircleDataset(path_to_folder)
        dimension = dataset[0].shape[-1]
    else:
        raise Exception('Currently only allows circle')

    return dimension

if len(DATA_DIR) == 0:
    raise Exception('Please specify path to data directory!')

NUM_CIRCLE = 0
if config.dataset == 'circle':
    NUM_CIRCLE = 2
    CATEGORY = NUM_CIRCLE + 1
    CNTL_DIM = CATEGORY + NUM_CIRCLE*2
    CONDITIONAL = True

if NUM_CIRCLE == 0 and config.dataset == 'circle':
    raise Exception('Dataset is circle but NUM_CIRCLE == 0.')


RESTORE_MODE = config.restore_mode # if True, it will load saved model from OUT_PATH and continue to train
START_ITER = 0  # starting iteration
OUTPUT_PATH = config.output_path

DIM = load_dim(config.validset_path)
CRITIC_ITERS = config.critic_iter # How many iterations to train the critic for
GENER_ITERS = 1
N_GPUS = 1 # Number of GPUs
BATCH_SIZE = config.batch_size

END_ITER = config.end_iter # How many iterations to train for
LAMBDA = config.lambda_gp # Gradient penalty lambda hyperparameter
OUTPUT_DIM = DIM*DIM*CATEGORY # Number of pixels in each image
PJ_ITERS = config.proj_iter
C = 1/DIM           # Normalizing Factor for the centroid

centroid_fn = CentroidFunction(BATCH_SIZE, NUM_CIRCLE, DIM, DIM)  # BATCH SIZE, Number of CH, WIDTH, HEIGHT

def proj_loss(fake_data, real_data):
    """
    Fake data requires to be pushed from tanh range to [0, 1]
    """
    x_fake, y_fake = centroid_fn(fake_data)
    x_real, y_real = centroid_fn(real_data)
    centerError = torch.norm(C*x_fake - C*x_real) + torch.norm(C*y_fake - C*y_real)
    # radiusError = torch.abs(p1_fn(fake_data) - p1_fn(real_data))
    radiusError = torch.norm((p1_fn(fake_data) - p1_fn(real_data)))
    return centerError + radiusError

def weights_init(m):
    if isinstance(m, MyConvo2d): 
        if m.conv.weight is not None:
            if m.he_init:
                init.kaiming_uniform_(m.conv.weight)
            else:
                init.xavier_uniform_(m.conv.weight)
        if m.conv.bias is not None:
            init.constant_(m.conv.bias, 0.0)
    if isinstance(m, nn.Linear):
        if m.weight is not None:
            init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.constant_(m.bias, 0.0)

def load_data(path_to_folder):
    data_transform = transforms.Compose([
                 transforms.Resize(64),
                 transforms.CenterCrop(64),
                 transforms.ToTensor(),
                 transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])
                ])
    if IMAGE_DATA_SET == 'matsci':
        dataset = MatSciDataset(path_to_folder)
    elif IMAGE_DATA_SET == 'circle':
        dataset = CircleDataset(path_to_folder)
    else:
        dataset = datasets.ImageFolder(root=path_to_folder,transform=data_transform)
    dataset_loader = torch.utils.data.DataLoader(dataset,batch_size=BATCH_SIZE, shuffle=True, drop_last=True, pin_memory=True)
    return dataset_loader

def training_data_loader():
    return load_data(DATA_DIR)

def val_data_loader():
    return load_data(VAL_DIR)

def calc_gradient_penalty(netD, real_data, fake_data):
    alpha = torch.rand(BATCH_SIZE, 1)
    alpha = alpha.expand(BATCH_SIZE, int(real_data.nelement()/BATCH_SIZE)).contiguous()
    
    alpha = alpha.view(BATCH_SIZE, CATEGORY, DIM, DIM)
    alpha = alpha.to(device)
    
    fake_data = fake_data.view(BATCH_SIZE, CATEGORY, DIM, DIM)
    interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())

    interpolates = interpolates.to(device)
    interpolates.requires_grad_(True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.view(gradients.size(0), -1)                              
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty

def generate_image(netG, noise=None, lv=None):
    if noise is None:
        noise = gen_rand_noise()
    if CONDITIONAL:
        if lv is None:
            # locationX and locationY randomly picks the centroid of the generated circles for the tensorboard.
            # radius is calculated based on the area of the circle.
            # using the conversion with (1/DIM)^2 * pi * r^2 = "normalized area",
            # 'r' is based on the unit of pixel.
            locationX = 30.0/DIM*torch.ones(BATCH_SIZE, NUM_CIRCLE) + 66.0/DIM*torch.rand(BATCH_SIZE, NUM_CIRCLE)
            locationY = 30.0/DIM*torch.ones(BATCH_SIZE, NUM_CIRCLE) + 66.0/DIM*torch.rand(BATCH_SIZE, NUM_CIRCLE)
            radius = 0.12*torch.rand(BATCH_SIZE, NUM_CIRCLE) + 0.05
            lv = torch.cat((locationX, locationY, radius), dim=1)
            lv = lv.to(device)
    else:
        lv = None
    with torch.no_grad():
        noisev = noise
        lv_v = lv 
    samples = netG(noisev, lv_v)
    samples = torch.argmax(samples.view(BATCH_SIZE, CATEGORY, DIM, DIM), dim=1).unsqueeze(1)
    samples = (samples * 255/(CATEGORY))
    return samples

def gen_rand_noise():
    noise = torch.randn(BATCH_SIZE, 128)
    noise = noise.to(device)

    return noise

cuda_available = torch.cuda.is_available()
device = torch.device("cuda" if cuda_available else "cpu")
fixed_noise = gen_rand_noise() 

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

if RESTORE_MODE:
    aG = torch.load(OUTPUT_PATH + "generator.pt")
    aD = torch.load(OUTPUT_PATH + "discriminator.pt")
else:
    if CONDITIONAL:
        aG = GoodGenerator(64, DIM*DIM*CATEGORY, ctrl_dim=NUM_CIRCLE+4)      # +4 for the centroid.
        aD = GoodDiscriminator(64)
    else:
        aG = GoodGenerator(64, DIM*DIM*CATEGORY, ctrl_dim=0)
        aD = GoodDiscriminator(64)
aG.apply(weights_init)
aD.apply(weights_init)

LR = 1e-4
optimizer_g = torch.optim.Adam(aG.parameters(), lr=LR, betas=(0,0.9))
optimizer_d = torch.optim.Adam(aD.parameters(), lr=LR, betas=(0,0.9))
if CONDITIONAL:
    optimizer_pj = torch.optim.Adam(aG.parameters(), lr=LR, betas=(0, 0.9))
one = torch.FloatTensor([1])
mone = one * -1
aG = aG.to(device)
aD = aD.to(device)
one = one.to(device)
mone = mone.to(device)

writer = SummaryWriter()
#Reference: https://github.com/caogang/wgan-gp/blob/master/gan_cifar10.py
def train():
    dataloader = training_data_loader() 
    dataiter = iter(dataloader)
    for iteration in range(START_ITER, END_ITER):
        start_time = time.time()
        print("Iter: " + str(iteration))
        start = timer()
        #---------------------TRAIN G------------------------
        for p in aD.parameters():
            p.requires_grad_(False)  # freeze D

        gen_cost = None
        try:
            real_data = next(dataiter)
        except StopIteration:
            dataiter = iter(dataloader)
            real_data = dataiter.next()

        if CONDITIONAL:
            x_real, y_real = centroid_fn(real_data.to(device))
            x_real, y_real = x_real*C, y_real*C
            real_p1 = torch.cat((x_real, y_real, p1_fn(real_data.to(device))), dim=1)
            real_p1 = real_p1.to(device)
        else:
            real_p1 = None
        for i in range(GENER_ITERS):
            print("Generator iters: " + str(i))
            aG.zero_grad()
            noise = gen_rand_noise()
            noise.requires_grad_(True)
            fake_data = aG(noise, real_p1)
            gen_cost = aD(fake_data)
            gen_cost = gen_cost.mean()
            gen_cost.backward(mone)
            gen_cost = -gen_cost
        
        optimizer_g.step()
        end = timer()
        print(f'---train G elapsed time: {end - start}')
        print(fake_data.min(), real_data.min())
        if CONDITIONAL:
            #Projection steps
            pj_cost = None
            for i in range(PJ_ITERS):
                print('Projection iters: {}'.format(i))
                aG.zero_grad()
                noise = gen_rand_noise()
                noise.requires_grad=True
                fake_data = aG(noise, real_p1)
                pj_cost = proj_loss(fake_data.view(-1, CATEGORY, DIM, DIM), real_data.to(device))
                pj_cost = pj_cost.mean()
                pj_cost.backward()
                optimizer_pj.step()


        #---------------------TRAIN D------------------------
        for p in aD.parameters():  # reset requires_grad
            p.requires_grad_(True)  # they are set to False below in training G
        for i in range(CRITIC_ITERS):
            print("Critic iter: " + str(i))
            
            start = timer()
            aD.zero_grad()

            # gen fake data and load real data
            noise = gen_rand_noise()
            batch = next(dataiter, None)
            if batch is None:
                dataiter = iter(dataloader)
                batch = dataiter.next()
            #batch = batch[0] #batch[1] contains labels
            real_data = batch.to(device) #TODO: modify load_data for each loading
            #real_p1.to(device)
            with torch.no_grad():
                noisev = noise  # totally freeze G, training D
                if CONDITIONAL:
                    x_real, y_real = centroid_fn(real_data)
                    x_real, y_real = x_real*C, y_real*C
                    real_p1 = torch.cat((x_real, y_real, p1_fn(real_data)), dim=1)
                    real_p1 = real_p1.to(device)
                else:
                    real_p1 = None
            end = timer(); print(f'---gen G elapsed time: {end-start}')
            start = timer()
            fake_data = aG(noisev, real_p1).detach()
            end = timer(); print(f'---load real imgs elapsed time: {end-start}')
            start = timer()

            # train with real data
            disc_real = aD(real_data)
            disc_real = disc_real.mean()

            # train with fake data
            disc_fake = aD(fake_data)
            disc_fake = disc_fake.mean()

            # train with interpolates data
            gradient_penalty = calc_gradient_penalty(aD, real_data, fake_data)

            # final disc cost
            disc_cost = disc_fake - disc_real + gradient_penalty
            disc_cost.backward()
            w_dist = disc_fake  - disc_real
            
            optimizer_d.step()
            #------------------VISUALIZATION----------
            if i == CRITIC_ITERS-1:
                writer.add_scalar('data/disc_cost', disc_cost, iteration)
                writer.add_scalar('data/disc_fake', disc_fake, iteration)
                writer.add_scalar('data/disc_real', disc_real, iteration)
                writer.add_scalar('data/gradient_pen', gradient_penalty, iteration)
                if CONDITIONAL:
                    writer.add_scalar('data/p1_cost', pj_cost.cpu().detach(), iteration)

            end = timer(); print(f'---train D elapsed time: {end-start}')
        #---------------VISUALIZATION---------------------
        writer.add_scalar('data/gen_cost', gen_cost, iteration)

        lib.plot.plot(OUTPUT_PATH + 'time', time.time() - start_time)
        lib.plot.plot(OUTPUT_PATH + 'train_disc_cost', disc_cost.cpu().data.numpy())
        lib.plot.plot(OUTPUT_PATH + 'train_gen_cost', gen_cost.cpu().data.numpy())
        lib.plot.plot(OUTPUT_PATH + 'wasserstein_distance', w_dist.cpu().data.numpy())
        if iteration % 10 == 0:
            fake_2 = torch.argmax(fake_data.view(BATCH_SIZE, CATEGORY, DIM, DIM), dim=1).unsqueeze(1)
            fake_2 = (fake_2 * 255 / (CATEGORY))
            fake_2 = fake_2.int()
            fake_2 = fake_2.cpu().detach().clone()
            fake_2 = torchvision.utils.make_grid(fake_2, nrow=8, padding=2)
            writer.add_image('G/images', fake_2, iteration)
        if iteration % 10 == 0:
            val_loader = val_data_loader() 
            dev_disc_costs = []
            for _, images in enumerate(val_loader):
                imgs = torch.Tensor(images[0])
                imgs = imgs.to(device)
                with torch.no_grad():
                    imgs_v = imgs

                D = aD(imgs_v)
                _dev_disc_cost = -D.mean().cpu().data.numpy()
                dev_disc_costs.append(_dev_disc_cost)
            lib.plot.plot(OUTPUT_PATH + 'dev_disc_cost.png', np.mean(dev_disc_costs))
            lib.plot.flush()
            gen_images = generate_image(aG, fixed_noise)
            torchvision.utils.save_image(gen_images, OUTPUT_PATH + 'samples_{}.png'.format(iteration), nrow=8, padding=2)
            grid_images = torchvision.utils.make_grid(gen_images, nrow=8, padding=2)
            writer.add_image('images', grid_images, iteration)
    #----------------------Save model----------------------
            torch.save(aG, OUTPUT_PATH + "generator.pt")
            torch.save(aD, OUTPUT_PATH + "discriminator.pt")
        lib.plot.tick()

train()


