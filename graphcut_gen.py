"""
Datafree Circle Generator
"""

import os, sys
sys.path.append(os.getcwd())

import numpy as np
import matplotlib
from matplotlib import pyplot as pyplot
matplotlib.use('Agg')
import argparse


import torch
from torch import nn
from torch.nn import functional as F
import torch.nn.init as init
import torchvision
import libs as lib
import libs.plot

from models.wgan import *
from models.checkers import *
from torch_submod.graph_cuts import TotalVariation2dWeighted as tv2dw
from tensorboardX import SummaryWriter

#from gan_train import weights_init

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
class GraphCut(nn.Module):
    """
    Graphcut layer that predicts log of
    weights given an input image
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64 , 3, padding=1)
        self.conv2 = nn.Conv2d(64, 1, 3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        out = self.conv2(x)
        return out
        

def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--outdir', help='Output directory', default='./')
    parser.add_argument('-a', '--area', nargs='+', type=float, default=[0.5, 0.7])
    parser.add_argument('-gs', '--gensteps', help='Gen update steps', type=int, default=1)
    parser.add_argument('-tv', '--tvsteps', help='TV reg steps (unused for now)', type=int, default=5)
    parser.add_argument('-e', '--epochs', type=int, default=100000)
    parser.add_argument('-bs','--batchsize', type=int, default=32)
    return parser

def get_next_run(outdir):
    ll = os.listdir(outdir)
    ll = [i for i in ll if 'run' in i]
    #print(ll)
    if len(ll) == 0:
        path = os.path.join(outdir, 'run_000')
        os.makedirs(path)
    else:
        idxs = sorted([int(d.split('_')[-1]) for d in ll])
        path_idx = idxs[-1] + 1
        path = os.path.join(outdir, 'run_{:03}'.format(path_idx))
        os.makedirs(path)
    return path

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

def main():
    parser = build_parser()
    args = parser.parse_args()

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    
    outdir = get_next_run(args.outdir)

    # Setup Generator and TV Regularizer
    gen = GoodGenerator(64, 64*64*1, ctrl_dim=1)
    graphcut_l = GraphCut()

    gen = gen.to(device)
    gen.apply(weights_init)
    graphcut_l = graphcut_l.to(device)

    opt_g = torch.optim.Adam(list(gen.parameters()) + list(graphcut_l.parameters()), lr=3e-4)
    #opt_gcl = torch.optim.Adam(graphcut_l.parameters(), lr=3e-4)
    
    writer = SummaryWriter(outdir)
    ### Training
    for epoch in range(args.epochs):
        z = torch.randn(args.batchsize, 128)
        p1 = torch.rand(args.batchsize, 1) * (args.area[1] - args.area[0]) + args.area[0]
        z = z.to(device)
        p1 = p1.to(device)

        gen_imgs = gen(z, p1)#.view(args.batchsize, 1, 64, 64)
        #print(p1_fn(gen_imgs, True), p1)
        p1_loss = torch.norm(p1_fn(gen_imgs, True) - p1).mean()
        gen_regs = []
        #for i in range(args.batchsize):
        w_img = graphcut_l(gen_imgs)
        w_row = torch.exp(w_img[:,:,:, :-1])
        w_col = torch.exp(w_img[:,:,:-1,:])
        #print(w_row.shape, w_col.shape)
        tmp_gen_reg = tv2dw.apply(gen_imgs[0, 0, ...], w_row[0,0,...], w_col[0,0,...])

        #graph_loss = 0.5 * torch.pow(tmp_gen_reg.to(device).unsqueeze(0).unsqueeze(1) - gen_imgs, 2).mean()
        graph_loss = -torch.norm(tmp_gen_reg.to(device).unsqueeze(0).unsqueeze(1) - gen_imgs)
        #graph_loss = torch.norm(tmp_gen_reg.to(device))
        loss_val = p1_loss + graph_loss
        #print(p1_loss.shape)
        opt_g.zero_grad()
        loss_val.backward()
        opt_g.step()
        print('Epoch: {}, p1_loss:{:.03f}, graph_loss:{:.03f}'.format(epoch, p1_loss.item(), graph_loss.item()))
        if epoch%10 == 0:
            writer.add_scalar('p1_loss', p1_loss, epoch)
            writer.add_scalar('graphcut loss', graph_loss, epoch)
            gen.eval()
            with torch.no_grad():
                z = torch.randn(args.batchsize, 128)
                p1 = torch.rand(args.batchsize, 1) * (args.area[1] - args.area[0]) + args.area[0]

                z = z.to(device)
                p1 = p1.to(device)
                out_img = gen(z, p1)
                out_grid = torchvision.utils.make_grid(out_img, out_img, epoch)
                #print(out_grid.min(), out_grid.max())
                writer.add_image('out_img', out_grid)

if __name__ == '__main__':
    main()

