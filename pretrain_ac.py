# From MOCO:
import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
import torch.nn.parallel #
import torch.backends.cudnn as cudnn #
import torch.distributed as dist #
import torch.optim
import torch.multiprocessing as mp #
import torch.utils.data
import torch.utils.data.distributed #
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import moco.loader
import moco.builder

import dataset
import torch
import torch.nn as nn
from torch.autograd import Variable
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as F

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--dataroot', type=str, default="data/cat2dog", metavar='DIR', help='path to dataset')
# architecture
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                            help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=600, type=int, metavar='N',
                            help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                            help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int, metavar='N',
                            help='total batch size of all GPUs on current node when '
                                 'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                            metavar='LR', help='initial learning rate', dest='lr') #0.03
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                            help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=1.9, type=float, metavar='M',
                            help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                            metavar='W', help='weight decay (default: 1e-4)',
                                                dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                            metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='models/checkpoint.pt', type=str, metavar='PATH',
                            help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=1, type=int,
                            help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int,
                            help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:2036', type=str,
                            help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                            help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                            help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                            help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', default=True, action='store_true',
                            help='Use multi-processing distributed training to launch '
                                 'N processes per node, which has N GPUs. This is the '
                                 'fastest way to use PyTorch for either single node or '
                                 'multi node data parallel training')
parser.add_argument('--moco-dim', default=8, type=int,
                            help='feature dimension (default: 128)')
parser.add_argument('--moco-k', default=65536, type=int,
                            help='queue size; number of negative keys (default: 65536)')
parser.add_argument('--moco-m', default=0.999, type=float,
                            help='moco momentum of updating key encoder (default: 0.999)')
parser.add_argument('--moco-t', default=0.07, type=float,
                            help='softmax temperature (default: 0.07)')
parser.add_argument('--mlp', action='store_true',
                            help='use mlp head')
parser.add_argument('--aug-plus', action='store_true',
                            help='use moco v2 data augmentation')
parser.add_argument('--cos', action='store_true',
                            help='use cosine lr schedule')

#opt.parser options
parser.add_argument("--phase", type=str, default="train")
parser.add_argument('--resize_size', type=int, default=256, help='resized image size for training')
parser.add_argument('--crop_size', type=int, default=216, help='cropped image size for training')
parser.add_argument('--input_dim_a', type=int, default=3, help='# of input channels for domain A')
parser.add_argument('--input_dim_b', type=int, default=3, help='# of input channels for domain B')
parser.add_argument("--mm", type=str, default="a", help="train content or attribute encoder")
parser.add_argument("--no_flip", action="store_true", help="specified if no flipping")

#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
print("Using", torch.cuda.device_count(), "devices")
####################################################################
#---------------------------- Encoders -----------------------------
####################################################################
class E_content(nn.Module):
  def __init__(self, input_dim_a, input_dim_b):
    super(E_content, self).__init__()
    encA_c = []
    tch = 64
    encA_c += [LeakyReLUConv2d(input_dim_a, tch, kernel_size=7, stride=1, padding=3)]
    for i in range(1, 3):
      encA_c += [ReLUINSConv2d(tch, tch * 2, kernel_size=3, stride=2, padding=1)]
      tch *= 2
    for i in range(0, 3):
      encA_c += [INSResBlock(tch, tch)]
        
    encB_c = []
    tch = 64
    encB_c += [LeakyReLUConv2d(input_dim_b, tch, kernel_size=7, stride=1, padding=3)]
    for i in range(1, 3):
      encB_c += [ReLUINSConv2d(tch, tch * 2, kernel_size=3, stride=2, padding=1)]
      tch *= 2
    for i in range(0, 3):
      encB_c += [INSResBlock(tch, tch)]
        
    enc_share = []
    for i in range(0, 1):
      enc_share += [INSResBlock(tch, tch)]
      enc_share += [GaussianNoiseLayer()]
      self.conv_share = nn.Sequential(*enc_share)
        
    self.convA = nn.Sequential(*encA_c)
    self.convB = nn.Sequential(*encB_c)
    
  def forward(self, xa, xb):
    outputA = self.convA(xa)
    outputB = self.convB(xb)
    outputA = self.conv_share(outputA)
    outputB = self.conv_share(outputB)
    return outputA, outputB
    
  def forward_a(self, xa):
    outputA = self.convA(xa)
    outputA = self.conv_share(outputA)
    return outputA
    
  def forward_b(self, xb):
    outputB = self.convB(xb)
    outputB = self.conv_share(outputB)
    return outputB

class E_attr(nn.Module):
  def __init__(self, input_dim_a, input_dim_b, output_nc=8):
    super(E_attr, self).__init__()
    dim = 64
    self.model_a = nn.Sequential(
        nn.ReflectionPad2d(3),
        nn.Conv2d(input_dim_a, dim, 7, 1),
        nn.ReLU(inplace=True),
        nn.ReflectionPad2d(1),
        nn.Conv2d(dim, dim*2, 4, 2),
        nn.ReLU(inplace=True),
        nn.ReflectionPad2d(1),
        nn.Conv2d(dim*2, dim*4, 4, 2),
        nn.ReLU(inplace=True),
        nn.ReflectionPad2d(1),
        nn.Conv2d(dim*4, dim*4, 4, 2),
        nn.ReLU(inplace=True),
        nn.ReflectionPad2d(1),
        nn.Conv2d(dim*4, dim*4, 4, 2),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d(1),
        nn.Conv2d(dim*4, output_nc, 1, 1, 0))
    self.model_b = nn.Sequential(
        nn.ReflectionPad2d(3),
        nn.Conv2d(input_dim_b, dim, 7, 1),
        nn.ReLU(inplace=True),
        nn.ReflectionPad2d(1),
        nn.Conv2d(dim, dim*2, 4, 2),
        nn.ReLU(inplace=True),
        nn.ReflectionPad2d(1),
        nn.Conv2d(dim*2, dim*4, 4, 2),
        nn.ReLU(inplace=True),
        nn.ReflectionPad2d(1),
        nn.Conv2d(dim*4, dim*4, 4, 2),
        nn.ReLU(inplace=True),
        nn.ReflectionPad2d(1),
        nn.Conv2d(dim*4, dim*4, 4, 2),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d(1),
        nn.Conv2d(dim*4, output_nc, 1, 1, 0))
    return

  def forward(self, xa, xb):
    xa = self.model_a(xa)
    xb = self.model_b(xb)
    output_A = xa.view(xa.size(0), -1)
    output_B = xb.view(xb.size(0), -1)
    return output_A, output_B

  def forward_a(self, xa):
    xa = self.model_a(xa)
    output_A = xa.view(xa.size(0), -1)
    return output_A

  def forward_b(self, xb):
    xb = self.model_b(xb)
    output_B = xb.view(xb.size(0), -1)
    return output_B

class E_attr_concat(nn.Module):
  def __init__(self, input_dim_a, input_dim_b, output_nc=8, norm_layer=None, nl_layer=None):
    super(E_attr_concat, self).__init__()

    ndf = 64
    n_blocks=4
    max_ndf = 4

    conv_layers_A = [nn.ReflectionPad2d(1)]
    conv_layers_A += [nn.Conv2d(input_dim_a, ndf, kernel_size=4, stride=2, padding=0, bias=True)]
    for n in range(1, n_blocks):
      input_ndf = ndf * min(max_ndf, n)  # 2**(n-1)
      output_ndf = ndf * min(max_ndf, n+1)  # 2**n
      conv_layers_A += [BasicBlock(input_ndf, output_ndf, norm_layer, nl_layer)]
    conv_layers_A += [nl_layer(), nn.AdaptiveAvgPool2d(1)] # AvgPool2d(13)
    self.fc_A = nn.Sequential(*[nn.Linear(output_ndf, output_nc)])
    self.fcVar_A = nn.Sequential(*[nn.Linear(output_ndf, output_nc)])
    self.conv_A = nn.Sequential(*conv_layers_A)

    conv_layers_B = [nn.ReflectionPad2d(1)]
    conv_layers_B += [nn.Conv2d(input_dim_b, ndf, kernel_size=4, stride=2, padding=0, bias=True)]
    for n in range(1, n_blocks):
      input_ndf = ndf * min(max_ndf, n)  # 2**(n-1)
      output_ndf = ndf * min(max_ndf, n+1)  # 2**n
      conv_layers_B += [BasicBlock(input_ndf, output_ndf, norm_layer, nl_layer)]
    conv_layers_B += [nl_layer(), nn.AdaptiveAvgPool2d(1)] # AvgPool2d(13)
    self.fc_B = nn.Sequential(*[nn.Linear(output_ndf, output_nc)])
    self.fcVar_B = nn.Sequential(*[nn.Linear(output_ndf, output_nc)])
    self.conv_B = nn.Sequential(*conv_layers_B)

  def forward(self, xa, xb):
    x_conv_A = self.conv_A(xa)
    conv_flat_A = x_conv_A.view(xa.size(0), -1)
    output_A = self.fc_A(conv_flat_A)
    outputVar_A = self.fcVar_A(conv_flat_A)
    x_conv_B = self.conv_B(xb)
    conv_flat_B = x_conv_B.view(xb.size(0), -1)
    output_B = self.fc_B(conv_flat_B)
    outputVar_B = self.fcVar_B(conv_flat_B)
    return output_A, outputVar_A, output_B, outputVar_B

  def forward_a(self, xa):
    x_conv_A = self.conv_A(xa)
    conv_flat_A = x_conv_A.view(xa.size(0), -1)
    output_A = self.fc_A(conv_flat_A)
    outputVar_A = self.fcVar_A(conv_flat_A)
    return output_A, outputVar_A

  def forward_b(self, xb):
    x_conv_B = self.conv_B(xb)
    conv_flat_B = x_conv_B.view(xb.size(0), -1)
    output_B = self.fc_B(conv_flat_B)
    outputVar_B = self.fcVar_B(conv_flat_B)
    return output_B, outputVar_B

####################################################################
#------------------------- Basic Functions -------------------------
####################################################################
def get_scheduler(optimizer, opts, cur_ep=-1):
  if opts.lr_policy == 'lambda':
    def lambda_rule(ep):
      lr_l = 1.0 - max(0, ep - opts.n_ep_decay) / float(opts.n_ep - opts.n_ep_decay + 1)
      return lr_l
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule, last_epoch=cur_ep)
  elif opts.lr_policy == 'step':
    scheduler = lr_scheduler.StepLR(optimizer, step_size=opts.n_ep_decay, gamma=0.1, last_epoch=cur_ep)
  else:
    return NotImplementedError('no such learn rate policy')
  return scheduler

def meanpoolConv(inplanes, outplanes):
  sequence = []
  sequence += [nn.AvgPool2d(kernel_size=2, stride=2)]
  sequence += [nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0, bias=True)]
  return nn.Sequential(*sequence)

def convMeanpool(inplanes, outplanes):
  sequence = []
  sequence += conv3x3(inplanes, outplanes)
  sequence += [nn.AvgPool2d(kernel_size=2, stride=2)]
  return nn.Sequential(*sequence)

def get_norm_layer(layer_type='instance'):
  if layer_type == 'batch':
    norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
  elif layer_type == 'instance':
    norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
  elif layer_type == 'none':
    norm_layer = None
  else:
    raise NotImplementedError('normalization layer [%s] is not found' % layer_type)
  return norm_layer

def get_non_linearity(layer_type='relu'):
  if layer_type == 'relu':
    nl_layer = functools.partial(nn.ReLU, inplace=True)
  elif layer_type == 'lrelu':
    nl_layer = functools.partial(nn.LeakyReLU, negative_slope=0.2, inplace=False)
  elif layer_type == 'elu':
    nl_layer = functools.partial(nn.ELU, inplace=True)
  else:
    raise NotImplementedError('nonlinearity activitation [%s] is not found' % layer_type)
  return nl_layer
def conv3x3(in_planes, out_planes):
  return [nn.ReflectionPad2d(1), nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=0, bias=True)]

def gaussian_weights_init(m):
  classname = m.__class__.__name__
  if classname.find('Conv') != -1 and classname.find('Conv') == 0:
    m.weight.data.normal_(0.0, 0.02)

####################################################################
#-------------------------- Basic Blocks --------------------------
####################################################################

## The code of LayerNorm is modified from MUNIT (https://github.com/NVlabs/MUNIT)
class LayerNorm(nn.Module):
  def __init__(self, n_out, eps=1e-5, affine=True):
    super(LayerNorm, self).__init__()
    self.n_out = n_out
    self.affine = affine
    if self.affine:
      self.weight = nn.Parameter(torch.ones(n_out, 1, 1))
      self.bias = nn.Parameter(torch.zeros(n_out, 1, 1))
    return
  def forward(self, x):
    normalized_shape = x.size()[1:]
    if self.affine:
      return F.layer_norm(x, normalized_shape, self.weight.expand(normalized_shape), self.bias.expand(normalized_shape))
    else:
      return F.layer_norm(x, normalized_shape)

class BasicBlock(nn.Module):
  def __init__(self, inplanes, outplanes, norm_layer=None, nl_layer=None):
    super(BasicBlock, self).__init__()
    layers = []
    if norm_layer is not None:
      layers += [norm_layer(inplanes)]
    layers += [nl_layer()]
    layers += conv3x3(inplanes, inplanes)
    if norm_layer is not None:
      layers += [norm_layer(inplanes)]
    layers += [nl_layer()]
    layers += [convMeanpool(inplanes, outplanes)]
    self.conv = nn.Sequential(*layers)
    self.shortcut = meanpoolConv(inplanes, outplanes)
  def forward(self, x):
    out = self.conv(x) + self.shortcut(x)
    return out

class LeakyReLUConv2d(nn.Module):
  def __init__(self, n_in, n_out, kernel_size, stride, padding=0, norm='None', sn=False):
    super(LeakyReLUConv2d, self).__init__()
    model = []
    model += [nn.ReflectionPad2d(padding)]
    if sn:
      model += [spectral_norm(nn.Conv2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=0, bias=True))]
    else:
      model += [nn.Conv2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=0, bias=True)]
    if 'norm' == 'Instance':
      model += [nn.InstanceNorm2d(n_out, affine=False)]
    model += [nn.LeakyReLU(inplace=True)]
    self.model = nn.Sequential(*model)
    self.model.apply(gaussian_weights_init)
    #elif == 'Group'
  def forward(self, x):
    return self.model(x)

class ReLUINSConv2d(nn.Module):
  def __init__(self, n_in, n_out, kernel_size, stride, padding=0):
    super(ReLUINSConv2d, self).__init__()
    model = []
    model += [nn.ReflectionPad2d(padding)]
    model += [nn.Conv2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=0, bias=True)]
    model += [nn.InstanceNorm2d(n_out, affine=False)]
    model += [nn.ReLU(inplace=True)]
    self.model = nn.Sequential(*model)
    self.model.apply(gaussian_weights_init)
  def forward(self, x):
    return self.model(x)

class INSResBlock(nn.Module):
  def conv3x3(self, inplanes, out_planes, stride=1):
    return [nn.ReflectionPad2d(1), nn.Conv2d(inplanes, out_planes, kernel_size=3, stride=stride)]
  def __init__(self, inplanes, planes, stride=1, dropout=0.0):
    super(INSResBlock, self).__init__()
    model = []
    model += self.conv3x3(inplanes, planes, stride)
    model += [nn.InstanceNorm2d(planes)]
    model += [nn.ReLU(inplace=True)]
    model += self.conv3x3(planes, planes)
    model += [nn.InstanceNorm2d(planes)]
    if dropout > 0:
      model += [nn.Dropout(p=dropout)]
    self.model = nn.Sequential(*model)
    self.model.apply(gaussian_weights_init)
  def forward(self, x):
    residual = x
    out = self.model(x)
    out += residual
    return out

class MisINSResBlock(nn.Module):
  def conv3x3(self, dim_in, dim_out, stride=1):
    return nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=stride))
  def conv1x1(self, dim_in, dim_out):
    return nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1, padding=0)
  def __init__(self, dim, dim_extra, stride=1, dropout=0.0):
    super(MisINSResBlock, self).__init__()
    self.conv1 = nn.Sequential(
        self.conv3x3(dim, dim, stride),
        nn.InstanceNorm2d(dim))
    self.conv2 = nn.Sequential(
        self.conv3x3(dim, dim, stride),
        nn.InstanceNorm2d(dim))
    self.blk1 = nn.Sequential(
        self.conv1x1(dim + dim_extra, dim + dim_extra),
        nn.ReLU(inplace=False),
        self.conv1x1(dim + dim_extra, dim),
        nn.ReLU(inplace=False))
    self.blk2 = nn.Sequential(
        self.conv1x1(dim + dim_extra, dim + dim_extra),
        nn.ReLU(inplace=False),
        self.conv1x1(dim + dim_extra, dim),
        nn.ReLU(inplace=False))
    model = []
    if dropout > 0:
      model += [nn.Dropout(p=dropout)]
    self.model = nn.Sequential(*model)
    self.model.apply(gaussian_weights_init)
    self.conv1.apply(gaussian_weights_init)
    self.conv2.apply(gaussian_weights_init)
    self.blk1.apply(gaussian_weights_init)
    self.blk2.apply(gaussian_weights_init)
  def forward(self, x, z):
    residual = x
    z_expand = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), x.size(2), x.size(3))
    o1 = self.conv1(x)
    o2 = self.blk1(torch.cat([o1, z_expand], dim=1))
    o3 = self.conv2(o2)
    out = self.blk2(torch.cat([o3, z_expand], dim=1))
    out += residual
    return out

class GaussianNoiseLayer(nn.Module):
  def __init__(self,):
    super(GaussianNoiseLayer, self).__init__()
  def forward(self, x):
    if self.training == False:
      return x
    noise = Variable(torch.randn(x.size()).cuda(x.get_device()))
    return x + noise

class ReLUINSConvTranspose2d(nn.Module):
  def __init__(self, n_in, n_out, kernel_size, stride, padding, output_padding):
    super(ReLUINSConvTranspose2d, self).__init__()
    model = []
    model += [nn.ConvTranspose2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=True)]
    model += [LayerNorm(n_out)]
    model += [nn.ReLU(inplace=True)]
    self.model = nn.Sequential(*model)
    self.model.apply(gaussian_weights_init)
  def forward(self, x):
    return self.model(x)


####################################################################
#--------------------- Spectral Normalization ---------------------
#  This part of code is copied from pytorch master branch (0.5.0)
####################################################################
class SpectralNorm(object):
  def __init__(self, name='weight', n_power_iterations=1, dim=0, eps=1e-12):
    self.name = name
    self.dim = dim
    if n_power_iterations <= 0:
      raise ValueError('Expected n_power_iterations to be positive, but '
                       'got n_power_iterations={}'.format(n_power_iterations))
    self.n_power_iterations = n_power_iterations
    self.eps = eps
  def compute_weight(self, module):
    weight = getattr(module, self.name + '_orig')
    u = getattr(module, self.name + '_u')
    weight_mat = weight
    if self.dim != 0:
      # permute dim to front
      weight_mat = weight_mat.permute(self.dim,
                                            *[d for d in range(weight_mat.dim()) if d != self.dim])
    height = weight_mat.size(0)
    weight_mat = weight_mat.reshape(height, -1)
    with torch.no_grad():
      for _ in range(self.n_power_iterations):
        v = F.normalize(torch.matmul(weight_mat.t(), u), dim=0, eps=self.eps)
        u = F.normalize(torch.matmul(weight_mat, v), dim=0, eps=self.eps)
    sigma = torch.dot(u, torch.matmul(weight_mat, v))
    weight = weight / sigma
    return weight, u
  def remove(self, module):
    weight = getattr(module, self.name)
    delattr(module, self.name)
    delattr(module, self.name + '_u')
    delattr(module, self.name + '_orig')
    module.register_parameter(self.name, torch.nn.Parameter(weight))
  def __call__(self, module, inputs):
    if module.training:
      weight, u = self.compute_weight(module)
      setattr(module, self.name, weight)
      setattr(module, self.name + '_u', u)
    else:
      r_g = getattr(module, self.name + '_orig').requires_grad
      getattr(module, self.name).detach_().requires_grad_(r_g)

  @staticmethod
  def apply(module, name, n_power_iterations, dim, eps):
    fn = SpectralNorm(name, n_power_iterations, dim, eps)
    weight = module._parameters[name]
    height = weight.size(dim)
    u = F.normalize(weight.new_empty(height).normal_(0, 1), dim=0, eps=fn.eps)
    delattr(module, fn.name)
    module.register_parameter(fn.name + "_orig", weight)
    module.register_buffer(fn.name, weight.data)
    module.register_buffer(fn.name + "_u", u)
    module.register_forward_pre_hook(fn)
    return fn

def spectral_norm(module, name='weight', n_power_iterations=1, eps=1e-12, dim=None):
  if dim is None:
    if isinstance(module, (torch.nn.ConvTranspose1d,
                           torch.nn.ConvTranspose2d,
                           torch.nn.ConvTranspose3d)):
      dim = 1
    else:
      dim = 0
  SpectralNorm.apply(module, name, n_power_iterations, dim, eps)
  return module

def remove_spectral_norm(module, name='weight'):
  for k, hook in module._forward_pre_hooks.items():
    if isinstance(hook, SpectralNorm) and hook.name == name:
      hook.remove(module)
      del module._forward_pre_hooks[k]
      return module
  raise ValueError("spectral_norm of '{}' not found in {}".format(name, module))



def main():
  least_loss = 1e10
  args = parser.parse_args()
  args.gpu = [0,1,2,3,4,5,6,7]
  if args.mm == "c":
    m = E_content(args.input_dim_a, args.input_dim_b)
  else:
    m = E_attr(args.input_dim_a, args.input_dim_b)
  model = moco.builder.MoCo(
    m,
    args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.mlp)
  device = torch.device("cuda:0")
  model = torch.nn.parallel.DataParallel(model, device_ids=args.gpu)
  model.to(device)
  criterion = nn.CrossEntropyLoss().cuda()
  optimizer = torch.optim.SGD(model.parameters(), args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay)
  if args.resume is not None:
    print("resuming")
    #if os.path.isfile(args.resume):
      #print("=> loading checkpoint '{}'".format(args.resume))
      #if args.gpu is None:
    checkpoint = torch.load(args.resume)
      #else:
        #loc = 'cuda:{}'.format(args.gpu)
        #checkpoint = torch.load(args.resume, map_location=loc)
    args.start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    #optimizer.load_state_dict(checkpoint['optimizer'])
    least_loss = checkpoint["loss"]
    print(least_loss)
      #print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
    #else:
      #print("=> no checkpoint found at '{}'".format(args.resume))
  #cudnn.benchmark = True

  # Data loading code
  train_dataset = dataset.dataset_unpair(args)
  train_sampler = None#train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
  train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
    num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

  for epoch in range(args.start_epoch, args.start_epoch+args.epochs):
    #if args.distributed:
      #train_sampler.set_epoch(epoch)
    adjust_learning_rate(optimizer, epoch, args)

    epoch_loss = train(0, train_loader, model, criterion, optimizer, epoch, args)
    print("Loss at epoch", epoch, "|", epoch_loss)
    bested = True if epoch_loss < least_loss else False
    least_loss = epoch_loss if bested else least_loss 
    if not args.multiprocessing_distributed or (args.multiprocessing_distributed
      and args.rank % 1 == 0):
      save_checkpoint(epoch, {
      'epoch': epoch + 1,
      #'arch': args.arch,
      'state_dict': model.state_dict(),
      'optimizer' : optimizer.state_dict(),
      "loss" : least_loss,
    }, is_best=bested, filename='models/checkpoint_sgd.pt')

def train(gpu, train_loader, model, criterion, optimizer, epoch, args):
  print("training")
  #batch_time = AverageMeter('Time', ':6.3f')
  #data_time = AverageMeter('Data', ':6.3f')
  losses = AverageMeter('Loss', ':.4e')
  top1 = AverageMeter('Acc@1', ':6.2f')
  top5 = AverageMeter('Acc@5', ':6.2f')
  #progress = ProgressMeter(
    #len(train_loader),
    #[batch_time, data_time, losses, top1, top5],
    #prefix="Epoch: [{}]".format(epoch))

  model.train()
  cnt, epoch_loss = 1, 0
  #end = time.time()
  for i, (images_A, images_B) in enumerate(train_loader):
    # measure data loading time
    #data_time.update(time.time() - end)
    if args.gpu is not None:
        images_A[0] = images_A[0].cuda(gpu, non_blocking=True)
        images_A[1] = images_A[1].cuda(gpu, non_blocking=True)
        images_B[0] = images_B[0].cuda(gpu, non_blocking=True)
        images_B[1] = images_B[1].cuda(gpu, non_blocking=True)
    # compute output
    output1, output2, target1, target2 = model(ima_q=images_A[0], ima_k=images_A[1], imb_q=images_B[0], imb_k=images_B[1])
    loss = criterion(output1, target1)
    loss.requires_grad = True
    loss = loss + criterion(output2, target2)
    loss = loss/2
    acc1_a, acc5_a = accuracy(output1, target1, topk=(1, 5))
    acc1_b, acc5_b = accuracy(output2, target2, topk=(1, 5))
    losses.update(loss.item(), images_A[0].size(0)) #ALERT!!
    acc1, acc5 = (acc1_a +acc1_b)/2, (acc5_a + acc5_b)/2
    top1.update(acc1[0], images_A[0].size(0))
    top5.update(acc5[0], images_A[0].size(0))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    cnt += 1
    epoch_loss += loss.item()
  return epoch_loss/cnt

    # measure elapsed time
    #batch_time.update(time.time() - end)
    #end = time.time()

    #if i % args.print_freq == 0:
      #progress.display(i)

def save_checkpoint(epoch, state, is_best, filename="models/checkpoint_sgd.pt"):
  if is_best:
    torch.save(state, filename)
    print("Best at epoch:", epoch)
    #shutil.copyfile(filename, 'model_best.pt')

class AverageMeter(object):
  def __init__(self, name, fmt=':f'):
    self.name = name
    self.fmt = fmt
    self.reset()
  def reset(self):
    self.val, self.avg, self.sum, self.count = 0, 0, 0, 0
  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count
  def __str__(self):
    fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
    return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
  def __init__(self, num_batches, meters, prefix=""):
    self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
    self.meters = meters
    self.prefix = prefix
  def display(self, batch):
    entries = [self.prefix + self.batch_fmtstr.format(batch)]
    entries += [str(meter) for meter in self.meters]
    print('\t'.join(entries))
  def _get_batch_fmtstr(self, num_batches):
    num_digits = len(str(num_batches // 1))
    fmt = '{:' + str(num_digits) + 'd}'
    return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def adjust_learning_rate(optimizer, epoch, args):
  """Decay the learning rate based on schedule"""
  lr = args.lr
  if args.cos: #cosine lr schedule
    lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
  else:
    for milestone in args.schedule:
      lr *= 0.1 if epoch >= milestone else 1.
  for param_group in optimizer.param_groups:
    param_group['lr'] = lr

def accuracy(output, target, topk=(1,)):
  """Computes the accuracy over the k top predictions for the specified values of k"""
  with torch.no_grad():
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
      correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
      res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__ == "__main__":
  main()
