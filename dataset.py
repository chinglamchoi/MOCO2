import os
import torch.utils.data as data
from PIL import Image
from torchvision.transforms import Compose, Resize, RandomCrop, CenterCrop, RandomHorizontalFlip, ToTensor, Normalize, RandomApply, ColorJitter, RandomGrayscale
import random
import moco

class dataset_single(data.Dataset):
  def __init__(self, opts, setname, input_dim):
    self.dataroot = opts.dataroot
    images = os.listdir(os.path.join(self.dataroot, opts.phase + setname))
    self.img = [os.path.join(self.dataroot, opts.phase + setname, x) for x in images]
    self.size = len(self.img)
    self.input_dim = input_dim

    # setup image transformation
    normalize = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    if args.aug_plus:
      # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
      augmentation = [
        #transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        Resize((opts.resize_size, opts.resize_size), Image.BICUBIC),
        CenterCrop(opts.crop_size),
        RandomApply([ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        RandomGrayscale(p=0.2),
        RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.5),
        RandomHorizontalFlip(),
        ToTensor(),
        normalize
    ]
    else:
      # MoCo v1's aug: the same as InstDisc https://arxiv.org/abs/1805.01978
      augmentation = [
        #RandomResizedCrop(224, scale=(0.2, 1.)),
        Resize((opts.resize_size, opts.resize_size), Image.BICUBIC),
        CenterCrop(opts.crop_size),
        RandomGrayscale(p=0.2),
        ColorJitter(0.4, 0.4, 0.4, 0.4),
        RandomHorizontalFlip(),
        ToTensor(),
        normalize
    ]
    self.transforms = Compose(augmentation)
    print('%s: %d images'%(setname, self.size))
    return

  def __getitem__(self, index):
    data = self.load_img(self.img[index], self.input_dim)
    return data

  def load_img(self, img_name, input_dim):
    img = Image.open(img_name).convert('RGB')
    ff = moco.loader.TwoCropsTransform(self.transforms)
    q, k = ff(img) #, img
    if input_dim == 1:
      q = q[0, ...] * 0.299 + q[1, ...] * 0.587 + q[2, ...] * 0.114
      k = k[0, ...] * 0.299 + k[1, ...] * 0.587 + k[2, ...] * 0.114
      q = q.unsqueeze(0)
      k = k.unsqueeze(0)
    return [q, k]
    

  def __len__(self):
    return self.size

class dataset_unpair(data.Dataset):
  def __init__(self, opts):
    self.dataroot = opts.dataroot
    # A
    images_A = os.listdir(os.path.join(self.dataroot, opts.phase + 'A'))
    self.A = [os.path.join(self.dataroot, opts.phase + 'A', x) for x in images_A]

    # B
    images_B = os.listdir(os.path.join(self.dataroot, opts.phase + 'B'))
    self.B = [os.path.join(self.dataroot, opts.phase + 'B', x) for x in images_B]

    self.A_size = len(self.A)
    self.B_size = len(self.B)
    self.dataset_size = max(self.A_size, self.B_size)
    self.input_dim_A = opts.input_dim_a
    self.input_dim_B = opts.input_dim_b

    # setup image transformation
    transforms = [
      Resize((opts.resize_size, opts.resize_size), Image.BICUBIC),
      CenterCrop(opts.crop_size),
      RandomApply([ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
      RandomGrayscale(p=0.2),
      RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.5),
      RandomHorizontalFlip(),
    ]
    if opts.phase == 'train':
      transforms.append(RandomCrop(opts.crop_size))
    else:
      #ALERT: what to do!
      transforms.append(CenterCrop(opts.crop_size))
    if not opts.no_flip:
      transforms.append(RandomHorizontalFlip())
    transforms.append(ToTensor())
    transforms.append(Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
    self.transforms = Compose(transforms)
    print('A: %d, B: %d images'%(self.A_size, self.B_size))
    return

  def __getitem__(self, index):
    if self.dataset_size == self.A_size:
      data_A = self.load_img(self.A[index], self.input_dim_A)
      data_B = self.load_img(self.B[random.randint(0, self.B_size - 1)], self.input_dim_B)
    else:
      data_A = self.load_img(self.A[random.randint(0, self.A_size - 1)], self.input_dim_A)
      data_B = self.load_img(self.B[index], self.input_dim_B)
    return data_A, data_B #[q_A, k_A], [q_B, k_B]

  def load_img(self, img_name, input_dim):
    img = Image.open(img_name).convert('RGB')
    ff = moco.loader.TwoCropsTransform(self.transforms)
    q, k = ff(img)
    #img = self.transforms(img)
    if input_dim == 1:
      q = q[0, ...] * 0.299 + q[1, ...] * 0.587 + q[2, ...] * 0.114
      k = k[0, ...] * 0.299 + k[1, ...] * 0.587 + k[2, ...] * 0.114
      q = q.unsqueeze(0)
      k = k.unsqueeze(0)
    return [q, k]

  def __len__(self):
    return self.dataset_size
