# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn


class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07, mlp=False):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder
        self.encoder_k = base_encoder

        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        #print(x.shape) # [32, 3, 216, 216]
        print(torch.distributed.get_world_size())
        batch_size_this = x.shape[0]
        print("s1")
        x_gather = concat_all_gather(x)
        print("s2")
        #print(x_gather.shape) # [32, 3, 216, 216]
        batch_size_all = x_gather.shape[0]
        print("s3")
        num_gpus = batch_size_all // batch_size_this
        print(num_gpus)
        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x[0].shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        gpu_idx += 1
        print(gpu_idx)
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, ima_q, ima_k, imb_q, imb_k):
        """
        Input:
            im_q: a batch of query images (A, B)
            im_k: a batch of key images (A, B)
        Output:
            logits, targets
        """

        # compute query features
        q = self.encoder_q(ima_q, imb_q)  # queries: NxC
        print("f1")
        qq = torch.stack((q[0], q[1]))
        print("f2")
        q = nn.functional.normalize(qq, dim=1)
        print("f3")

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder
            print("f4")
            # shuffle for making use of BN
            ima_k, idx_unshuffle1 = self._batch_shuffle_ddp(ima_k)
            print("f5")
            imb_k, idx_unshuffle2 = self._batch_shuffle_ddp(imb_k)
            print("f6")
            k1, k2 = self.encoder_k(ima_k, imb_k)  # keys: NxC
            print("f7")
            k1 = nn.functional.normalize(k1, dim=1)
            print("f8")
            k2 = nn.functional.normalize(k2, dim=1)
            print("f9")
            # undo shuffle
            k1 = self._batch_unshuffle_ddp(k1, idx_unshuffle1) #ALERT!!
            print("f10")
            k2 = self._batch_unshuffle_ddp(k2, idx_unshuffle2)
            print("f11")
            k = torch.stack((k1, k2))
            print("f12")
        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        print("f13")
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        print("f14")
        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)
        print("f15")
        # apply temperature
        logits /= self.T
        print("f16")
        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        print("f17")
        # dequeue and enqueue
        self._dequeue_and_enqueue(k)
        print("f18")
        return logits, labels


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output