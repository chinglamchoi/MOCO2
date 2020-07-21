# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn

#def f():
#    global gpu_idx
#    gpu_idx += 1
#    gpu_idx = 0 if gpu_idx == 8 else gpu_idx

#gpu_idx = 0
class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, dim=8, K=65536, m=0.999, T=0.07, mlp=False):
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
        self.register_buffer("queue", torch.randn(2, dim, K))
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
    def _dequeue_and_enqueue(self, key1, key2):
        # gather keys before updating queue
        #keys = keys#concat_all_gather(keys)

        batch_size = key1.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[0, :, ptr:ptr + batch_size] = key1.transpose(0,1)
        self.queue[1, :, ptr:ptr + batch_size] = key2.transpose(0,1)
        ptr = (ptr + batch_size) % self.K  # move pointer
        self.queue_ptr[0] = ptr

    def forward(self, ima_q, ima_k, imb_q, imb_k):
        """
        Input:
            im_q: a batch of query images (A, B)
            im_k: a batch of key images (A, B)
        Output:
            logits, targets
        """

        # compute query features
        q1, q2 = self.encoder_q(ima_q, imb_q)  # queries: NxC
        #qq = torch.stack((q[0], q[1]))
        q1 = nn.functional.normalize(q1, dim=1)
        q2 = nn.functional.normalize(q2, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder
            # shuffle for making use of BN
            k1, k2 = self.encoder_k(ima_k, imb_k)  # keys: NxC
            k1 = nn.functional.normalize(k1, dim=1)
            k2 = nn.functional.normalize(k2, dim=1)
        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos1 = torch.einsum('nc,nc->n', [q1, k1]).unsqueeze(-1)
        l_pos2 = torch.einsum('nc,nc->n', [q2, k2]).unsqueeze(-1)
        # negative logits: NxK
        l_neg1 = torch.einsum('nc,ck->nk', [q1, self.queue[0].clone().detach()])
        l_neg2 = torch.einsum('nc,ck->nk', [q2, self.queue[1].clone().detach()])
        # logits: Nx(1+K)
        logits1 = torch.cat([l_pos1, l_neg1], dim=1)
        logits2 = torch.cat([l_pos2, l_neg2], dim=1)
        # apply temperature
        logits1 /= self.T
        logits2 /= self.T
        # labels: positive key indicators
        labels1 = torch.zeros(logits1.shape[0], dtype=torch.long).cuda()
        labels2 = torch.zeros(logits2.shape[0], dtype=torch.long).cuda()
        # dequeue and enqueue
        self._dequeue_and_enqueue(k1, k2)
        return logits1, logits2, labels1, labels2

"""
# utils
@torch.no_grad()
def concat_all_gather(tensor):
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    tensors_gather = [torch.ones_like(tensor)
                    for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    return output
"""
