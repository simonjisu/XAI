import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionHead(nn.Module):
    """AttentionHead"""
    def __init__(self, in_c, n_head):
        super(AttentionHead, self).__init__()
        """
        3.2 Attention Head
        
        args:
        - in_c: C
        - n_head: K
        """
        self.n_head = n_head
        self.conv = nn.Conv2d(in_c, n_head, kernel_size=3, padding=1, bias=False)
        self.diag = (1 - torch.eye(n_head, n_head))
        
    def forward(self, x):
        """
        args:
        - x: feature activations, (B, C, H, W)
        
        returns:
        - Tensor, K-attention masks(softmax by channel-wise)  (B, K, H, W)
        """
        B = x.size(0)
        conv_heads = self.conv(x)  # (B, C, H, W) > (B, K, H, W)
        masks = torch.softmax(conv_heads, dim=1)  # K-attention masks
        self.masks_score = masks.view(B, self.n_head, -1)
        return masks  # (B, K, H, W)
    
    def reg_loss(self):
        """
        calculate reg_loss
        """
        # (B, K, H*W) x (B, H*W, K) > (B, K, K)
        reg_loss = self.diag.to(self.masks_score.device) * torch.bmm(self.masks_score, self.masks_score.transpose(1, 2))
        return reg_loss.sum()


class AttentionOut(nn.Module):
    """AttentionOut"""
    def __init__(self, in_c, n_head, n_label=1, gate=False):
        super(AttentionOut, self).__init__()
        """
        3.3 Output head / 3.4 Layered attention gates
        
        args:
        - in_c: C
        - n_head: K
        - n_label: L
        - gate: if gate is `True`, returns attention gates
        """
        self.n_head = n_head
        self.n_label = n_label
        self.gate = gate
        if gate:
            assert self.n_label == 1, "Gate must set `n_label = 1`"
        self.conv = nn.Conv2d(in_c, n_head*n_label, kernel_size=3, padding=1, bias=False)

    def forward(self, x, masks):
        """
        args:
        - x: feature activations, (B, C, H, W)
        - masks: masks from `AttentionHead`, (B, K, H, W)
        
        returns:
        - scores: when `self.gate=False`, (B, K, L)
        - gates: when `self.gate=True`, (B, K, 1)
        """
        B = x.size(0)
        conv_outputs = self.conv(x)  # (B, C, H, W) > (B, K*L, H, W)
        outputs = conv_outputs.view(B, self.n_head, self.n_label, -1)  # (B, K, L, H*W)
        # (B, K, L, H*W) * (B, K, 1, H*W) > (B, K, L)
        scores = (outputs * masks.view(B, self.n_head, -1).unsqueeze(2)).sum(-1)
        if not self.gate:
            return scores
        else:
            # L = 1, returns Tensor (B, K, 1)
            gates = torch.softmax(torch.tanh(scores), dim=1)
            return gates


class AttentionModule(nn.Module):
    """AttentionModule"""
    def __init__(self, in_c, n_head, n_label, reg_weight=0.0):
        """
        calculate outputs of attention module
        
        args:
        - in_c: the number of input channels
        - n_head: the attention width, the number of layers using the attention mechanism
        - n_label: the number of class labels
        """
        super(AttentionModule, self).__init__()
        self.reg_weight = reg_weight
        self.attn_heads = AttentionHead(in_c, n_head)
        self.output_heads = AttentionOut(in_c, n_head, n_label=n_label, gate=False)
        self.attn_gates = AttentionOut(in_c, n_head, gate=True)
        
    def forward(self, x):
        """
        args:
        - x: feature activations, (B, C, H, W)
        
        returns:
        - outputs_vectors: predict vectors which applied the most meaningful attention head, (B, L)
        
        """
        masks = self.attn_heads(x)  # (B, K, H, W)  softmax: dim=1
        outputs = self.output_heads(x, masks)  # (B, K, L)
        gates = self.attn_gates(x, masks)  # (B, K, 1)  softmax: dim=1
        outputs_vectors = (outputs * gates).sum(1)  # (B, L)
        return outputs_vectors
        
    def reg_loss(self):
        return self.attn_heads.reg_loss() * self.reg_weight


class GlobalAttentionGate(nn.Module):
    """GlobalAttentionGate"""
    def __init__(self, in_c, n_hypothesis, gate_fn="softmax"):
        """
        args:
        - in_c: the number of input features
        - n_hypothesis: the number of total hypothesis, including original output vector and attention outputs
        """
        super(GlobalAttentionGate, self).__init__()
        self.n_hypothesis = n_hypothesis
        self.gate_fn = gate_fn
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.gate_layer = nn.Linear(in_c, n_hypothesis, bias=False)
    
    def cal_global_gates(self, x):
        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # (B, C) > (B, N)
        c = torch.tanh(self.gate_layer(x))
        if self.gate_fn == "softmax":
            global_gates = torch.softmax(c, dim=1)
        elif self.gate_fn == "sigmoid":
            global_gates = torch.sigmoid(c)
        return global_gates.unsqueeze(-1)  # (B, N, 1)
    
    def forward(self, x, hypothesis):
        """
        args:
        - x: last feature activations, (B, C, H, W)
        - hypothesis: all hypothesis, list type contains N+1 of (B, 1, L) size Tensor
        
        returns:
        - global_gates: (B, N)
        """
        global_gates = self.cal_global_gates(x)  # (B, N, 1)
        outputs = torch.cat(hypothesis, dim=1)  #(B, N, L)
        outputs = torch.log_softmax(outputs, dim=2)  # calculate log probs
        outputs_net = (outputs * global_gates).sum(1)  # (B, L)
        return outputs_net

    @staticmethod
    def loss_function(outputs_net, targets, reg_loss, reduction="mean"):
        loss = F.nll_loss(outputs_net, targets, reduction=reduction) + reg_loss
        return loss