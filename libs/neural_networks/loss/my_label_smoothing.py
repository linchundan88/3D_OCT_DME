import torch
import torch.nn as nn
from torch.nn import functional as F

'''https://www.aiuai.cn/aifarm1333.html  
solution no.1, add class weights support'''
class LabelSmoothLoss(nn.Module):
    def __init__(self, class_weight=None, smoothing=0.0):
        super(LabelSmoothLoss, self).__init__()
        self.class_weight = class_weight
        self.class_weight_avg = torch.mean(class_weight)
        self.smoothing = smoothing

    def forward(self, input, target):
        log_prob = F.log_softmax(input, dim=-1)
        weight = input.new_ones(input.size()) * \
                 self.smoothing / (input.size(-1) - 1.)
        weight.scatter_(-1, target.unsqueeze(-1), (1. - self.smoothing))

        if self.class_weight is None:
            loss = (-weight * log_prob).sum(dim=-1).mean()
        else:
            # (weight * log_prob).shape:  (batch_size, num_classes)
            # (weight * log_prob * self.class_weight).shape:  (batch_size, num_classes)
            loss = (-weight * log_prob * self.class_weight).sum(dim=-1).mean()
            loss = loss / self.class_weight_avg

        return loss


'''
new_labels = (1.0 - label_smoothing) * one_hot_labels + label_smoothing / num_classes

'''

'''
https://bbs.cvmart.net/topics/4241

    images, labels = images.cuda(), labels.cuda()
    N = labels.size(0)
    # C is the number of classes.
    smoothed_labels = torch.full(size=(N, C), fill_value=0.1 / (C - 1)).cuda()
    smoothed_labels.scatter_(dim=1, index=torch.unsqueeze(labels, dim=1), value=0.9)

    score = model(images)
    log_prob = torch.nn.functional.log_softmax(score, dim=1)
    loss = -torch.sum(log_prob * smoothed_labels) / N
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
'''


'''
# https://www.aiuai.cn/aifarm1333.html  solution no.4
class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

'''