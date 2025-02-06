import torch
import torch.nn as nn
import torch.nn.functional as F

class Loss_Function(nn.Module):
    def __init__(self):
        super(Loss_Function, self).__init__()

    def Spatial_Loss(self, weight, logit_lm, ground_truth):  # STAL loss function
        _, num_classes = logit_lm.size()
        p_i = torch.softmax(logit_lm, dim=1) # batchsize*5,vocabsize
        spatial_matrix = torch.index_select(weight, 0, ground_truth)  # select value by index
        y = F.one_hot(ground_truth, num_classes=num_classes)
        y_h = torch.max(y*p_i, dim=1, keepdim=False).values.unsqueeze(1)

        a = (1-y_h) * (1-spatial_matrix) *  torch.log((p_i) + 0.0000001) / (num_classes)

        b = y * torch.log(p_i + 0.0000001) 
        loss = a + b

        loss = torch.sum(loss, dim=1)
        loss = -torch.mean(loss, dim=0)
        return loss