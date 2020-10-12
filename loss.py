import torch
import torch.nn as nn

class STDLoss(nn.Module):
    def __init__(self, cluster_num):
        super(STDLoss, self).__init__()
        self.cluster_num = cluster_num

    def forward(self, target):
        C, S, W = int(target.shape[0]), int(target.shape[1]), int(target.shape[2])

        zeros = torch.zeros((C, S, W, self.cluster_num), dtype=torch.float, device=target.device) # [class, support, window, cluster]
        target_labels = (target - 1).reshape(C, S, W, 1) # [class, support, window, 1]
        one_hot = zeros.scatter_(3, target_labels, 1) # [class, support, window, cluster]
        one_hot.requires_grad = True
        one_hot_ = torch.mean(one_hot, dim=1).reshape(C, -1) # [class, window*cluster]
        inter_class_loss = 1 / torch.sum(torch.std(one_hot_, dim=0))


        if S > 1:
            class_one_hot = one_hot.reshape(C, S, -1) # [class, support, window*cluster]
            class_one_hot = torch.std(class_one_hot, dim=1) # [class, window*cluster]
            class_one_hot = torch.sum(class_one_hot, dim=1) # [class]
            in_class_loss = torch.mean(class_one_hot)
            
            return inter_class_loss + in_class_loss
        else:
            return inter_class_loss

class DistanceLoss(nn.Module):
    def __init__(self):
        super(DistanceLoss, self).__init__()
        self.cos = nn.CosineSimilarity(dim=1)

    def forward(self, cluster_centers):
        C, F = int(cluster_centers.shape[0]), int(cluster_centers.shape[1]) # [cluster, feature]

        cluster_centers_1 = cluster_centers.repeat(C, 1) # [cluster*cluster, feature]
        cluster_centers_2 = cluster_centers.unsqueeze(1).repeat(1, C, 1).reshape(C*C, F) # [cluster*cluster, feature]

        distances = self.cos(cluster_centers_1, cluster_centers_2) # [cluster*cluster]
        distances[distances==1] = -1

        loss = torch.sum(distances + 1)/(C*(C-1))

        return loss