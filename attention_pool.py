import torch
import torch.nn as nn

def weights_init(m):
    try:
        kaiming_normal_(m.weight)
    except Exception:
        return

class AttentionPooling(nn.Module):
    def __init__(self, class_num, sample_num, query_num, window_num, clip_num, feature_dim):
        super(AttentionPooling, self).__init__()
        self.class_num = class_num
        self.sample_num = sample_num
        self.window_num = window_num
        self.clip_num = clip_num
        self.feature_dim = feature_dim

        self.query_dim = class_num*query_num*window_num
        self.sample_dim = class_num*sample_num*window_num
        
        self.layer1 = nn.Sequential(
                        nn.Linear(self.sample_dim, self.sample_dim*2),
                        nn.BatchNorm1d(self.sample_dim*2, affine=False),
                        nn.ReLU(),
                        nn.Linear(self.sample_dim*2, self.sample_dim),
                        nn.BatchNorm1d(self.sample_dim, affine=False),
                        nn.ReLU())
        
        self.layer2 = nn.Sequential(
                        nn.Linear(self.clip_num*self.feature_dim, self.clip_num*self.feature_dim*2),
                        nn.BatchNorm1d(self.clip_num*self.feature_dim*2, affine=False),
                        nn.ReLU(),
                        nn.Linear(self.clip_num*self.feature_dim*2, self.clip_num*self.feature_dim),
                        nn.BatchNorm1d(self.clip_num*self.feature_dim, affine=False),
                        nn.ReLU())
        
        self.apply(weights_init)

    def forward(self, samples, batches):
        self.query_dim = int(batches.shape[0])

        samples_trans = torch.transpose(samples, 0, 1)    # [clip, class*sample*window, feature]
        samples_trans = torch.transpose(samples_trans, 1, 2)    # [clip, feature, class*sample*window]

        weight = torch.tensordot(batches, samples_trans, dims=2) # [query*(class+1)*window, class*sample*window]
        weight = self.layer1(weight)
        weight = weight.reshape(self.query_dim*self.class_num, self.sample_num*self.window_num).unsqueeze(1) # [query*(class+1)*window*class, 1, sample*window]

        samples = samples.unsqueeze(0).repeat(self.query_dim,1,1,1).reshape(self.query_dim*self.class_num, self.sample_num*self.window_num, -1) # [query*(class+1)*window*class, sample*window, clip*feature]
        samples = torch.bmm(weight, samples).squeeze(1)                        # [query*(class+1)*window*class, clip*feature]

        samples = self.layer2(samples)                                         # [query*(class+1)*window*class, clip*feature]
        samples = samples.reshape(self.query_dim, self.class_num, self.clip_num, -1)  # [query*(class+1)*window, class, clip, feature]

        return samples