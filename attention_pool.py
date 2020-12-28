import torch
import torch.nn as nn

class AttentionPooling(nn.Module):
    def __init__(self, class_num, sample_num, query_num, window_num, clip_num, feature_dim):
        super(AttentionPooling, self).__init__()
        self.class_num = class_num
        self.sample_num = sample_num
        self.window_num = window_num
        self.clip_num = clip_num
        self.feature_dim = feature_dim
        # self.k = k

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
        
        self.weights_init()
    
    def weights_init(self):
        nn.init.kaiming_normal_(self.layer1[0].weight)
        nn.init.kaiming_normal_(self.layer1[3].weight)
        nn.init.kaiming_normal_(self.layer2[0].weight)
        nn.init.kaiming_normal_(self.layer2[3].weight)

    def forward(self, samples, batches):
        self.query_dim = int(batches.shape[0])

        samples_trans = torch.transpose(samples, 0, 1)    # [clip, class*sample*window, feature]
        samples_trans = torch.transpose(samples_trans, 1, 2)    # [clip, feature, class*sample*window]

        weight = torch.tensordot(batches, samples_trans, dims=2) # [query*class*window, class*sample*window]
        weight = self.layer1(weight)
        weight = weight.reshape(self.query_dim, self.class_num*self.sample_num*self.window_num) # [query*class*window, class*sample*window]
        
        # if self.k > 0:
        #     _, topk_idx = torch.topk(weight, self.k, largest=True, sorted=True, out=None) # [query*class*window, class, k]
            
        #     weight_01 = weight.new_full((self.query_dim, self.class_num, self.sample_num*self.window_num), 0, requires_grad=True) # [query*class*window, class, sample*window]
        #     weight = weight_01.scatter_(2, topk_idx, 1)

        weight = weight.reshape(self.query_dim*self.class_num, self.sample_num*self.window_num).unsqueeze(1) # [query*class*window*class, 1, sample*window]

        samples = samples.unsqueeze(0).repeat(self.query_dim,1,1,1).reshape(self.query_dim*self.class_num, self.sample_num*self.window_num, -1) # [query*class*window*class, sample*window, clip*feature]
        samples = torch.bmm(weight, samples).squeeze(1)                        # [query*class*window*class, clip*feature]

        samples = self.layer2(samples)                                         # [query*class*window*class, clip*feature]
        samples = samples.reshape(self.query_dim, self.class_num, self.clip_num, -1)  # [query*class*window, class, clip, feature]

        return samples

class AttentionPoolingConv(nn.Module):
    def __init__(self, class_num, sample_num, query_num, window_num, clip_num, feature_dim):
        super(AttentionPoolingConv, self).__init__()
        self.class_num = class_num
        self.sample_num = sample_num
        self.window_num = window_num
        self.clip_num = clip_num
        self.feature_dim = feature_dim
        # self.k = k

        self.query_dim = class_num*query_num*window_num
        self.sample_dim = class_num*sample_num*window_num
        
        self.layer = nn.Sequential(
                        nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
                        nn.BatchNorm1d(16, affine=False),
                        nn.ReLU(),
                        nn.Conv1d(in_channels=16, out_channels=1, kernel_size=3, padding=1),
                        nn.BatchNorm1d(1, affine=False),
                        nn.ReLU())
        
        self.weights_init()
    
    def weights_init(self):
        nn.init.kaiming_normal_(self.layer[0].weight)
        nn.init.kaiming_normal_(self.layer[3].weight)

    def forward(self, samples, batches):
        self.query_dim = int(batches.shape[0])

        samples_trans = torch.transpose(samples, 0, 1)    # [clip, class*sample*window, feature]
        samples_trans = torch.transpose(samples_trans, 1, 2)    # [clip, feature, class*sample*window]

        weight = torch.tensordot(batches, samples_trans, dims=2) # [query*class*window, class*sample*window]
        weight = weight.unsqueeze(1) # [query*class*window, 1, class*sample*window]
        weight = self.layer(weight)

        weight = weight.reshape(self.query_dim*self.class_num, self.sample_num*self.window_num).unsqueeze(1) # [query*class*window*class, 1, sample*window]

        samples = samples.unsqueeze(0).repeat(self.query_dim,1,1,1).reshape(self.query_dim*self.class_num, self.sample_num*self.window_num, -1) # [query*class*window*class, sample*window, clip*feature]
        samples = torch.bmm(weight, samples).squeeze(1)                        # [query*class*window*class, clip*feature]

        samples = samples.reshape(self.query_dim, self.class_num, self.clip_num, -1)  # [query*class*window, class, clip, feature]

        return samples

