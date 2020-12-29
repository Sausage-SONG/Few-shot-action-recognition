# Public Packages
import torch
import torch.nn as nn
import math

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())

# Relation Network Module
class RelationNetwork(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RelationNetwork, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv1d(input_size*2,input_size,kernel_size=1),
                        nn.BatchNorm1d(input_size, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool1d(2))
        self.layer2 = nn.Sequential(
                        nn.Conv1d(input_size,input_size,kernel_size=3),
                        nn.BatchNorm1d(input_size, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool1d(2))
        self.fc1 = nn.Linear(75, hidden_size)
        self.fc2 = nn.Linear(hidden_size,1)

        # # Initialize itself
        self.apply(weights_init)

    def forward(self,x):    
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0),-1)
        out = nn.functional.relu(self.fc1(out))
        out = torch.sigmoid(self.fc2(out))
        return out

class RelationNetworkZero(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RelationNetworkZero, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv1d(input_size, int(input_size/2), kernel_size=1),
                        nn.BatchNorm1d(int(input_size/2), momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool1d(2))
        self.layer2 = nn.Sequential(
                        nn.Conv1d(int(input_size/2), int(input_size/4), kernel_size=3),
                        nn.BatchNorm1d(int(input_size/4), momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool1d(2))
        # self.fc1 = nn.Linear(105, hidden_size) # 5way
        self.fc1 = nn.Linear(75, hidden_size) # 3way
        self.fc2 = nn.Linear(hidden_size,1)

        # # Initialize itself
        self.apply(weights_init)

    def forward(self,x):    
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0),-1)
        out = nn.functional.relu(self.fc1(out))
        out = torch.sigmoid(self.fc2(out))
        return out

class MultiRelationNetwork(nn.Module):
    
    def __init__(self, input_size, hidden_size, main_fc_dim, feature_dim):
        super(MultiRelationNetwork, self).__init__()

        # # FC
        # self.main_fc = nn.Sequential(
        #                 nn.Linear(main_fc_dim*2, main_fc_dim),
        #                 nn.ReLU(),
        #                 nn.Linear(main_fc_dim, 1))

        # # Window-wise Inner Product
        # self.ip_layer = 

        # Clip-wise
        self.cw_layer = nn.Sequential(
                        nn.Conv1d(2, 5, kernel_size=3, padding=1),
                        nn.BatchNorm1d(5, momentum=1, affine=False),
                        nn.ReLU(),
                        nn.Conv1d(5, 1, kernel_size=3, padding=1),
                        nn.BatchNorm1d(1, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.Linear(feature_dim, int(feature_dim/2)),
                        nn.ReLU(),
                        nn.Linear(int(feature_dim/2), 1),
                        nn.ReLU())

        # Original
        self.ori_layer = nn.Sequential(
                        nn.Conv1d(input_size*2,input_size,kernel_size=1),
                        nn.BatchNorm1d(input_size, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool1d(2),
                        nn.Conv1d(input_size,input_size,kernel_size=3),
                        nn.BatchNorm1d(input_size, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool1d(2))
        self.ori_fc1 = nn.Linear(75, hidden_size)
        self.ori_fc2 = nn.Linear(hidden_size,1)

        # Initialize itself
        self.weights_init()
    
    def weights_init(self):
        # nn.init.kaiming_normal_(self.main_fc[0].weight)
        # nn.init.kaiming_normal_(self.main_fc[2].weight)

        nn.init.kaiming_normal_(self.cw_layer[0].weight)
        nn.init.kaiming_normal_(self.cw_layer[3].weight)
        nn.init.kaiming_normal_(self.cw_layer[6].weight)
        nn.init.kaiming_normal_(self.cw_layer[8].weight)

        nn.init.kaiming_normal_(self.ori_layer[0].weight)
        nn.init.kaiming_normal_(self.ori_layer[4].weight)
        nn.init.kaiming_normal_(self.ori_fc1.weight)
        nn.init.kaiming_normal_(self.ori_fc2.weight)

    def forward(self, support, query):
        '''
        param:support   shape [class*query, window, clip, class, feature]
        param:query     shape [class*query, window, clip, feature]
        return:         shape [class*query, class]
        '''
        Q_num, W, CL, C, FE = support.shape # class*query, window, clip, class, feature

        # # FC
        # support_fc = support.permute(0,3,1,2,4).reshape(Q_num*C, W, CL*FE)         # [class*query*class, window, clip*feature]
        # query_fc = query.unsqueeze(1).repeat(1,C,1,1,1).reshape(Q_num*C, W, CL*FE) # [class*query*class, window, clip*feature]
        # fc_cat = torch.cat((support_fc, query_fc), 2)                              # [class*query*class, window, clip*feature*2]
        # fc_scores = F.sigmoid(self.main_fc(fc_cat).reshape(Q_num, C, W))           # [class*query, class, window]
        # fc_scores = fc_scores.permute(0,2,1)                                       # [class*query, window, class]
        # fc_scores = F.softmax(fc_scores, dim=2)                                    # [class*query, window, class]

        # Window-wise Inner Product
        support_ip = support.permute(3,0,1,2,4).reshape(C, Q_num*W, -1) # [class, class*query*window, clip*feature]
        query_ip = query.reshape(1, Q_num*W, -1).repeat(C,1,1)          # [class, class*query*window, clip*feature]
        ip_scores = torch.einsum('ijk,ijk->ij', support_ip, query_ip)   # [class, class*query*window]
        print(ip_scores)
        return None
        ip_scores = ip_scores.reshape(C, Q_num, W).permute(1,2,0)       # [class*query, window, class]
        ip_scores = F.softmax(ip_scores, dim=2)                         # [class*query, window, class]
        
        # Clip-wise
        support_wc = support.permute(0,3,1,2,4).reshape(Q_num*C, W, CL, 1, FE)         # [class*query*class, window, clip, 1, feature]
        query_cw = query.unsqueeze(1).repeat(1,C,1,1,1).reshape(Q_num*C, W, CL, 1, FE) # [class*query*class, window, clip, 1, feature]
        cw_cat = torch.cat((support_wc, query_cw), 3).reshape(-1, 2, FE)               # [class*query*class*window*clip, 2, feature]
        cw_scores = self.cw_layer(cw_cat).reshape(Q_num, C, W, CL)                     # [class*query, class, window, clip]
        cw_scores = cw_scores.permute(0,2,1,3)                                         # [class*query, window, class, clip]
        cw_scores = F.softmax(torch.sum(cw_scores, dim=3), dim=2)                      # [class*query, window, class]

        # Original Relation
        support_rn = support.permute(0,1,3,2,4).reshape(Q_num*W*C, CL, FE)               # [class*query*window*class, clip, feature]
        query_rn = query.reshape(Q_num*W, 1, CL, FE).repeat(1,C,1,1).reshape(-1, CL, FE) # [class*query*window*class, clip, feature]
        ori_cat = torch.cat((support_rn, query_rn), 1)                                   # [class*query*window*class, clip*2, feature]
        ori_scores = self.ori_layer(ori_cat)
        ori_scores = ori_scores.reshape(ori_scores.size(0), -1)
        ori_scores = F.relu(self.ori_fc1(ori_scores))
        ori_scores = F.relu(self.ori_fc2(ori_scores)).reshape(Q_num, W, C)             # [class*query, window, class]
        ori_scores = F.softmax(ori_scores, dim=2)                                      # [class*query, window, class]

        final_scores = cw_scores + ori_scores + ip_scores  # [class*query, window, class]
        final_scores = F.softmax(final_scores, dim=2)      # [class*query, window, class]

        return final_scores