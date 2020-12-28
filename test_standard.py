# Public Packages
import torch                                     #
import torch.nn as nn                            #  Pytorch
from torch.autograd import Variable              #

import numpy as np                               #  Numpy
import os                                        #  OS
import sys
import random

# Local Packages
from relationNet import RelationNetwork as RN       #  Relation Network
from relationNet import RelationNetworkZero as RN0  #
from encoder import Simple3DEncoder as C3D          #  Conv3D
from tcn import TemporalConvNet as TCN              #  TCN
from attention_pool import AttentionPooling as AP   #  Attention Pooling
import dataset                                      #  Task Generator
from utils import *                                 #  Helper Functions

# Constant (Settings)
TCN_OUT_CHANNEL = 64                    # Num of channels of output of TCN
RELATION_DIM = 32                       # Dim of one layer of relation net
CLASS_NUM = 3                           # <X>-way  | Num of classes
SAMPLE_NUM = 5                          # <Y>-shot | Num of supports per class
QUERY_NUM = 3                           # Num of instances for validation per class
TEST_EPISODE = 500                      # Num of validation episode
FRAME_NUM = 10                          # Num of frames per clip
CLIP_NUM = 5                            # Num of clips per window
WINDOW_NUM = 3                          # Num of processing window per video
INST_NUM = 10                           # Num of videos selected in each class

DATA_FOLDERS = ["/data/ssongad/mit2/frame/train",        
                "/data/ssongad/mit2/frame/val"]

encoder_saved_model = "/data/ssongad/codes/ctc_ap_v2/models/CTC+MSE_Full_MIT_woRN_3W5S/0.46/encoder.pkl"     # Path of saved encoder model
tcn_saved_model = "/data/ssongad/codes/ctc_ap_v2/models/CTC+MSE_Full_MIT_woRN_3W5S/0.46/tcn.pkl"             # Path of saved tcn model
ap_saved_model = "/data/ssongad/codes/ctc_ap_v2/models/CTC+MSE_Full_MIT_woRN_3W5S/0.46/ap.pkl"  
rn_saved_model = "/data/ssongad/codes/ctc_ap_v2/models/CTC+MSE_Full_MIT_woRN_3W5S/0.46/rn.pkl"               # Path of saved relation net model
# rn0_saved_model = "/data/ssongad/codes/ctc_ap_v2/models/CTC_Full_HAA_Attention_v2_5S/0.7933333333333334/rn0.pkl"             # Path of saved rn0 model


# Device to be used
os.environ['CUDA_VISIBLE_DEVICES']="8"                          # GPU to be used
device = torch.device('cuda')

# TEST_SPLIT = "/data/ssongad/haa/test.txt"
TEST_SPLIT = "/data/ssongad/mit2/test.txt"
TEST_SPLIT = read_split(TEST_SPLIT)

def main():

    # Define models
    encoder = C3D(in_channels=3)
    encoder = nn.DataParallel(encoder)
    rn = RN(CLIP_NUM, RELATION_DIM)
    # rn0 = RN0(CLIP_NUM*(CLASS_NUM+1), RELATION_DIM)
    tcn = TCN(245760, [128,128,64,TCN_OUT_CHANNEL])
    ap = AP(CLASS_NUM, SAMPLE_NUM, QUERY_NUM, WINDOW_NUM, CLIP_NUM, TCN_OUT_CHANNEL)

    # Move models to GPU
    encoder.to(device)
    rn.to(device)
    # rn0.to(device)
    tcn.to(device)
    ap.to(device)

    # Load Saved Model
    if os.path.exists(encoder_saved_model):
        encoder.load_state_dict(torch.load(encoder_saved_model))
    if os.path.exists(rn_saved_model):
        rn.load_state_dict(torch.load(rn_saved_model))
    if os.path.exists(tcn_saved_model):
        tcn.load_state_dict(torch.load(tcn_saved_model))
    if os.path.exists(ap_saved_model):
        ap.load_state_dict(torch.load(ap_saved_model))

    # Testing
    with torch.no_grad():
        accuracies = []

        test_episode = 0
        while test_episode < TEST_EPISODE:
            print("Test_Epi[{}]".format(test_episode), end="\t")
            
            # Data Loading
            try:
                the_dataset = dataset.StandardDataset(DATA_FOLDERS, "test", (None, None, TEST_SPLIT), CLASS_NUM, INST_NUM, FRAME_NUM, CLIP_NUM, WINDOW_NUM)
                dataloader = dataset.get_data_loader(the_dataset, num_per_class=SAMPLE_NUM+QUERY_NUM, num_workers=0)
                data, data_labels = dataloader.__iter__().next()     # [class*(support+query), window*clip, RGB, frame, H, W]
            except Exception:
                print("Skipped")
                continue
            
            data = data.view(-1, 3, FRAME_NUM, 128, 128)
            total_num_covered = CLASS_NUM * QUERY_NUM

            # Generate support & query split
            query_index = []
            support_index = []
            for i in range(CLASS_NUM):
                start = (SAMPLE_NUM+QUERY_NUM) * i
                end = (SAMPLE_NUM+QUERY_NUM) * (i+1)
                index = list(range(start, end))
                q = random.sample(index, QUERY_NUM)
                s = list(set(index)-set(q))
                query_index.extend(q)
                support_index.extend(s)
            random.shuffle(query_index)
            query_index = torch.tensor(query_index)
            support_index = torch.tensor(support_index)
            
            # Encoding
            embed = encoder(Variable(data).to(device))
            embed = embed.view(CLASS_NUM*(SAMPLE_NUM+QUERY_NUM), WINDOW_NUM*CLIP_NUM, -1)  # [class*(support+query), window*clip, feature]

            # TCN Processing
            embed = torch.transpose(embed, 1, 2)           # [class*(support+query), feature(channel), window*clip(length)]
            embed = tcn(embed)
            embed = torch.transpose(embed, 1, 2)           # [class*(support+query), window*clip, feature]

            # Split data into support & query
            samples = embed[support_index] # [class*support, window*clip, feature]
            batches = embed[query_index]   # [class*query, window*clip, feature]
            batches_labels = data_labels[query_index]

            # Attention Pooling
            samples = samples.reshape(CLASS_NUM*SAMPLE_NUM*WINDOW_NUM, CLIP_NUM, -1)  # [class*sample*window, clip, feature]
            batches = batches.reshape(CLASS_NUM*QUERY_NUM*WINDOW_NUM, CLIP_NUM, -1)  # [query*class*window, clip, feature]
            samples = ap(samples, batches)                    # [query*class*window, class, clip, feature]
            # samples = samples.reshape(CLASS_NUM, SAMPLE_NUM, WINDOW_NUM, CLIP_NUM, -1) # [class, support, window, clip, feature]
            # samples = torch.mean(samples, dim=1).permute(1,0,2,3).unsqueeze(0) # [1, window, class, clip, feature]
            # samples = samples.repeat(CLASS_NUM*QUERY_NUM,1,1,1,1).reshape(-1, CLASS_NUM, CLIP_NUM, TCN_OUT_CHANNEL) # [query*class*window, class, clip, feature]

            # Compute Relation
            # batches_rn = batches.unsqueeze(0).repeat(CLASS_NUM,1,1,1)  # [class, query*class*window, clip, feature]
            # batches_rn = torch.transpose(batches_rn,0,1)               # [query*class*window, class, clip, feature]
            # relations = torch.cat((samples,batches_rn),2).reshape(-1,CLIP_NUM*2,TCN_OUT_CHANNEL)    # [query*class*window, class, clip*2(channel), feature]
            # relations = rn(relations).reshape(QUERY_NUM*CLASS_NUM, WINDOW_NUM, CLASS_NUM)    # [query*class, window, class]
            
            cos = nn.CosineSimilarity(dim=2)
            samples = samples.reshape(CLASS_NUM*QUERY_NUM*WINDOW_NUM, CLASS_NUM, -1)
            batches = batches.reshape(CLASS_NUM*QUERY_NUM*WINDOW_NUM,1,-1).repeat(1,CLASS_NUM,1)
            relations = cos(samples,batches).reshape(QUERY_NUM*CLASS_NUM, WINDOW_NUM, CLASS_NUM)

            # Compute Zero Probability
            # samples_rn0 = samples.reshape(QUERY_NUM*CLASS_NUM*WINDOW_NUM, CLASS_NUM*CLIP_NUM, TCN_OUT_CHANNEL)  # [query*class*window, class*clip, feature]
            # relations_rn0 = torch.cat((batches, samples_rn0), 1)   # [query*class*window, (class+1)*clip, feature]
            # blank_prob = rn0(relations_rn0).reshape(QUERY_NUM*CLASS_NUM, WINDOW_NUM, 1)  # [query*class, window, 1]
            blank_prob = torch.full(size=(QUERY_NUM*CLASS_NUM, WINDOW_NUM, 1), fill_value=1, dtype=torch.float).to(device)

            # Generate final probabilities
            relations_ctc = torch.cat((blank_prob, relations), 2)
            final_outcome = nn.functional.softmax(relations_ctc, 2)  # [query*class, window(length), class+1]

            # Predict
            # relations_mse = nn.functional.softmax(torch.sum(relations, 1), dim=1) # [query*class, class]
            # _, predict_labels = torch.max(relations_mse.data, 1)
            # batches_labels = batches_labels - 1
            # predict_labels = ctc_predict(final_outcome.cpu().numpy())
            predict_labels = ctc_predict_single(final_outcome)
            batches_labels = batches_labels.numpy()

            # Counting Correct Ones
            rewards = [compute_score(prediction, truth) for prediction, truth in zip(predict_labels, batches_labels)]
            total_rewards = np.sum(rewards)

            # Record accuracy
            accuracy = total_rewards/total_num_covered
            accuracies.append(accuracy)
            print("Accuracy = {}".format(accuracy))
            print(batches_labels)
            print(predict_labels)
            test_accuracy, _ = mean_confidence_interval(accuracies)
            print(test_accuracy)
            print()

            test_episode += 1
    
        # Overall accuracy
        test_accuracy, _ = mean_confidence_interval(accuracies)
        print("Final_Accu = {}".format(test_accuracy))

# Program Starts
if __name__ == '__main__':
    main()
