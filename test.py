# Public Packages
import torch                                     #
import torch.nn as nn                            #  Pytorch
from torch.autograd import Variable              #

import numpy as np                               #  Numpy
import os                                        #  OS
import sys

# Local Packages
from relationNet import RelationNetwork as RN       #  Relation Network
from relationNet import RelationNetworkZero as RN0  #
# from i3d import InceptionI3d as I3D               #  I3D
from i3d import Simple3DEncoder as C3D              #  Conv3D
from tcn import TemporalConvNet as TCN              #  TCN
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

DATA_FOLDERS = ["/data/ssongad/haa/new_normalized_frame/",        #
                "/data/ssongad/haa/normalized_frame_scale2",      # Data path => [original, 2x, 3x]
                "/data/ssongad/haa/normalized_frame_scale3"]      #

encoder_saved_model = "/data/ssongad/codes/ctc2/models/TCN_Simple3DEnconder_CTC_VSL/encoder_0.8377777777777777.pkl"     # Path of saved encoder model
rn_saved_model = "/data/ssongad/codes/ctc2/models/TCN_Simple3DEnconder_CTC_VSL/rn_0.8377777777777777.pkl"               # Path of saved relation net model
tcn_saved_model = "/data/ssongad/codes/ctc2/models/TCN_Simple3DEnconder_CTC_VSL/tcn_0.8377777777777777.pkl"             # Path of saved tcn model
rn0_saved_model = "/data/ssongad/codes/ctc2/models/TCN_Simple3DEnconder_CTC_VSL/rn0_0.8377777777777777.pkl"             # Path of saved rn0 model

# Device to be used
os.environ['CUDA_VISIBLE_DEVICES']="1"                          # GPU to be used
device = torch.device('cuda')

def main():

    # Define models
    encoder = C3D(in_channels=3)
    encoder = nn.DataParallel(encoder)
    rn = RN(CLIP_NUM, RELATION_DIM)
    rn0 = RN0(CLIP_NUM*(CLASS_NUM+1), RELATION_DIM)
    tcn = TCN(245760, [128,128,64,TCN_OUT_CHANNEL])

    # Move models to GPU
    encoder.to(device)
    rn.to(device)
    rn0.to(device)
    tcn.to(device)

    # Load Saved Model
    if encoder_saved_model != "":
        encoder.load_state_dict(torch.load(encoder_saved_model))
    if rn_saved_model != "":
        rn.load_state_dict(torch.load(rn_saved_model))
    if tcn_saved_model != "":
        tcn.load_state_dict(torch.load(tcn_saved_model))
    if rn0_saved_model != "":
        rn0.load_state_dict(torch.load(rn0_saved_model))

    # Testing
    with torch.no_grad():
        accuracies = []
        for test_episode in range(TEST_EPISODE):
            print("Test_Epi[{}]".format(test_episode), end="\t")
            
            # Data Loading
            haaDataset = dataset.HAADataset(DATA_FOLDERS, None, "test", CLASS_NUM, SAMPLE_NUM, INST_NUM, FRAME_NUM, CLIP_NUM, WINDOW_NUM)
            sample_dataloader = dataset.get_HAA_data_loader(haaDataset,num_per_class=SAMPLE_NUM)
            batch_dataloader = dataset.get_HAA_data_loader(haaDataset,num_per_class=QUERY_NUM,shuffle=True)
            try:
                samples, _ = sample_dataloader.__iter__().next()            # [batch, clip, RGB, frame, H, W]
                batches, batches_labels = batch_dataloader.__iter__().next()   # [batch, window*clip, RGB, frame, H, W]
            except Exception:
                print("Skipped")
                continue

            total_rewards = 0
            total_num_covered = CLASS_NUM * QUERY_NUM
            
            # Encoding
            samples = samples.view(CLASS_NUM*SAMPLE_NUM*WINDOW_NUM*CLIP_NUM, 3, FRAME_NUM, 128, 128)
            samples = encoder(Variable(samples).to(device))
            samples = samples.view(CLASS_NUM*SAMPLE_NUM*WINDOW_NUM, CLIP_NUM, -1)    # [support*class*window, clip, feature]

            batches = batches.view(CLASS_NUM*QUERY_NUM*WINDOW_NUM*CLIP_NUM, 3, FRAME_NUM, 128, 128)
            batches = encoder(Variable(batches).to(device))
            batches = batches.view(CLASS_NUM*QUERY_NUM*WINDOW_NUM,CLIP_NUM,-1)       # [query*class*window, clip, feature]

            # TCN Processing
            samples = torch.transpose(samples,1,2)       # [support*class*window, feature(channel), clip(length)]
            samples = tcn(samples)
            samples = torch.transpose(samples,1,2)       # [support*class*window, clip, feature]
            samples = samples.view(CLASS_NUM, SAMPLE_NUM, WINDOW_NUM, CLIP_NUM, -1)  # [class, sample, window, clip, feature]
            samples = torch.sum(samples,1).squeeze(1)    # [class, window, clip, feature]
            samples, _ = torch.max(samples, 1)           # [class, clip, feature]

            batches = torch.transpose(batches,1,2)       # [query*class*window, feature(channel), clip(length)]
            batches = tcn(batches)
            batches = torch.transpose(batches,1,2)       # [query*class*window, clip, feature]

            # Compute Relation
            samples_rn = samples.unsqueeze(0).repeat(QUERY_NUM*CLASS_NUM*WINDOW_NUM,1,1,1)
            batches_rn = batches.unsqueeze(0).repeat(CLASS_NUM,1,1,1)
            batches_rn = torch.transpose(batches_rn,0,1)                      # [query*class*window, class, clip(length), feature(channel)]
            relations = torch.cat((samples_rn,batches_rn),2).view(-1,CLIP_NUM*2,TCN_OUT_CHANNEL)    # [query*class*window, clip*2(channel), feature]
            relations = rn(relations).view(QUERY_NUM*CLASS_NUM, WINDOW_NUM, CLASS_NUM)    # [query*class, window, class]

            # Compute Zero Probability
            samples_rn0 = samples.reshape(CLASS_NUM*CLIP_NUM, -1).unsqueeze(0).repeat(QUERY_NUM*CLASS_NUM*WINDOW_NUM,1,1)
            relations_rn0 = torch.cat((batches, samples_rn0), 1)
            blank_prob = rn0(relations_rn0).view(QUERY_NUM*CLASS_NUM, WINDOW_NUM, 1)

            # Generate final probabilities
            relations = torch.cat((blank_prob, relations), 2)
            final_outcome = nn.functional.softmax(relations, 2)  # [query*class, window(length), class+1]

            # Predict
            predict_labels = ctc_predict(final_outcome.cpu().numpy())
            batches_labels = batches_labels.numpy()

            # Counting Correct Ones
            rewards = [compute_score(prediction, truth) for prediction, truth in zip(predict_labels, batches_labels)]
            total_rewards += np.sum(rewards)

            # Record accuracy
            accuracy = total_rewards/total_num_covered
            accuracies.append(accuracy)
            print("Accuracy = {}".format(accuracy))
            print(batches_labels)
            print(predict_labels)
            print()

    
        # Overall accuracy
        test_accuracy, _ = mean_confidence_interval(accuracies)
        print("Final_Accu = {}".format(test_accuracy))

# Program Starts
if __name__ == '__main__':
    main()
