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
SAMPLE_NUM_PER_CLASS = 5                # <Y>-shot | Num of supports per class
BATCH_NUM_PER_CLASS = 3                 # Num of instances for validation per class
TEST_EPISODE = 50                 # Num of validation episode
NUM_FRAME_PER_CLIP = 10                 # Num of frames per clip
NUM_CLIP = 5                            # Num of clips per window
NUM_WINDOW = 3                          # Num of processing window per video
NUM_INST = 10                           # Num of videos selected in each class

DATA_FOLDERS = ["/data/ssongad/haa/new_normalized_frame/",        #
                "/data/ssongad/haa/normalized_frame_scale2",      # Data path => [original, 2x, 3x]
                "/data/ssongad/haa/normalized_frame_scale3"]      #

encoder_saved_model = "/data/ssongad/codes/ctc/models/TCN_Simple3DEnconder_MSE/TCN_Simple3DEnconder_01/encoder_0.7466666666666667.pkl"          # Path of saved encoder model
rn_saved_model = "/data/ssongad/codes/ctc/models/TCN_Simple3DEnconder_MSE/TCN_Simple3DEnconder_01/rn_0.7466666666666667.pkl"               # Path of saved relation net model
tcn_saved_model = "/data/ssongad/codes/ctc/models/TCN_Simple3DEnconder_MSE/TCN_Simple3DEnconder_01/tcn_0.7466666666666667.pkl"              # Path of saved tcn model
rn0_saved_model = ""              # Path of saved rn0 model

# Device to be used
os.environ['CUDA_VISIBLE_DEVICES']="3,7"                          # GPU to be used
device = torch.device('cuda')

def main():

    # Define models
    encoder = C3D(in_channels=3)
    encoder = nn.DataParallel(encoder)
    rn = RN(NUM_CLIP, RELATION_DIM)
    rn0 = RN0(NUM_CLIP*(CLASS_NUM+1), RELATION_DIM)
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
            haaDataset_support = dataset.HAADataset(DATA_FOLDERS, None, "train", "support", CLASS_NUM, SAMPLE_NUM_PER_CLASS, NUM_INST, NUM_FRAME_PER_CLIP, NUM_CLIP, NUM_WINDOW)
            haaDataset_query = dataset.HAADataset(DATA_FOLDERS, haaDataset_support.get_classes(), "train", "query", CLASS_NUM, SAMPLE_NUM_PER_CLASS, NUM_INST, NUM_FRAME_PER_CLIP, NUM_CLIP, NUM_WINDOW)
            sample_dataloader = dataset.get_HAA_data_loader(haaDataset_support,num_per_class=SAMPLE_NUM_PER_CLASS)
            batch_dataloader = dataset.get_HAA_data_loader(haaDataset_query,num_per_class=BATCH_NUM_PER_CLASS,shuffle=True)
            try:
                samples, _ = sample_dataloader.__iter__().next()            # [batch, clip, RGB, frame, H, W]
                batches, batch_labels = batch_dataloader.__iter__().next()   # [batch, window*clip, RGB, frame, H, W]
            except Exception:
                print("Skipped")
                continue

            total_rewards = 0
            total_num_covered = CLASS_NUM * BATCH_NUM_PER_CLASS
            
            # Encoding
            samples = samples.view(CLASS_NUM*SAMPLE_NUM_PER_CLASS*NUM_CLIP, 3, NUM_FRAME_PER_CLIP, 128, 128)
            samples = encoder(Variable(samples).to(device))
            samples = samples.view(CLASS_NUM,SAMPLE_NUM_PER_CLASS,NUM_CLIP,-1)
            samples = torch.sum(samples,1).squeeze(1)                              # [class, clip, feature]

            batches = batches.view(CLASS_NUM*BATCH_NUM_PER_CLASS*NUM_WINDOW*NUM_CLIP, 3, NUM_FRAME_PER_CLIP, 128, 128)
            batches = encoder(Variable(batches).to(device))
            batches = batches.view(CLASS_NUM*BATCH_NUM_PER_CLASS*NUM_WINDOW,NUM_CLIP,-1) # [batch*window, clip, feature]

            # TCN Processing
            samples = torch.transpose(samples,1,2)       # [class, feature(channel), clip(length)]
            samples = tcn(samples)
            samples = torch.transpose(samples,1,2)       # [class, clip, feature]

            batches = torch.transpose(batches,1,2)       # [batch*window, feature(channel), clip(length)]
            batches = tcn(batches)
            batches = torch.transpose(batches,1,2)       # [batch*window, clip, feature]

            # Compute Relation
            samples_rn = samples.unsqueeze(0).repeat(BATCH_NUM_PER_CLASS*CLASS_NUM*NUM_WINDOW,1,1,1)
            batches_rn = batches.unsqueeze(0).repeat(CLASS_NUM,1,1,1)      
            batches_rn = torch.transpose(batches_rn,0,1)                      # [batch*window, class, clip(length), feature(channel)]
            relations = torch.cat((samples_rn,batches_rn),2).view(-1,NUM_CLIP*2,TCN_OUT_CHANNEL)    # [batch*window*class, clip*2(channel), feature]
            relations = rn(relations).view(BATCH_NUM_PER_CLASS*CLASS_NUM, NUM_WINDOW, CLASS_NUM)    # [batch, window, class]

            # Compute Zero Probability
            samples_rn0 = samples.reshape(CLASS_NUM*NUM_CLIP, -1).unsqueeze(0).repeat(BATCH_NUM_PER_CLASS*CLASS_NUM*NUM_WINDOW,1,1)
            relations_rn0 = torch.cat((batches, samples_rn0), 1)
            blank_prob = rn0(relations_rn0).view(BATCH_NUM_PER_CLASS*CLASS_NUM, NUM_WINDOW, 1)

            # Generate final probabilities
            relations = torch.cat((blank_prob, relations), 2)
            final_outcome = nn.functional.softmax(relations, 2)  # [batch, window(length), class+1]

            # Predict
            predict_labels = ctc_predict(final_outcome.cpu().numpy())
            batch_labels = batch_labels.numpy()

            # Counting Correct Ones
            rewards = [compute_score(prediction, truth) for prediction, truth in zip(predict_labels, batch_labels)]
            total_rewards += np.sum(rewards)

            # Record accuracy
            accuracy = total_rewards/total_num_covered
            accuracies.append(accuracy)
            print("Accuracy = {}".format(accuracy))
            print(batch_labels)
            print(predict_labels)
            print()

    
        # Overall accuracy
        test_accuracy, _ = utils.mean_confidence_interval(accuracies)
        print("Final_Accu = {}".format(test_accuracy))

# Program Starts
if __name__ == '__main__':
    main()
