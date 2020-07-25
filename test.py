# Public Packages
import torch                                     #
import torch.nn as nn                            #  Pytorch
from torch.autograd import Variable              #

import numpy as np                               #  Numpy
import os                                        #  OS
import sys

# Local Packages
from relationNet import RelationNetwork as RN    #  Relation Network
from i3d import InceptionI3d as I3D              #  I3D
from i3d import Simple3DEncoder as C3D           #  Conv3D
from tcn import TemporalConvNet as TCN           #  TCN
import dataset                                   #  Task Generator
import utils                                     #  Helper Functions

# Constant (Settings)
TCN_OUT_CHANNEL = 64                    # Num of channels of output of TCN
RELATION_DIM = 32                       # Dim of one layer of relation net
CLASS_NUM = 3                           # <X>-way  | Num of classes
SAMPLE_NUM_PER_CLASS = 7                # <Y>-shot | Num of supports per class
BATCH_NUM_PER_CLASS = 5                 # Num of instances for validation per class
TEST_EPISODE = 200                      # Num of validation episode
NUM_FRAME_PER_CLIP = 10                 # Num of frames per clip
NUM_CLIP = 5                            # Num of clips per video
NUM_INST = 10                           # Num of videos selected in each class

DATA_FOLDER = "/data/ssongad/haa/new_normalized_frame/"           # Data path
encoder_saved_model = "/data/ssongad/codes/tcn/models/TCN_Simple3DEnconder_01/encoder_0.7466666666666667.pkl"       # Path of saved encoder model
rn_saved_model = "/data/ssongad/codes/tcn/models/TCN_Simple3DEnconder_01/rn_0.7466666666666667.pkl"                 # Path of saved relation net model
tcn_saved_model = "/data/ssongad/codes/tcn/models/TCN_Simple3DEnconder_01/tcn_0.7466666666666667.pkl"               # Path of saved tcn model

# Device to be used
os.environ['CUDA_VISIBLE_DEVICES']="3,7"                          # GPU to be used
device = torch.device('cuda')

def main():

    # Define models
    # encoder = I3D(in_channels=3)
    encoder = C3D(in_channels=3)
    encoder = nn.DataParallel(encoder)
    rn = RN(NUM_CLIP,RELATION_DIM)
    tcn = TCN(245760, [128,128,64,TCN_OUT_CHANNEL])
    mse = nn.MSELoss()

    # Move models to GPU
    encoder.to(device)
    rn.to(device)
    tcn.to(device)
    mse.to(device)

    # Load Saved Model
    if encoder_saved_model != "":
        encoder.load_state_dict(torch.load(encoder_saved_model))
    if rn_saved_model != "":
        rn.load_state_dict(torch.load(rn_saved_model))
    if tcn_saved_model != "":
        tcn.load_state_dict(torch.load(tcn_saved_model))

    # Testing
    with torch.no_grad():
        accuracies = []
        for test_episode in range(TEST_EPISODE):
            print("Test_Epi[{}]".format(test_episode), end="\t")
            
            # Data Loading
            total_rewards = 0
            total_num_covered = CLASS_NUM * BATCH_NUM_PER_CLASS
            haaDataset = dataset.HAADataset(DATA_FOLDER, "test", CLASS_NUM, SAMPLE_NUM_PER_CLASS, NUM_INST, NUM_FRAME_PER_CLIP, NUM_CLIP)
            sample_dataloader = dataset.get_HAA_data_loader(haaDataset,num_per_class=SAMPLE_NUM_PER_CLASS)
            val_dataloader = dataset.get_HAA_data_loader(haaDataset,num_per_class=BATCH_NUM_PER_CLASS,shuffle=True)
            samples, _ = sample_dataloader.__iter__().next()
            batches, batches_labels = val_dataloader.__iter__().next()
            
            # Encoding
            samples = samples.view(CLASS_NUM*SAMPLE_NUM_PER_CLASS*NUM_CLIP, 3, NUM_FRAME_PER_CLIP, 128, 128)
            samples = encoder(Variable(samples).to(device))
            samples = samples.view(CLASS_NUM,SAMPLE_NUM_PER_CLASS,NUM_CLIP,-1)
            samples = torch.sum(samples,1).squeeze(1)
            batches = batches.view(CLASS_NUM*BATCH_NUM_PER_CLASS*NUM_CLIP, 3, NUM_FRAME_PER_CLIP, 128, 128)
            batches = encoder(Variable(batches).to(device))
            batches = batches.view(CLASS_NUM*BATCH_NUM_PER_CLASS,NUM_CLIP,-1)

            # TCN Processing
            samples = torch.transpose(samples,1,2)
            samples = tcn(samples)
            samples = torch.transpose(samples,1,2)
            batches = torch.transpose(batches,1,2)
            batches = tcn(batches)
            batches = torch.transpose(batches,1,2) #[batch, clip, feature]

            # Compute Relation
            samples = samples.unsqueeze(0).repeat(BATCH_NUM_PER_CLASS*CLASS_NUM,1,1,1)
            batches = batches.unsqueeze(0).repeat(CLASS_NUM,1,1,1)
            batches = torch.transpose(batches,0,1)
            relations = torch.cat((samples,batches),2).view(-1,NUM_CLIP*2,TCN_OUT_CHANNEL)
            relations = rn(relations).view(-1,CLASS_NUM)

            # Predict
            _, predict_labels = torch.max(relations.data,1)

            # Counting Correct Ones
            rewards = [1 if predict_labels[j]==batches_labels[j] else 0 for j in range(len(predict_labels))]
            total_rewards += np.sum(rewards)

            # Record accuracy
            accuracy = total_rewards/total_num_covered
            accuracies.append(accuracy)
            print("Accuracy = {}".format(accuracy))
            print(batches_labels)
            print(predict_labels)
            print()

    
        # Overall accuracy
        test_accuracy, _ = utils.mean_confidence_interval(accuracies)
        print("Final_Accu = {}".format(test_accuracy))

# Program Starts
if __name__ == '__main__':
    main()
