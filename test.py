# Public Packages
import torch                                     #
import torch.nn as nn                            #  Pytorch
from torch.optim.lr_scheduler import StepLR      #
from torch.autograd import Variable              #

import argparse                                  #  Argument
import numpy as np                               #  Numpy
import os                                        #  OS
import sys

# Local Packages
from relationNet import RelationNetwork as RN    #  Relation Network
from i3d import InceptionI3d as I3D              #  I3D
import dataset                                   #  Task Generator
import utils                                     #  Helper Functions

# Constant
FEATURE_DIM = 32                         # Dim of output of encoder
RELATION_DIM = 16                        # Dim of one layer of relation net
CLASS_NUM = 3                            # <X>-way
SAMPLE_NUM_PER_CLASS = 5                 # <Y>-shot
BATCH_NUM_PER_CLASS = 5                  # Batch size
TEST_EPISODE = 100                       # Num of testing episode
NUM_FRAME_PER_CLIP = 16                  # Num of frame in each clip
SEQ_LEN = 3                              # Sequence Length for LSTM
NUM_INST = 10                            # Num of videos selected in each class
DATA_FOLDER = "/data/ssongad/haa/new_normalized_frame/"                # Data path
encoder_saved_model = "/data/ssongad/codes/lstm/models/wrong_optim/encoder_3way_5shot_0.8093333333333333.pkl"                                          # Path of saved encoder model
rn_saved_model = "/data/ssongad/codes/lstm/models/wrong_optim/rn_3way_5shot_0.8093333333333333.pkl"                                               # Path of saved relation net model
lstm_saved_model = "/data/ssongad/codes/lstm/models/wrong_optim/lstm_3way_5shot_0.8093333333333333.pkl"                                             # Path of saved lstm model
# Device to be used
os.environ['CUDA_VISIBLE_DEVICES']="3,7"                        # GPU to be used
device = torch.device('cuda')

def main():

    # i3d == the I3D network
    encoder = I3D(in_channels=3)
    encoder = nn.DataParallel(encoder)
    # rn == the relation network
    rn = RN(FEATURE_DIM,RELATION_DIM)
    # lstm == LSTM
    lstm = nn.LSTM(26624, 864, 2, batch_first=True)

    # Move the model to GPU
    encoder.to(device)
    rn.to(device)
    lstm.to(device)

    # Load Saved Model
    if encoder_saved_model != "":
        encoder.load_state_dict(torch.load(encoder_saved_model))
    if rn_saved_model != "":
        rn.load_state_dict(torch.load(rn_saved_model))
    if lstm_saved_model != "":
        lstm.load_state_dict(torch.load(lstm_saved_model))

    # Validation
    with torch.no_grad():
        accuracies = []
        for test_episode in range(TEST_EPISODE):
            print("Test_Epi[{}]".format(test_episode), end="\t")
            
            # Data Loading #TODO
            total_rewards = 0
            total_num_tested = CLASS_NUM * BATCH_NUM_PER_CLASS
            haaDataset = dataset.HAADataset(DATA_FOLDER, "test", CLASS_NUM, SAMPLE_NUM_PER_CLASS, NUM_INST, NUM_FRAME_PER_CLIP, SEQ_LEN)
            print("Classes", haaDataset.class_names, end="\t")
            sample_dataloader = dataset.get_HAA_data_loader(haaDataset,num_per_class=SAMPLE_NUM_PER_CLASS)
            test_dataloader = dataset.get_HAA_data_loader(haaDataset,num_per_class=BATCH_NUM_PER_CLASS,shuffle=True)
            samples, _ = sample_dataloader.__iter__().next()
            batches, batches_labels = test_dataloader.__iter__().next()
            
            # Encoding
            samples = samples.view(CLASS_NUM*SAMPLE_NUM_PER_CLASS*SEQ_LEN, 3, NUM_FRAME_PER_CLIP, 128, 128)
            samples = encoder(Variable(samples).to(device))
            samples = samples.view(CLASS_NUM,SAMPLE_NUM_PER_CLASS,SEQ_LEN,-1)
            samples = torch.sum(samples,1).squeeze(1)
            batches = batches.view(CLASS_NUM*BATCH_NUM_PER_CLASS*SEQ_LEN, 3, NUM_FRAME_PER_CLIP, 128, 128)
            batches = encoder(Variable(batches).to(device))
            batches = batches.view(CLASS_NUM*BATCH_NUM_PER_CLASS,SEQ_LEN,-1)

            # LSTM Processing
            samples_hidden = torch.rand(2,CLASS_NUM,864).to(device)
            samples_cell = torch.rand(2,CLASS_NUM,864).to(device)
            batches_hidden = torch.rand(2,CLASS_NUM*BATCH_NUM_PER_CLASS,864).to(device)
            batches_cell = torch.rand(2,CLASS_NUM*BATCH_NUM_PER_CLASS,864).to(device)
            samples, _ = lstm(samples, (samples_hidden,samples_cell))
            batches, _ = lstm(batches, (batches_hidden,batches_cell)) 

            # Compute Relation
            samples = samples.unsqueeze(0).repeat(BATCH_NUM_PER_CLASS*CLASS_NUM,1,1,1)
            batches = batches.unsqueeze(0).repeat(CLASS_NUM,1,1,1)
            batches = torch.transpose(batches,0,1)

            relations = torch.cat((samples,batches),2).view(-1,FEATURE_DIM*2,9,9)
            relations = rn(relations).view(-1,CLASS_NUM)

            # Predict
            _, predict_labels = torch.max(relations.data,1)

            # Counting Correct Ones
            rewards = [1 if predict_labels[j]==batches_labels[j] else 0 for j in range(len(predict_labels))]
            total_rewards += np.sum(rewards)

            # Record accuracy
            accuracy = total_rewards/1.0/total_num_tested
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
