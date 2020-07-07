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

# # Arguments Parser setup
# parser = argparse.ArgumentParser()                                              #
# parser.add_argument("-e","--episode",type = int, default= 10000)                #
# parser.add_argument("-t","--test_episode", type = int, default = 100)           #
# parser.add_argument("-w","--class_num",type = int, default = 5)                 # For definition
# parser.add_argument("-s","--sample_num_per_class",type = int, default = 1)      # Check #Constant#
# parser.add_argument("-b","--batch_num_per_class",type = int, default = 3)       #
# parser.add_argument("-l","--learning_rate", type = float, default = 0.001)      # 
# parser.add_argument("-n","--num_frame_per_clip",type = int, default = 32)       #

# # Parse arguments
# args = parser.parse_args()

# Constant
FEATURE_DIM = 32                         # Dim of output of encoder
RELATION_DIM = 16                        # Dim of one layer of relation net
CLASS_NUM = 5                            # <X>-way
SAMPLE_NUM_PER_CLASS = 1                 # <Y>-shot
BATCH_NUM_PER_CLASS = 3                  # Batch size
EPISODE = 20000                          # Num of training episode 
TEST_EPISODE = 100                       # Num of testing episode
LEARNING_RATE = 0.001                    # Initial learning rate
NUM_FRAME_PER_CLIP = 16                  # Num of frame in each clip
SEQ_LEN = 3                              # Sequence Length for LSTM
NUM_INST = 10                            # Num of videos selected in each class
DATA_FOLDER = "/data/ssongad/haa/new_normalized_frame/"           # Data path
encoder_saved_model = ""                                          # Path of saved encoder model
rn_saved_model = ""                                               # Path of saved relation net model
lstm_saved_model = ""                                             # Path of saved lstm model
# Device to be used
os.environ['CUDA_VISIBLE_DEVICES']="1,2,3"                        # GPU to be used
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

    # Define Optimizer
    encoder_optim = torch.optim.Adam(encoder.parameters(),lr=LEARNING_RATE)
    rn_optim = torch.optim.Adam(rn.parameters(),lr=LEARNING_RATE)
    lstm_optim = torch.optim.Adam(rn.parameters(),lr=LEARNING_RATE)

    # Define Scheduler
    encoder_scheduler = StepLR(encoder_optim,step_size=100000,gamma=0.5)
    rn_scheduler = StepLR(rn_optim,step_size=100000,gamma=0.5)
    lstm_scheduler = StepLR(rn_optim,step_size=100000,gamma=0.5)

    # Load Saved Model
    if encoder_saved_model != "":
        encoder.load_state_dict(torch.load(encoder_saved_model))
    if rn_saved_model != "":
        rn.load_state_dict(torch.load(rn_saved_model))
    if lstm_saved_model != "":
        lstm.load_state_dict(torch.load(lstm_saved_model))
    max_accuracy = 0
    accuracy_history = []

    for episode in range(EPISODE):
        print("Train_Epi[{}] Pres_Accu = {}".format(episode, max_accuracy), end="\t")

        # Update "step" for scheduler
        rn_scheduler.step(episode)
        encoder_scheduler.step(episode)
        lstm_scheduler.step(episode)

        # Setup Data #TODO
        haaDataset = dataset.HAADataset(DATA_FOLDER, "train", CLASS_NUM, SAMPLE_NUM_PER_CLASS, NUM_INST, NUM_FRAME_PER_CLIP, SEQ_LEN)
        print("Classes", haaDataset.class_names)
        sample_dataloader = dataset.get_HAA_data_loader(haaDataset,num_per_class=SAMPLE_NUM_PER_CLASS)
        batch_dataloader = dataset.get_HAA_data_loader(haaDataset,num_per_class=BATCH_NUM_PER_CLASS)
        samples, _ = sample_dataloader.__iter__().next()
        batches,batch_labels = batch_dataloader.__iter__().next()

        # Encoding
        samples = samples.view(CLASS_NUM*SAMPLE_NUM_PER_CLASS*SEQ_LEN, 3, NUM_FRAME_PER_CLIP, 128, 128)
        samples = encoder(Variable(samples).to(device))
        samples = samples.view(CLASS_NUM,SAMPLE_NUM_PER_CLASS,SEQ_LEN,-1)
        samples = torch.sum(samples,1).squeeze(1)
        batches = batches.view(CLASS_NUM*BATCH_NUM_PER_CLASS*SEQ_LEN, 3, NUM_FRAME_PER_CLIP, 128, 128)
        batches = encoder(Variable(batches).to(device))
        batches = batches.view(CLASS_NUM*BATCH_NUM_PER_CLASS,SEQ_LEN,-1)

        # LSTM Processing
        samples, _ = lstm(samples)
        batches, _ = lstm(batches)

        # Compute Relation
        samples = samples.unsqueeze(0).repeat(BATCH_NUM_PER_CLASS*CLASS_NUM,1,1,1)
        batches = batches.unsqueeze(0).repeat(CLASS_NUM,1,1,1)
        batches = torch.transpose(batches,0,1)

        relations = torch.cat((samples,batches),2).view(-1,FEATURE_DIM*2,9,9)
        relations = rn(relations).view(-1,CLASS_NUM)

        # Compute Loss
        mse = nn.MSELoss().to(device)
        one_hot_labels = Variable(torch.zeros(BATCH_NUM_PER_CLASS*CLASS_NUM, CLASS_NUM).scatter_(1, batch_labels.view(-1,1), 1).to(device))
        loss = mse(relations,one_hot_labels)

        # Train Model
        encoder.zero_grad()
        rn.zero_grad()
        lstm.zero_grad()
        loss.backward()

        nn.utils.clip_grad_norm(encoder.parameters(),0.5)
        nn.utils.clip_grad_norm(rn.parameters(),0.5)

        encoder_optim.step()
        rn_optim.step()
        lstm_optim.step()

        # Periodically Print Loss #TODO

        # Testing
        if (episode % 200 == 0 and episode != 0) or episode == EPISODE-1:
            with torch.no_grad():
                accuracies = []

                for test_episode in range(TEST_EPISODE):
                    print("Test_Epi[{}] Pres_Accu = {}".format(test_episode, max_accuracy), end="\t")
                    
                    # Data Loading #TODO
                    total_rewards = 0
                    total_num_tested = CLASS_NUM * BATCH_NUM_PER_CLASS
                    haaDataset = dataset.HAADataset(DATA_FOLDER, "test", CLASS_NUM, SAMPLE_NUM_PER_CLASS, NUM_INST, NUM_FRAME_PER_CLIP, SEQ_LEN)
                    print("Classes", haaDataset.class_names)
                    sample_dataloader = dataset.get_HAA_data_loader(haaDataset,num_per_class=SAMPLE_NUM_PER_CLASS)
                    test_dataloader = dataset.get_HAA_data_loader(haaDataset,num_per_class=BATCH_NUM_PER_CLASS)
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
                    samples, _ = lstm(samples)
                    batches, _ = lstm(batches)  

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
            
                # Overall accuracy
                test_accuracy, _ = utils.mean_confidence_interval(accuracies)
                accuracy_history.append(test_accuracy)
                print("Test_Accu = {}".format(test_accuracy))

                # Write history
                file = open("accuracy_log.txt", "w")
                file.write(str(accuracy_history))
                file.close()

                # Save Model
                if test_accuracy > max_accuracy:
                    # save networks
                    torch.save(encoder.state_dict(), str("./models/encoder_" + str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot_" + str(test_accuracy)+".pkl"))
                    torch.save(rn.state_dict(), str("./models/rn_"+ str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot_" + str(test_accuracy)+".pkl"))
                    torch.save(lstm.state_dict(), str("./models/lstm_"+str(CLASS_NUM)+"way_"+ str(SAMPLE_NUM_PER_CLASS) +"shot_" + str(test_accuracy)+".pkl"))

                    max_accuracy = test_accuracy
                    print("Model Saved with accuracy={}".format(max_accuracy))
    
    print("final accuracy = {}".format(max_accuracy))

# Program Starts
if __name__ == '__main__':
    main()
