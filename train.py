# Public Packages
import torch                                     #
import torch.nn as nn                            #  Pytorch
from torch.optim.lr_scheduler import StepLR      #
from torch.autograd import Variable              #

import numpy as np                               #  Numpy
import os                                        #  
import sys                                       #  OS

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
SAMPLE_NUM_PER_CLASS = 5                # <Y>-shot | Num of supports per class
BATCH_NUM_PER_CLASS = 5                 # Num of instances for validation per class
TRAIN_EPISODE = 20000                   # Num of training episode 
VALIDATION_EPISODE = 50                 # Num of validation episode
VALIDATION_FREQUENCY = 200              # After each <X> training episodes, do validation once
LEARNING_RATE = 0.001                   # Initial learning rate
NUM_FRAME_PER_CLIP = 10                 # Num of frames per clip
NUM_CLIP = 5                            # Num of clips per video
NUM_INST = 10                           # Num of videos selected in each class

EXP_NAMES = ["TCN_Simple3DEnconder_01"]

EXP_NAME = "TCN_Simple3DEnconder_01"                                # Name of this experiment
DATA_FOLDER = "/data/ssongad/haa/new_normalized_frame/"           # Data path
encoder_saved_model = ""                                          # Path of saved encoder model
rn_saved_model = ""                                               # Path of saved relation net model
tcn_saved_model = ""                                              # Path of saved tcn model

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

    # Define Optimizer
    encoder_optim = torch.optim.Adam(encoder.parameters(),lr=LEARNING_RATE)
    rn_optim = torch.optim.Adam(rn.parameters(),lr=LEARNING_RATE)
    tcn_optim = torch.optim.Adam(tcn.parameters(),lr=LEARNING_RATE)

    # Define Scheduler
    encoder_scheduler = StepLR(encoder_optim,step_size=2000,gamma=0.5)
    rn_scheduler = StepLR(rn_optim,step_size=2000,gamma=0.5)
    tcn_scheduler = StepLR(tcn_optim,step_size=2000,gamma=0.5)

    # Load Saved Model
    if encoder_saved_model != "":
        encoder.load_state_dict(torch.load(encoder_saved_model))
    if rn_saved_model != "":
        rn.load_state_dict(torch.load(rn_saved_model))
    if tcn_saved_model != "":
        tcn.load_state_dict(torch.load(tcn_saved_model))
    
    max_accuracy = 0                # Currently the best accuracy
    accuracy_history = []           # Only for logging

    # Prepare output folder
    if EXP_NAME == "":
        output_folder = "./models/"
    else:
        output_folder = "./models/" + EXP_NAME + "/"
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    # Training Loop
    for episode in range(TRAIN_EPISODE):

        print("Train_Epi[{}] Pres_Accu = {}".format(episode, max_accuracy), end="\t")

        # Update "step" for scheduler
        rn_scheduler.step(episode)
        encoder_scheduler.step(episode)
        tcn_scheduler.step(episode)

        # Setup Data
        haaDataset = dataset.HAADataset(DATA_FOLDER, "train", CLASS_NUM, SAMPLE_NUM_PER_CLASS, NUM_INST, NUM_FRAME_PER_CLIP, NUM_CLIP)
        # print("Classes", haaDataset.class_names, end="\t")
        sample_dataloader = dataset.get_HAA_data_loader(haaDataset,num_per_class=SAMPLE_NUM_PER_CLASS)
        batch_dataloader = dataset.get_HAA_data_loader(haaDataset,num_per_class=BATCH_NUM_PER_CLASS,shuffle=True)
        samples, _ = sample_dataloader.__iter__().next()
        batches,batch_labels = batch_dataloader.__iter__().next()   # [batch, clip, RGB, frame, H, W]

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

        # Compute Loss
        one_hot_labels = Variable(torch.zeros(BATCH_NUM_PER_CLASS*CLASS_NUM, CLASS_NUM).scatter_(1, batch_labels.view(-1,1), 1).to(device))
        loss = mse(relations,one_hot_labels)
        print("Loss = {}".format(loss))

        # Back Propagation
        encoder.zero_grad()
        rn.zero_grad()
        tcn.zero_grad()
        loss.backward()

        # Clip Gradient
        nn.utils.clip_grad_norm_(encoder.parameters(),0.5)
        nn.utils.clip_grad_norm_(rn.parameters(),0.5)
        nn.utils.clip_grad_norm_(tcn.parameters(),0.5)

        # Update Models
        encoder_optim.step()
        rn_optim.step()
        tcn_optim.step()

        # Validation Loop
        if (episode % VALIDATION_FREQUENCY == 0 and episode != 0) or episode == TRAIN_EPISODE-1:
            with torch.no_grad():
                accuracies = []

                for validation_episode in range(VALIDATION_EPISODE):
                    print("Val_Epi[{}] Pres_Accu = {}".format(validation_episode, max_accuracy), end="\t")
                    
                    # Data Loading #TODO
                    total_rewards = 0
                    total_num_covered = CLASS_NUM * BATCH_NUM_PER_CLASS
                    haaDataset = dataset.HAADataset(DATA_FOLDER, "test", CLASS_NUM, SAMPLE_NUM_PER_CLASS, NUM_INST, NUM_FRAME_PER_CLIP, NUM_CLIP)
                    print("Classes", haaDataset.class_names)
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
            
                # Overall accuracy
                val_accuracy, _ = utils.mean_confidence_interval(accuracies)
                accuracy_history.append(val_accuracy)
                print("Val_Accu = {}".format(val_accuracy))

                # Write history
                file = open("accuracy_log.txt", "w")
                file.write(str(accuracy_history))
                file.close()

                # Save Model
                if val_accuracy > max_accuracy:
                    # save networks
                    torch.save(encoder.state_dict(), output_folder + "encoder_"+str(val_accuracy)+".pkl")
                    torch.save(rn.state_dict(), output_folder + "rn_"+str(val_accuracy)+".pkl")
                    torch.save(tcn.state_dict(), output_folder + "tcn_"+str(val_accuracy)+".pkl")

                    max_accuracy = val_accuracy
                    print("Model Saved with accuracy={}".format(max_accuracy))
    
    print("final accuracy = {}".format(max_accuracy))

# Program Starts
if __name__ == '__main__':
    main()
