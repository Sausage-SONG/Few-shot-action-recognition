# Public Packages
import torch                                     #
import torch.nn as nn                            #  Pytorch
from torch.optim.lr_scheduler import StepLR      #
from torch.autograd import Variable              #

import argparse                                  #  Argument
import numpy as np                               #  Numpy
import os                                        #  OS

# Local Packages
from relationNet import RelationNetwork as RN    #  Relation Network
from i3d import InceptionI3d as I3D              #  I3D
import dataset                                   #  Task Generator
import utils                                     #  Helper Functions

# Arguments Parser setup
parser = argparse.ArgumentParser()                                              #
parser.add_argument("-e","--episode",type = int, default= 10000)                #
parser.add_argument("-t","--test_episode", type = int, default = 100)           #
parser.add_argument("-w","--class_num",type = int, default = 3)                 # For definition
parser.add_argument("-s","--sample_num_per_class",type = int, default = 5)      # Check #Constant#
parser.add_argument("-b","--batch_num_per_class",type = int, default = 5)       #
parser.add_argument("-l","--learning_rate", type = float, default = 0.001)      # 
parser.add_argument("-n","--num_frame_per_video",type = int, default = 32)      #

# Parse arguments
args = parser.parse_args()

# Constant
FEATURE_DIM = 1024 #TODO
RELATION_DIM = 32 #TODO
CLASS_NUM = args.class_num                             # <X>-way
SAMPLE_NUM_PER_CLASS = args.sample_num_per_class       # <Y>-shot
BATCH_NUM_PER_CLASS = args.batch_num_per_class         # Batch size
EPISODE = args.episode                                 # Num of training episode 
TEST_EPISODE = args.test_episode                       # Num of testing episode
LEARNING_RATE = args.learning_rate                     # Learning rate
NUM_FRAME_PER_VIDEO = args.num_frame_per_video         # Num of frame in each video
NUM_INST = 10
DATA_FOLDER = "/data/ssongad/hmdb51/yolo_frame/"
encoder_saved_model = "" #"/data/ssongad/codes/model_naive/models/HAA/encoder_3way_5shot_0.664.pkl"
rn_saved_model = ""      #"/data/ssongad/codes/model_naive/models/HAA/rn_3way_5shot_0.664.pkl"
# Device to be used
# os.environ['CUDA_VISIBLE_DEVICES']="0"             # GPU to be used
device = torch.device('cuda')#:'+str(gpu_index))

def main():

    # i3d == the I3D network
    encoder = I3D(in_channels=3)
    # encoder = nn.DataParallel(encoder)
    # rn == the relation network
    rn = RN(FEATURE_DIM,RELATION_DIM)

    # Move the model to GPU
    encoder.to(device)
    rn.to(device)

    # Define Optimizer
    encoder_optim = torch.optim.Adam(encoder.parameters(),lr=LEARNING_RATE)
    rn_optim = torch.optim.Adam(rn.parameters(),lr=LEARNING_RATE)

    # Define Scheduler
    encoder_scheduler = StepLR(encoder_optim,step_size=100000,gamma=0.5)
    rn_scheduler = StepLR(rn_optim,step_size=100000,gamma=0.5)

    # Load Saved Model
    if encoder_saved_model != "":
        encoder.load_state_dict(torch.load(encoder_saved_model))
    if rn_saved_model != "":
        rn.load_state_dict(torch.load(rn_saved_model))
    max_accuracy = 0

    for episode in range(EPISODE):
        print("Train_Epi[{}] Pres_Accu = {}".format(episode, max_accuracy), end="\t")

        # Update "step" for scheduler
        rn_scheduler.step(episode)
        encoder_scheduler.step(episode)

        # Setup Data #TODO
        haaDataset = dataset.HAADataset(DATA_FOLDER, "train", CLASS_NUM, SAMPLE_NUM_PER_CLASS, NUM_INST, NUM_FRAME_PER_VIDEO)
        print("Classes", haaDataset.class_names)
        sample_dataloader = dataset.get_HAA_data_loader(haaDataset,num_per_class=SAMPLE_NUM_PER_CLASS)
        batch_dataloader = dataset.get_HAA_data_loader(haaDataset,num_per_class=BATCH_NUM_PER_CLASS)
        samples, _ = sample_dataloader.__iter__().next()
        batches,batch_labels = batch_dataloader.__iter__().next()

        # Encoding
        #sample_features = encoder(Variable(samples).to(device))
        samples = encoder(Variable(samples).to(device))
        samples = samples.view(CLASS_NUM,SAMPLE_NUM_PER_CLASS,FEATURE_DIM,8,8)
        samples = torch.sum(samples,1).squeeze(1)
        #batch_features = encoder(Variable(batches).to(device))
        batches = encoder(Variable(batches).to(device))
        batches = batches.view(CLASS_NUM*BATCH_NUM_PER_CLASS,FEATURE_DIM,8,8)

        # Compute Relation
        #sample_features_ext = sample_features.unsqueeze(0).repeat(BATCH_NUM_PER_CLASS*CLASS_NUM,1,1,1,1)      #
        samples = samples.unsqueeze(0).repeat(BATCH_NUM_PER_CLASS*CLASS_NUM,1,1,1,1)
        #batch_features_ext = batch_features.unsqueeze(0).repeat(CLASS_NUM,1,1,1,1)                        #
        batches = batches.unsqueeze(0).repeat(CLASS_NUM,1,1,1,1)
        batches = torch.transpose(batches,0,1)                                         # TODO   
        relations = torch.cat((samples,batches),2).view(-1,FEATURE_DIM*2,8,8)   #
        relations = rn(relations).view(-1,CLASS_NUM)
        return                                                 #

        # Compute Loss
        mse = nn.MSELoss().to(device)                                                                                                       #
        one_hot_labels = Variable(torch.zeros(BATCH_NUM_PER_CLASS*CLASS_NUM, CLASS_NUM).scatter_(1, batch_labels.view(-1,1), 1).to(device)) # TODO
        loss = mse(relations,one_hot_labels)                                                                                                #

        # Train Model
        encoder.zero_grad()
        rn.zero_grad()
        loss.backward()

        nn.utils.clip_grad_norm(encoder.parameters(),0.5)
        nn.utils.clip_grad_norm(rn.parameters(),0.5)

        encoder_optim.step()
        rn_optim.step()

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
                    haaDataset = dataset.HAADataset(DATA_FOLDER, "test", CLASS_NUM, SAMPLE_NUM_PER_CLASS, NUM_INST, NUM_FRAME_PER_VIDEO)
                    print("Classes", haaDataset.class_names)
                    sample_dataloader = dataset.get_HAA_data_loader(haaDataset,num_per_class=SAMPLE_NUM_PER_CLASS)  #
                    test_dataloader = dataset.get_HAA_data_loader(haaDataset,num_per_class=BATCH_NUM_PER_CLASS)            #
                    samples, _ = sample_dataloader.__iter__().next()
                    batches, batches_labels = test_dataloader.__iter__().next()
                    
                    samples = encoder(Variable(samples).to(device)) # 5x64                       #
                    samples = samples.view(CLASS_NUM,SAMPLE_NUM_PER_CLASS,FEATURE_DIM,8,8)   # TODO
                    samples = torch.sum(samples,1).squeeze(1)                                              #
                        
                    # Encoding
                    samples = samples.unsqueeze(0).repeat(CLASS_NUM*BATCH_NUM_PER_CLASS,1,1,1,1)
                    batches = encoder(Variable(batches).to(device)) # 20x64                          #
                    batches = batches.view(CLASS_NUM*BATCH_NUM_PER_CLASS,FEATURE_DIM,8,8)

                    # Compute Relation
                    batches = batches.unsqueeze(0).repeat(1*CLASS_NUM,1,1,1,1)                            #
                    batches = torch.transpose(batches,0,1)                                            # TODO
                    relations = torch.cat((samples,batches),2).view(-1,FEATURE_DIM*2,8,8)    #
                    relations = rn(relations).view(-1,CLASS_NUM)                                                     #

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
                print("Test_Accu = {}".format(test_accuracy))

                # Save Model
                if test_accuracy > max_accuracy:
                    # save networks
                    torch.save(encoder.state_dict(),str("./models/encoder_" + str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot_" + str(test_accuracy)+".pkl"))
                    torch.save(rn.state_dict(),str("./models/rn_"+ str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot_" + str(test_accuracy)+".pkl"))

                    max_accuracy = test_accuracy
                    print("Model Saved with accuracy={}".format(max_accuracy))
    
    print("final accuracy = {}".format(max_accuracy))

# Program Starts
if __name__ == '__main__':
    main()