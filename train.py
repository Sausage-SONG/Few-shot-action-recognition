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
from relationNet import CNNEncoder as cnn        #
import task_generator as tg                      #  Task Generator
import utils                                     #  Helper Functions


# Arguments Parser setup
parser = argparse.ArgumentParser()                                              #
parser.add_argument("-e","--episode",type = int, default= 5000)                 #
parser.add_argument("-t","--test_episode", type = int, default = 100)           #
parser.add_argument("-w","--class_num",type = int, default = 3)                 # For definition
parser.add_argument("-s","--sample_num_per_class",type = int, default = 7)      # Check #Constant#
parser.add_argument("-b","--batch_num_per_class",type = int, default = 10)      #
parser.add_argument("-l","--learning_rate", type = float, default = 0.001)      # 
parser.add_argument("-n","--num_frame_per_video",type = int, default = 32)      #

# Parse arguments
args = parser.parse_args()

# Device to be used
gpu_index = 2                                          # GPU to be used
torch.cuda.set_device(gpu_index)
device = torch.device('cuda:'+str(gpu_index))
# Constant
FEATURE_DIM = 64 #TODO
RELATION_DIM = 8 #TODO
CLASS_NUM = args.class_num                             # <X>-way
SAMPLE_NUM_PER_CLASS = args.sample_num_per_class       # <Y>-shot
BATCH_NUM_PER_CLASS = args.batch_num_per_class         # Batch size
EPISODE = args.episode                                 # Num of training episode 
TEST_EPISODE = args.test_episode                       # Num of testing episode
LEARNING_RATE = args.learning_rate                     # Learning rate
NUM_FRAME_PER_VIDEO = args.num_frame_per_video         # Num of frame in each video


def main():
    utils.print_stage("Program Starts")

    metatrain_folders,metatest_folders = tg.HAA_folders()

    # i3d == the I3D network
    encoder = cnn()
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

    # Load Saved Model #TODO

    # Training Starts Here
    utils.print_stage("Start Training")

    # Accuracy Record
    max_accuracy = 0 #TODO read accuracy from somewhere

    for episode in range(EPISODE):
        print("{}\tepisode{}".format(max_accuracy, episode))

        # Update "step" for scheduler
        rn_scheduler.step(episode)
        encoder_scheduler.step(episode)

        # Setup Data #TODO
        task = tg.HAATask(metatrain_folders,CLASS_NUM,SAMPLE_NUM_PER_CLASS,BATCH_NUM_PER_CLASS)
        sample_dataloader = tg.get_HAA_data_loader(task,num_per_class=SAMPLE_NUM_PER_CLASS,split="train",shuffle=False)
        batch_dataloader = tg.get_HAA_data_loader(task,num_per_class=BATCH_NUM_PER_CLASS,split="test",shuffle=True)
        samples,sample_labels = sample_dataloader.__iter__().next() #25*3*84*84
        batches,batch_labels = batch_dataloader.__iter__().next()

        # Encoding
        sample_features = encoder(Variable(samples).to(device)) # 25*64*19*19                      #
        sample_features = sample_features.view(CLASS_NUM,SAMPLE_NUM_PER_CLASS,FEATURE_DIM,62,62)   # TODO
        sample_features = torch.sum(sample_features,1).squeeze(1)                                  #
        batch_features = encoder(Variable(batches).to(device)) # 20x64*5*5                         #

        # Compute Relation
        sample_features_ext = sample_features.unsqueeze(0).repeat(BATCH_NUM_PER_CLASS*CLASS_NUM,1,1,1,1)      #
        batch_features_ext = batch_features.unsqueeze(0).repeat(CLASS_NUM,1,1,1,1)                            #
        batch_features_ext = torch.transpose(batch_features_ext,0,1)                                          # TODO
        relation_pairs = torch.cat((sample_features_ext,batch_features_ext),2).view(-1,FEATURE_DIM*2,62,62)   #
        relations = rn(relation_pairs).view(-1,CLASS_NUM)                                                     #

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
            utils.print_stage("Testing")
            accuracies = []

            for _ in range(TEST_EPISODE):
                
                # Data Loading #TODO
                total_rewards = 0
                task = tg.HAATask(metatest_folders,CLASS_NUM,SAMPLE_NUM_PER_CLASS,15)                                            #
                sample_dataloader = tg.get_HAA_data_loader(task,num_per_class=SAMPLE_NUM_PER_CLASS,split="train",shuffle=False)  #
                num_per_class = 5                                                                                                # TODO
                test_dataloader = tg.get_HAA_data_loader(task,num_per_class=num_per_class,split="test",shuffle=False)            #
                sample_images,sample_labels = sample_dataloader.__iter__().next()                                                #

                for test_images,test_labels in test_dataloader:
                    batch_size = test_labels.shape[0]
                    # Encoding
                    sample_features = encoder(Variable(sample_images).to(device)) # 5x64                       #
                    sample_features = sample_features.view(CLASS_NUM,SAMPLE_NUM_PER_CLASS,FEATURE_DIM,62,62)   # TODO
                    sample_features = torch.sum(sample_features,1).squeeze(1)                                  #
                    test_features = encoder(Variable(test_images).to(device)) # 20x64                          #

                    # Compute Relation
                    sample_features_ext = sample_features.unsqueeze(0).repeat(batch_size,1,1,1,1)                         #
                    test_features_ext = test_features.unsqueeze(0).repeat(1*CLASS_NUM,1,1,1,1)                            #
                    test_features_ext = torch.transpose(test_features_ext,0,1)                                            # TODO
                    relation_pairs = torch.cat((sample_features_ext,test_features_ext),2).view(-1,FEATURE_DIM*2,62,62)    #
                    relations = rn(relation_pairs).view(-1,CLASS_NUM)                                                     #

                    # Predict
                    _, predict_labels = torch.max(relations.data,1)

                    # Counting Correct Ones
                    rewards = [1 if predict_labels[j]==test_labels[j] else 0 for j in range(batch_size)]
                    total_rewards += np.sum(rewards)

                # Record accuracy
                accuracy = total_rewards/1.0/CLASS_NUM/15
                accuracies.append(accuracy)
        
            # Overall accuracy
            test_accuracy, _ = utils.mean_confidence_interval(accuracies)

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