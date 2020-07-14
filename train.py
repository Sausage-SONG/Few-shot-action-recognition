# Public Packages
import torch                                     #
import torch.nn as nn                            #  Pytorch
from torch.optim.lr_scheduler import StepLR      #
from torch.autograd import Variable              #

import numpy as np                               #  Numpy
<<<<<<< Updated upstream
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

=======
import os                                        #  
import sys                                       #  OS

# Local Packages
from relationNet import RelationNetwork as RN    #  Relation Network
from i3d import InceptionI3d as I3D              #  I3D
from tcn import TemporalConvNet as TCN           #  TCN
import dataset                                   #  Task Generator
import utils                                     #  Helper Functions

# Constant (Settings)
FEATURE_DIM = 32                        # Dim of input of relation net
RELATION_DIM = 16                       # Dim of one layer of relation net
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

EXP_NAME = ""                                                     # Name of this experiment
DATA_FOLDER = "/data/ssongad/haa/new_normalized_frame/"           # Data path
encoder_saved_model = ""                                          # Path of saved encoder model
rn_saved_model = ""                                               # Path of saved relation net model
tcn_saved_model = ""                                              # Path of saved tcn model

# Device to be used
os.environ['CUDA_VISIBLE_DEVICES']="3,7,2"                          # GPU to be used
device = torch.device('cuda')
>>>>>>> Stashed changes

def main():
    utils.print_stage("Program Starts")

    metatrain_folders,metatest_folders = tg.HAA_folders()

<<<<<<< Updated upstream
    # i3d == the I3D network
    encoder = cnn()
    # rn == the relation network
    rn = RN(FEATURE_DIM,RELATION_DIM)
=======
    # Define models
    encoder = I3D(in_channels=3)
    encoder = nn.DataParallel(encoder)
    rn = RN(FEATURE_DIM,RELATION_DIM)
    tcn = TCN(26624, [32,32,32,32])
    mse = nn.MSELoss()
>>>>>>> Stashed changes

    # Move models to GPU
    encoder.to(device)
    rn.to(device)
<<<<<<< Updated upstream
=======
    tcn.to(device)
    mse.to(device)
>>>>>>> Stashed changes

    # Define Optimizer
    encoder_optim = torch.optim.Adam(encoder.parameters(),lr=LEARNING_RATE)
    rn_optim = torch.optim.Adam(rn.parameters(),lr=LEARNING_RATE)
<<<<<<< Updated upstream

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
=======
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
    output_folder = "./models/" + EXP_NAME + "/"
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    # Training Loop
    for episode in range(TRAIN_EPISODE):

        print("Train_Epi[{}] Pres_Accu = {}".format(episode, max_accuracy), end="\t")
>>>>>>> Stashed changes

        # Update "step" for scheduler
        rn_scheduler.step(episode)
        encoder_scheduler.step(episode)
<<<<<<< Updated upstream

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
=======
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
        samples = tcn(samples)
        batches = tcn(batches)

        # Compute Relation
        samples = samples.unsqueeze(0).repeat(BATCH_NUM_PER_CLASS*CLASS_NUM,1,1,1)
        batches = batches.unsqueeze(0).repeat(CLASS_NUM,1,1,1)
        batches = torch.transpose(batches,0,1) 
        relations = torch.cat((samples,batches),2).view(-1,FEATURE_DIM*2,9,9)
        relations = rn(relations).view(-1,CLASS_NUM)
>>>>>>> Stashed changes

        # Compute Loss
        mse = nn.MSELoss().to(device)                                                                                                       #
        one_hot_labels = Variable(torch.zeros(BATCH_NUM_PER_CLASS*CLASS_NUM, CLASS_NUM).scatter_(1, batch_labels.view(-1,1), 1).to(device)) # TODO
        loss = mse(relations,one_hot_labels)                                                                                                #

        # Back Propagation
        encoder.zero_grad()
        rn.zero_grad()
<<<<<<< Updated upstream
=======
        tcn.zero_grad()
>>>>>>> Stashed changes
        loss.backward()

        # Clip Gradient
        nn.utils.clip_grad_norm(encoder.parameters(),0.5)
        nn.utils.clip_grad_norm(rn.parameters(),0.5)
<<<<<<< Updated upstream
=======
        nn.utils.clip_grad_norm(tcn.parameters(),0.5)
>>>>>>> Stashed changes

        # Update Models
        encoder_optim.step()
        rn_optim.step()
<<<<<<< Updated upstream

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
=======
        tcn_optim.step()

        # Validation Loop
        if (episode % VALIDATION_FREQUENCY == 0 and episode != 0) or episode == EPISODE-1:
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
                    samples = tcn(samples)
                    batches = tcn(batches)
>>>>>>> Stashed changes

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

<<<<<<< Updated upstream
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
=======
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
                    torch.save(lstm.state_dict(), output_folder + "tcn_"+str(val_accuracy)+".pkl")

                    max_accuracy = val_accuracy
                    print("Model Saved with accuracy={}".format(max_accuracy))
>>>>>>> Stashed changes
    
    print("final accuracy = {}".format(max_accuracy))

# Program Starts
if __name__ == '__main__':
    main()