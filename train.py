import torch                                     #
import torch.nn as nn                            #  Pytorch
from torch.optim.lr_scheduler import StepLR      #
from torch.autograd import Variable              #

import numpy as np                               #  Numpy
import os                                        #  OS

from relationNet import RelationNetwork as RN       #  Relation Net
from relationNet import RelationNetworkZero as RN0  #
# from i3d import InceptionI3d as I3D               #  I3D
from i3d import Simple3DEncoder as C3D              #  Conv3D
from tcn import TemporalConvNet as TCN              #  TCN
import dataset                                      #  Dataset
from utils import *                                 #  Helper Functions

# Constant (Settings)
TCN_OUT_CHANNEL = 64                    # Num of channels of output of TCN
RELATION_DIM = 32                       # Dim of one layer of relation net
CLASS_NUM = 3                           # <X>-way  | Num of classes
SAMPLE_NUM_PER_CLASS = 5                # <Y>-shot | Num of supports per class
BATCH_NUM_PER_CLASS = 3                 # Num of instances for validation per class
TRAIN_EPISODE = 20000                   # Num of training episode 
VALIDATION_EPISODE = 50                 # Num of validation episode
VALIDATION_FREQUENCY = 200              # After each <X> training episodes, do validation once
LEARNING_RATE = 0.0005                   # Initial learning rate
NUM_FRAME_PER_CLIP = 10                 # Num of frames per clip
NUM_CLIP = 5                            # Num of clips per window
NUM_WINDOW = 3                          # Num of processing window per video
NUM_INST = 10                           # Num of videos selected in each class

EXP_NAMES = ["TCN_Simple3DEnconder_01", "OCCUPY", "TCN_Simple3DEnconder_CTC", "TCN_Simple3DEnconder_MSE"]

EXP_NAME = "TCN_Simple3DEnconder_CTC"                             # Name of this experiment
DATA_FOLDERS = ["/data/ssongad/haa/new_normalized_frame/",        #
                "/data/ssongad/haa/normalized_frame_scale2",      # Data path => [original, 2x, 3x]
                "/data/ssongad/haa/normalized_frame_scale3"]      #

encoder_saved_model = "/data/ssongad/codes/ctc/models/TCN_Simple3DEnconder_MSE/TCN_Simple3DEnconder_01/encoder_0.7466666666666667.pkl"          # Path of saved encoder model
rn_saved_model = "/data/ssongad/codes/ctc/models/TCN_Simple3DEnconder_MSE/TCN_Simple3DEnconder_01/rn_0.7466666666666667.pkl"               # Path of saved relation net model
tcn_saved_model = "/data/ssongad/codes/ctc/models/TCN_Simple3DEnconder_MSE/TCN_Simple3DEnconder_01/tcn_0.7466666666666667.pkl"              # Path of saved tcn model
rn0_saved_model = ""              # Path of saved rn0 model
MAX_ACCURACY = 0                  # Accuracy of the loaded model

# Device to be used
os.environ['CUDA_VISIBLE_DEVICES']="2,3"                          # GPU to be used
device = torch.device('cuda')                                     #

def main():
    timestamp = time_tick("Start")
    write_log("Experiment Name: {}\n".format(EXP_NAME))

    # Define models
    encoder = C3D(in_channels=3)
    encoder = nn.DataParallel(encoder)
    rn = RN(NUM_CLIP, RELATION_DIM)
    rn0 = RN0(NUM_CLIP*(CLASS_NUM+1), RELATION_DIM)
    tcn = TCN(245760, [128,128,64,TCN_OUT_CHANNEL])
    ctc = nn.CTCLoss()
    # mse = nn.MSELoss()
    logSoftmax = nn.LogSoftmax(2)

    # Move models to GPU
    encoder.to(device)
    rn.to(device)
    rn0.to(device)
    tcn.to(device)
    ctc.to(device)
    logSoftmax.to(device)

    # Define Optimizer
    encoder_optim = torch.optim.Adam(encoder.parameters(),lr=LEARNING_RATE)
    rn_optim = torch.optim.Adam(rn.parameters(),lr=LEARNING_RATE)
    tcn_optim = torch.optim.Adam(tcn.parameters(),lr=LEARNING_RATE)
    rn0_optim = torch.optim.Adam(rn0.parameters(),lr=LEARNING_RATE)

    # Define Scheduler
    encoder_scheduler = StepLR(encoder_optim,step_size=2000,gamma=0.5)
    rn_scheduler = StepLR(rn_optim,step_size=2000,gamma=0.5)
    tcn_scheduler = StepLR(tcn_optim,step_size=2000,gamma=0.5)
    rn0_scheduler = StepLR(rn0_optim,step_size=2000,gamma=0.5)

    log, timestamp = time_tick("Definition", timestamp)
    write_log(log)

    # Load Saved Model
    if encoder_saved_model != "":
        encoder.load_state_dict(torch.load(encoder_saved_model))
    if rn_saved_model != "":
        rn.load_state_dict(torch.load(rn_saved_model))
    if tcn_saved_model != "":
        tcn.load_state_dict(torch.load(tcn_saved_model))
    if rn0_saved_model != "":
        rn0.load_state_dict(torch.load(rn0_saved_model))
    
    max_accuracy = MAX_ACCURACY     # Currently the best accuracy
    accuracy_history = []           # Only for logging

    # Prepare output folder
    if EXP_NAME == "":
        output_folder = "./models/"
    else:
        output_folder = "./models/" + EXP_NAME + "/"
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    
    log, timestamp = time_tick("Models Loading", timestamp)
    write_log(log)
    
    # Do it only once, so moved here
    input_lengths = torch.full(size=(BATCH_NUM_PER_CLASS*CLASS_NUM,), fill_value=NUM_WINDOW, dtype=torch.long).to(device)
    target_lengths = torch.full(size=(BATCH_NUM_PER_CLASS*CLASS_NUM,), fill_value=1, dtype=torch.long).to(device)
    zeros = torch.zeros(BATCH_NUM_PER_CLASS*CLASS_NUM, NUM_WINDOW).to(device)

    skipped = 0

    # Training Loop
    for episode in range(TRAIN_EPISODE):

        print("Train_Epi[{}|{}] Pres_Accu = {}".format(episode, skipped, max_accuracy), end="\t")
        write_log("Training Episode {} | ".format(episode), end="")
        timestamp = time_tick("Restart")

        # Setup Data
        haaDataset_support = dataset.HAADataset(DATA_FOLDERS, None, "train", "support", CLASS_NUM, SAMPLE_NUM_PER_CLASS, NUM_INST, NUM_FRAME_PER_CLIP, NUM_CLIP, NUM_WINDOW)
        haaDataset_query = dataset.HAADataset(DATA_FOLDERS, haaDataset_support.get_classes(), "train", "query", CLASS_NUM, SAMPLE_NUM_PER_CLASS, NUM_INST, NUM_FRAME_PER_CLIP, NUM_CLIP, NUM_WINDOW)
        sample_dataloader = dataset.get_HAA_data_loader(haaDataset_support,num_per_class=SAMPLE_NUM_PER_CLASS)
        batch_dataloader = dataset.get_HAA_data_loader(haaDataset_query,num_per_class=BATCH_NUM_PER_CLASS,shuffle=True)
        try:
            samples, _ = sample_dataloader.__iter__().next()             # [batch, clip, RGB, frame, H, W]
            batches, batch_labels = batch_dataloader.__iter__().next()   # [batch, window*clip, RGB, frame, H, W]
        except Exception:
            skipped += 1
            print("Skipped")
            write_log("Data Loading Error | Total Error = {}".format(skipped))
            continue
        
        log, timestamp = time_tick("Data Loading", timestamp)
        write_log("{} | ".format(log), end="")

        # Encoding
        samples = samples.view(CLASS_NUM*SAMPLE_NUM_PER_CLASS*NUM_CLIP, 3, NUM_FRAME_PER_CLIP, 128, 128)
        samples = encoder(Variable(samples).to(device))
        samples = samples.view(CLASS_NUM,SAMPLE_NUM_PER_CLASS,NUM_CLIP,-1)
        samples = torch.sum(samples,1).squeeze(1)                              # [class, clip, feature]

        batches = batches.view(CLASS_NUM*BATCH_NUM_PER_CLASS*NUM_WINDOW*NUM_CLIP, 3, NUM_FRAME_PER_CLIP, 128, 128)
        batches = encoder(Variable(batches).to(device))
        batches = batches.view(CLASS_NUM*BATCH_NUM_PER_CLASS*NUM_WINDOW,NUM_CLIP,-1) # [batch*window, clip, feature]

        log, timestamp = time_tick("Encoding", timestamp)
        write_log("{} | ".format(log), end="")

        # TCN Processing
        samples = torch.transpose(samples,1,2)       # [class, feature(channel), clip(length)]
        samples = tcn(samples)
        samples = torch.transpose(samples,1,2)       # [class, clip, feature]

        batches = torch.transpose(batches,1,2)       # [batch*window, feature(channel), clip(length)]
        batches = tcn(batches)
        batches = torch.transpose(batches,1,2)       # [batch*window, clip, feature]

        log, timestamp = time_tick("TCN", timestamp)
        write_log("{} | ".format(log), end="")

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

        log, timestamp = time_tick("Relation", timestamp)
        write_log("{} | ".format(log), end="")

        # Generate final probabilities
        relations = torch.cat((blank_prob, relations), 2)
        final_outcome = torch.transpose(logSoftmax(relations), 0, 1)  # [window(length), batch, class+1]

        # Compute Loss
        # batch_labels = batch_labels.unsqueeze(1).repeat(1,NUM_WINDOW).view(BATCH_NUM_PER_CLASS*CLASS_NUM*NUM_WINDOW)
        # one_hot_labels = Variable(torch.zeros(BATCH_NUM_PER_CLASS*CLASS_NUM*NUM_WINDOW, CLASS_NUM).scatter_(1, batch_labels.view(-1,1), 1).to(device))
        # loss = mse(relations,one_hot_labels)
        loss = ctc(final_outcome, batch_labels, input_lengths, target_lengths)
        print("Loss = {}".format(loss))

        # Back Propagation
        encoder.zero_grad()
        rn.zero_grad()
        rn0.zero_grad()
        tcn.zero_grad()
        loss.backward()

        # gradient = []
        # for _, p in rn.named_parameters():
        #     gradient.append(p.grad.norm().item())
        # write_log("{} | ".format(gradient), end="")

        # Clip Gradient
        nn.utils.clip_grad_norm_(encoder.parameters(),0.5)
        nn.utils.clip_grad_norm_(rn.parameters(),0.5)
        nn.utils.clip_grad_norm_(rn0.parameters(),0.5)
        nn.utils.clip_grad_norm_(tcn.parameters(),0.5)

        # Update Models
        encoder_optim.step()
        rn_optim.step()
        rn0_optim.step()
        tcn_optim.step()

        # Update "step" for scheduler
        rn_scheduler.step()
        rn0_scheduler.step()
        encoder_scheduler.step()
        tcn_scheduler.step()

        log, timestamp = time_tick("Loss & Step", timestamp)
        write_log("{} | ".format(log), end="")
        write_log("Loss = {}".format(loss))

        # Validation Loop
        if (episode % VALIDATION_FREQUENCY == 0 and episode != 0) or episode == TRAIN_EPISODE-1:
            write_log("\n")
            with torch.no_grad():
                accuracies = []

                for validation_episode in range(VALIDATION_EPISODE):
                    print("Val_Epi[{}] Pres_Accu = {}".format(validation_episode, max_accuracy), end="\t")
                    write_log("Validating Episode {} | ".format(validation_episode), end="")

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
                        write_log("Data Loading Error")
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
                    # predict_labels = predict_labels.cpu().numpy()
                    # batch_labels = batch_labels.unsqueeze(1).repeat(1,NUM_WINDOW).view(BATCH_NUM_PER_CLASS*CLASS_NUM*NUM_WINDOW)
                    batch_labels = batch_labels.numpy()

                    # Counting Correct Ones #TODO use ctc as score
                    rewards = [compute_score(prediction, truth) for prediction, truth in zip(predict_labels, batch_labels)]
                    # rewards = [1 if predict_labels[i] == batch_labels[i] else 0 for i in range(len(predict_labels))]
                    total_rewards += np.sum(rewards)

                    # Record accuracy
                    accuracy = total_rewards/total_num_covered
                    accuracies.append(accuracy)
                    print("Accu = {}".format(accuracy))
                    write_log("Accuracy = {}".format(accuracy))

                # Overall accuracy
                val_accuracy, _ = mean_confidence_interval(accuracies)
                accuracy_history.append(val_accuracy)
                print("Final Val_Accu = {}".format(val_accuracy))
                write_log("Validation Accuracy = {} | ".format(val_accuracy), end="")

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
                    torch.save(rn0.state_dict(), output_folder + "rn0_"+str(val_accuracy)+".pkl")

                    max_accuracy = val_accuracy
                    print("Models Saved with accuracy={}".format(max_accuracy))
                    write_log("Models Saved")
                else:
                    write_log("")
    
    print("Training Done")
    print("Final Accuracy = {}".format(max_accuracy))
    write_log("\nFinal Accuracy = {}".format(max_accuracy))

# Program Starts
if __name__ == '__main__':
    main()
