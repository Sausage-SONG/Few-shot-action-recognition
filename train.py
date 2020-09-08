import torch                                     #
import torch.nn as nn                            #  Pytorch
from torch.optim.lr_scheduler import StepLR      #
from torch.autograd import Variable              #

import numpy as np                               #  Numpy
import os                                        #  OS

from relationNet import RelationNetwork as RN       #  Relation Net
from relationNet import RelationNetworkZero as RN0  #
# from i3d import InceptionI3d as I3D               #  I3D
from encoder import Simple3DEncoder as C3D              #  Conv3D
from tcn import TemporalConvNet as TCN              #  TCN
import dataset                                      #  Dataset
from utils import *                                 #  Helper Functions
from config import *                                #  Config
from transformer import *

# Device to be used
os.environ['CUDA_VISIBLE_DEVICES'] = GPU            # GPU to be used
device = torch.device('cuda')                       #

def main():
    timestamp = time_tick("Start")
    write_log("Experiment Name: {}\n".format(EXP_NAME))

    # Define Models
    encoder = C3D(in_channels=3)
    encoder = nn.DataParallel(encoder)
    rn = RN(1, RELATION_DIM)
    rn0 = RN0(CLASS_NUM+1, RELATION_DIM)
    tcn = TCN(10240, [512,256,128,TCN_OUT_CHANNEL])
    tcn = nn.DataParallel(tcn)
    encoder_layer = TransformerEncoderLayer(TCN_OUT_CHANNEL,nhead=2)
    trans_encoder = TransformerEncoder(encoder_layer,1)
    decoder_layer = CustomTransformerDecoderLayer(TCN_OUT_CHANNEL,nhead=2)
    trans_decoder = TransformerDecoder(decoder_layer,1)
    # tcn = nn.DataParallel(tcn)
    ctc = nn.CTCLoss()
    # mse = nn.MSELoss()
    logSoftmax = nn.LogSoftmax(2)

    # Move models to computing device
    encoder.to(device)
    rn.to(device)
    rn0.to(device)
    tcn.to(device)
    trans_encoder.to(device)
    trans_decoder.to(device)
    ctc.to(device)
    logSoftmax.to(device)

    # Define Optimizers
    encoder_optim = torch.optim.AdamW(encoder.parameters(), lr=LEARNING_RATE) #weight_decay=0.0001
    rn_optim = torch.optim.AdamW(rn.parameters(), lr=LEARNING_RATE)
    tcn_optim = torch.optim.AdamW(tcn.parameters(), lr=LEARNING_RATE)
    rn0_optim = torch.optim.AdamW(rn0.parameters(), lr=LEARNING_RATE)
    trans_encoder_optim = torch.optim.AdamW(trans_encoder.parameters(),lr=LEARNING_RATE)
    trans_decoder_optim = torch.optim.AdamW(trans_decoder.parameters(),lr=LEARNING_RATE)

    # Define Schedulers
    encoder_scheduler = StepLR(encoder_optim, step_size=2000, gamma=0.5)
    rn_scheduler = StepLR(rn_optim, step_size=2000, gamma=0.5)
    tcn_scheduler = StepLR(tcn_optim, step_size=2000, gamma=0.5)
    rn0_scheduler = StepLR(rn0_optim, step_size=2000, gamma=0.5)
    trans_encoder_scheduler = StepLR(trans_encoder_optim, step_size=2000, gamma=0.5)
    trans_decoder_scheduler = StepLR(trans_decoder_optim, step_size=2000, gamma=0.5)

    log, timestamp = time_tick("Definition", timestamp)
    write_log(log)

    # Load Saved Models & Optimizers & Schedulers
    # if os.path.exists(ENCODER_MODEL):
    #     encoder.load_state_dict(torch.load(ENCODER_MODEL))
    # if os.path.exists(RN_MODEL):
    #     rn.load_state_dict(torch.load(RN_MODEL))
    # if os.path.exists(TCN_MODEL):
    #     tcn.load_state_dict(torch.load(TCN_MODEL))
    # if os.path.exists(RN0_MODEL):
    #     rn0.load_state_dict(torch.load(RN0_MODEL))
    # if os.path.exists(TRANS_EN):
    #     trans_encoder.load_state_dict(torch.load(TRANS_EN))
    # if os.path.exists(TRANS_DE):
    #     trans_decoder.load_state_dict(torch.load(TRANS_DE))
    
    # if os.path.exists(ENCODER_OPTIM):
    #     encoder_optim.load_state_dict(torch.load(ENCODER_OPTIM))
    # if os.path.exists(RN_OPTIM):
    #     rn_optim.load_state_dict(torch.load(RN_OPTIM))
    # if os.path.exists(TCN_OPTIM):
    #     tcn_optim.load_state_dict(torch.load(TCN_OPTIM))
    # if os.path.exists(RN0_OPTIM):
    #     rn0_optim.load_state_dict(torch.load(RN0_OPTIM))
    # if os.path.exists(TRANS_EN_OPTIM):
    #     trans_encoder_optim.load_state_dict(torch.load(TRANS_EN_OPTIM))
    # if os.path.exists(TRANS_DE_OPTIM):
    #     trans_decoder_optim.load_state_dict(torch.load(TRANS_DE_OPTIM))

    # if os.path.exists(ENCODER_SCHEDULER):
    #     encoder_scheduler.load_state_dict(torch.load(ENCODER_SCHEDULER))
    # if os.path.exists(RN_SCHEDULER):
    #     rn_scheduler.load_state_dict(torch.load(RN_SCHEDULER))
    # if os.path.exists(TCN_SCHEDULER):
    #     tcn_scheduler.load_state_dict(torch.load(TCN_SCHEDULER))
    # if os.path.exists(RN0_SCHEDULER):
    #     rn0_scheduler.load_state_dict(torch.load(RN0_SCHEDULER))
    # if os.path.exists(TRANS_EN_SCHEDULER):
    #     trans_encoder_scheduler.load_state_dict(torch.load(TRANS_EN_SCHEDULER))
    # if os.path.exists(TRANS_DE_SCHEDULER):
    #     trans_decoder_scheduler.load_state_dict(torch.load(TRANS_DE_SCHEDULER))
    
    max_accuracy = MAX_ACCURACY     # Currently the best accuracy
    accuracy_history = []           # Only for logging

    # Prepare output folder
    output_folder = os.path.join("./models", EXP_NAME)
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    
    log, timestamp = time_tick("Models Loading", timestamp)
    write_log(log)
    
    # Some Constant Tensors
    input_lengths = torch.full(size=(QUERY_NUM*CLASS_NUM,), fill_value=WINDOW_NUM, dtype=torch.long).to(device)
    target_lengths = torch.full(size=(QUERY_NUM*CLASS_NUM,), fill_value=1, dtype=torch.long).to(device)
    zeros = torch.zeros(QUERY_NUM*CLASS_NUM, WINDOW_NUM).to(device)

    # Training Loop
    episode = 0
    while episode < TRAIN_EPISODE:

        print("Train_Epi[{}] Pres_Accu = {}".format(episode, max_accuracy), end="\t")
        write_log("Training Episode {} | ".format(episode), end="")
        timestamp = time_tick("Restart")

        # Setup Data
        try:
            if DATASET == "haa":
                the_dataset = dataset.HAADataset(DATA_FOLDERS, None, "train", CLASS_NUM, SAMPLE_NUM, INST_NUM, FRAME_NUM, CLIP_NUM, WINDOW_NUM)
            elif DATASET == "kinetics":
                the_dataset = dataset.KineticsDataset(DATA_FOLDER, "train", (TRAIN_SPLIT, TEST_SPLIT), CLASS_NUM, SAMPLE_NUM, INST_NUM, FRAME_NUM, CLIP_NUM, WINDOW_NUM)
            sample_dataloader = dataset.get_data_loader(the_dataset,num_per_class=SAMPLE_NUM)
            batch_dataloader = dataset.get_data_loader(the_dataset,num_per_class=QUERY_NUM,shuffle=True)
            samples, _ = sample_dataloader.__iter__().next()             # [support*class, window*clip, RGB, frame, H, W]
            batches, batches_labels = batch_dataloader.__iter__().next()   # [query*class, window*clip, RGB, frame, H, W]
        except Exception:
            print("Skipped")
            write_log("Data Loading Error")
            continue
        
        log, timestamp = time_tick("Data Loading", timestamp)
        write_log("{} | ".format(log), end="")

        # Encoding
        samples = samples.view(CLASS_NUM*SAMPLE_NUM*WINDOW_NUM*CLIP_NUM, 3, FRAME_NUM, 128, 128)
        samples = encoder(Variable(samples).to(device))
        samples = samples.view(CLASS_NUM*SAMPLE_NUM, WINDOW_NUM*CLIP_NUM, -1)    # [support*class, window*clip, feature]

        batches = batches.view(CLASS_NUM*QUERY_NUM*WINDOW_NUM*CLIP_NUM, 3, FRAME_NUM, 128, 128)
        batches = encoder(Variable(batches).to(device))
        batches = batches.view(CLASS_NUM*QUERY_NUM, WINDOW_NUM*CLIP_NUM,-1)      # [query*class, window*clip, feature]

        log, timestamp = time_tick("Encoding", timestamp)
        write_log("{} | ".format(log), end="")

        # TCN Processing
        samples = torch.transpose(samples,1,2)       # [support*class, feature(channel), window*clip(length)]
        samples = tcn(samples)
        samples = torch.transpose(samples,1,2)       # [support*class, window*clip, feature]
        samples = samples.view(CLASS_NUM, SAMPLE_NUM, WINDOW_NUM, CLIP_NUM, -1)  # [class, sample, window, clip, feature]
        samples = torch.transpose(samples,0,2)        # [window, sample, class, clip, feature]
        samples = samples.reshape(WINDOW_NUM*SAMPLE_NUM,CLASS_NUM,-1)   # [window*sample(length), class(batch), clip*feature(embedding)]
        memory = trans_encoder(samples)   # transformer encoder takes (length, batch, embedding)

        batches = torch.transpose(batches,1,2)       # [query*class, feature(channel), window*clip(length)]
        batches = tcn(batches)
        batches = torch.transpose(batches,1,2)       # [query*class, window*clip, feature]
        batches = batches.reshape(CLASS_NUM*QUERY_NUM*WINDOW_NUM, 1, -1)  # [query*class*window, 1, clip*feature]
        batches_rn = batches.repeat(1,CLASS_NUM,1)      # [query*window*class, class, clip*feature]
        tgt = trans_decoder(batches_rn,memory)          # [query*window*class(length), class(batch), clip*feature(embedding)]
        log, timestamp = time_tick("TCN", timestamp)
        write_log("{} | ".format(log), end="")

        # Compute Relation
        samples_rn = tgt.reshape(QUERY_NUM*WINDOW_NUM*CLASS_NUM*CLASS_NUM, CLIP_NUM*TCN_OUT_CHANNEL).unsqueeze(1)         # [query*class*window*class, 1, clip*feature]
        batches_rn = batches_rn.reshape(QUERY_NUM*WINDOW_NUM*CLASS_NUM*CLASS_NUM, CLIP_NUM*TCN_OUT_CHANNEL).unsqueeze(1)  # [query*class*window*class, 1, clip*feature]
        relations = torch.cat((samples_rn,batches_rn),1).view(-1,2,CLIP_NUM*TCN_OUT_CHANNEL)    # [query*class*window*class, 2(channel), clip*feature]
        relations = rn(relations).view(QUERY_NUM*CLASS_NUM, WINDOW_NUM, CLASS_NUM)    # [query*class, window, class]

        # Compute Zero Probability
        samples_rn0 = tgt.reshape(QUERY_NUM*WINDOW_NUM*CLASS_NUM, CLASS_NUM, -1)      # [query*window*class, class, clip*feature]
        batches_rn0 = batches.reshape(CLASS_NUM*QUERY_NUM*WINDOW_NUM,-1).unsqueeze(1) # [query*window*class, 1, clip*feature]
        relations_rn0 = torch.cat((batches_rn0, samples_rn0), 1)                      # [query*class*window, (class+1), clip*feature]
        blank_prob = rn0(relations_rn0).view(QUERY_NUM*CLASS_NUM, WINDOW_NUM, 1)

        log, timestamp = time_tick("Relation", timestamp)
        write_log("{} | ".format(log), end="")

        # Generate final probabilities
        relations = torch.cat((blank_prob, relations), 2)
        final_outcome = torch.transpose(logSoftmax(relations), 0, 1)  # [window(length), query*class, class+1]

        # Compute Loss
        # batches_labels = batches_labels.unsqueeze(1).repeat(1,WINDOW_NUM).view(QUERY_NUM*CLASS_NUM*WINDOW_NUM)
        # one_hot_labels = Variable(torch.zeros(QUERY_NUM*CLASS_NUM*WINDOW_NUM, CLASS_NUM).scatter_(1, batches_labels.view(-1,1), 1).to(device))
        # loss = mse(relations,one_hot_labels)
        loss = ctc(final_outcome, batches_labels, input_lengths, target_lengths)
        print("Loss = {}".format(loss))

        # Back Propagation
        encoder.zero_grad()
        rn.zero_grad()
        rn0.zero_grad()
        tcn.zero_grad()
        trans_encoder.zero_grad()
        trans_decoder.zero_grad()
        loss.backward()

        # Clip Gradient
        nn.utils.clip_grad_norm_(encoder.parameters(),0.5)
        nn.utils.clip_grad_norm_(rn.parameters(),0.5)
        nn.utils.clip_grad_norm_(rn0.parameters(),0.5)
        nn.utils.clip_grad_norm_(tcn.parameters(),0.5)
        nn.utils.clip_grad_norm_(trans_decoder.parameters(),0.5)
        nn.utils.clip_grad_norm_(trans_encoder.parameters(),0.5)

        # Update Models
        encoder_optim.step()
        rn_optim.step()
        rn0_optim.step()
        tcn_optim.step()
        trans_encoder_optim.step()
        trans_decoder_optim.step()

        # Update "step" for scheduler
        rn_scheduler.step()
        rn0_scheduler.step()
        encoder_scheduler.step()
        tcn_scheduler.step()
        trans_encoder_scheduler.step()
        trans_decoder_scheduler.step()

        log, timestamp = time_tick("Loss & Step", timestamp)
        write_log("{} | ".format(log), end="")
        write_log("Loss = {}".format(loss))
        episode += 1

        # Validation Loop
        if (episode % VALIDATION_FREQUENCY == 0 and episode != 0) or episode == TRAIN_EPISODE:
            write_log("\n")
            with torch.no_grad():
                accuracies = []
                
                validation_episode = 0
                while validation_episode < VALIDATION_EPISODE:
                    print("Val_Epi[{}] Pres_Accu = {}".format(validation_episode, max_accuracy), end="\t")
                    write_log("Validating Episode {} | ".format(validation_episode), end="")

                    # Data Loading
                    try:
                        if DATASET == "haa":
                            the_dataset = dataset.HAADataset(DATA_FOLDERS, None, "train", CLASS_NUM, SAMPLE_NUM, INST_NUM, FRAME_NUM, CLIP_NUM, WINDOW_NUM)
                        elif DATASET == "kinetics":
                            the_dataset = dataset.KineticsDataset(DATA_FOLDER, "train", (TRAIN_SPLIT, TEST_SPLIT), CLASS_NUM, SAMPLE_NUM, INST_NUM, FRAME_NUM, CLIP_NUM, WINDOW_NUM)
                        sample_dataloader = dataset.get_data_loader(the_dataset,num_per_class=SAMPLE_NUM)
                        batch_dataloader = dataset.get_data_loader(the_dataset,num_per_class=QUERY_NUM,shuffle=True)
                        samples, _ = sample_dataloader.__iter__().next()            # [query*class, clip, RGB, frame, H, W]
                        batches, batches_labels = batch_dataloader.__iter__().next()   # [query*class, window*clip, RGB, frame, H, W]
                    except Exception:
                        print("Skipped")
                        write_log("Data Loading Error")
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
                    samples = torch.transpose(samples,1,2)       # [support*class, feature(channel), window*clip(length)]
                    samples = tcn(samples)
                    samples = torch.transpose(samples,1,2)       # [support*class, window*clip, feature]
                    samples = samples.view(CLASS_NUM, SAMPLE_NUM, WINDOW_NUM, CLIP_NUM, -1)  # [class, sample, window, clip, feature]
                    samples = torch.transpose(samples,0,2)        # [window, sample, class, clip, feature]
                    samples = samples.reshape(WINDOW_NUM*SAMPLE_NUM,CLASS_NUM*CLIP_NUM,-1)
                    memory = trans_encoder(samples)   # transformer encoder takes (length, batch, embedding)

                    batches = torch.transpose(batches,1,2)       # [query*class, feature(channel), window*clip(length)]
                    batches = tcn(batches)
                    batches = torch.transpose(batches,1,2)       # [query*class, window*clip, feature]
                    batches = batches.reshape(CLASS_NUM*QUERY_NUM*WINDOW_NUM, CLIP_NUM,-1)  # [query*class*window, clip, feature]
                    batches_rn = batches.repeat(1,CLASS_NUM,1)      # [query*window*class, class*clip, feature]
                    tgt = trans_decoder(batches_rn,memory)          # [query*window*class, class*clip, feature]
                    
                    # Compute Relation
                    samples_rn = tgt.reshape(QUERY_NUM*WINDOW_NUM*CLASS_NUM*CLASS_NUM, CLIP_NUM*TCN_OUT_CHANNEL).unsqueeze(1)         # [query*class*window*class, 1, clip*feature]
                    batches_rn = batches_rn.reshape(QUERY_NUM*WINDOW_NUM*CLASS_NUM*CLASS_NUM, CLIP_NUM*TCN_OUT_CHANNEL).unsqueeze(1)  # [query*class*window*class, 1, clip*feature]
                    relations = torch.cat((samples_rn,batches_rn),1).view(-1,2,CLIP_NUM*TCN_OUT_CHANNEL)    # [query*class*window*class, 2(channel), clip*feature]
                    relations = rn(relations).view(QUERY_NUM*CLASS_NUM, WINDOW_NUM, CLASS_NUM)    # [query*class, window, class]
                    
                    # Compute Zero Probability
                    samples_rn0 = tgt.reshape(QUERY_NUM*WINDOW_NUM*CLASS_NUM, CLASS_NUM, -1)      # [query*window*class, class, clip*feature]
                    batches_rn0 = batches.reshape(CLASS_NUM*QUERY_NUM*WINDOW_NUM,-1).unsqueeze(1) # [query*window*class, 1, clip*feature]
                    relations_rn0 = torch.cat((batches_rn0, samples_rn0), 1)                      # [query*class*window, (class+1), clip*feature]
                    blank_prob = rn0(relations_rn0).view(QUERY_NUM*CLASS_NUM, WINDOW_NUM, 1)

                    # Generate final probabilities
                    relations = torch.cat((blank_prob, relations), 2)
                    final_outcome = nn.functional.softmax(relations, 2)  # [query*class, window(length), class+1]

                    # Predict
                    predict_labels = ctc_predict(final_outcome.cpu().numpy())
                    batches_labels = batches_labels.numpy()

                    # Counting Correct Ones #TODO use ctc as score
                    rewards = [compute_score(prediction, truth) for prediction, truth in zip(predict_labels, batches_labels)]
                    # rewards = [1 if predict_labels[i] == batches_labels[i] else 0 for i in range(len(predict_labels))]
                    total_rewards += np.sum(rewards)

                    # Record accuracy
                    accuracy = total_rewards/total_num_covered
                    accuracies.append(accuracy)
                    print("Accu = {}".format(accuracy))
                    write_log("Accuracy = {}".format(accuracy))

                    validation_episode += 1

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
                    # Prepare folder
                    folder_for_this_accuracy = os.path.join(output_folder, str(val_accuracy))
                    max_accuracy = val_accuracy
                    print("Models Saved with accuracy={}".format(max_accuracy))
                    write_log("Models Saved")
                else:
                    folder_for_this_accuracy = os.path.join(output_folder, "Latest")
                    write_log("")

                if not os.path.exists(folder_for_this_accuracy):
                    os.mkdir(folder_for_this_accuracy)

                # Save networks
                torch.save(encoder.state_dict(), os.path.join(folder_for_this_accuracy, "encoder.pkl"))
                torch.save(rn.state_dict(), os.path.join(folder_for_this_accuracy, "rn.pkl"))
                torch.save(tcn.state_dict(), os.path.join(folder_for_this_accuracy, "tcn.pkl"))
                torch.save(rn0.state_dict(), os.path.join(folder_for_this_accuracy, "rn0.pkl"))
                torch.save(trans_encoder.state_dict(), os.path.join(folder_for_this_accuracy, "trans_encoder.pkl"))
                torch.save(trans_decoder.state_dict(), os.path.join(folder_for_this_accuracy, "trans_decoder.pkl"))

                torch.save(encoder_optim.state_dict(), os.path.join(folder_for_this_accuracy, "encoder_optim.pkl"))
                torch.save(rn_optim.state_dict(), os.path.join(folder_for_this_accuracy, "rn_optim.pkl"))
                torch.save(tcn_optim.state_dict(), os.path.join(folder_for_this_accuracy, "tcn_optim.pkl"))
                torch.save(rn0_optim.state_dict(), os.path.join(folder_for_this_accuracy, "rn0_optim.pkl"))
                torch.save(trans_encoder_optim.state_dict(), os.path.join(folder_for_this_accuracy, "trans_encoder_optim.pkl"))
                torch.save(trans_decoder_optim.state_dict(), os.path.join(folder_for_this_accuracy, "trans_decoder_optim.pkl"))

                torch.save(encoder_scheduler.state_dict(), os.path.join(folder_for_this_accuracy, "encoder_scheduler.pkl"))
                torch.save(rn_scheduler.state_dict(), os.path.join(folder_for_this_accuracy, "rn_scheduler.pkl"))
                torch.save(tcn_scheduler.state_dict(), os.path.join(folder_for_this_accuracy, "tcn_scheduler.pkl"))
                torch.save(rn0_scheduler.state_dict(), os.path.join(folder_for_this_accuracy, "rn0_scheduler.pkl"))
                torch.save(trans_encoder_scheduler.state_dict(), os.path.join(folder_for_this_accuracy, "trans_encoder_scheduler.pkl"))
                torch.save(trans_decoder_scheduler.state_dict(), os.path.join(folder_for_this_accuracy, "trans_decoder_scheduler.pkl"))
                
                os.system("cp ./config.py '" + folder_for_this_accuracy + "'")
                os.system("cp ./log.txt '" + folder_for_this_accuracy + "'")

    print("Training Done")
    print("Final Accuracy = {}".format(max_accuracy))
    write_log("\nFinal Accuracy = {}".format(max_accuracy))

# Program Starts
if __name__ == '__main__':
    main()
