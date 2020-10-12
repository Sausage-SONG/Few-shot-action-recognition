import torch                                     #
import torch.nn as nn                            #  Pytorch
from torch.optim.lr_scheduler import StepLR      #
from torch.autograd import Variable              #

import numpy as np                               #  Numpy
import os                                        #  OS

from relationNet import RelationNetwork as RN       #  Relation Net
from relationNet import RelationNetworkZero as RN0  #
# from i3d import InceptionI3d as I3D               #  I3D
from encoder import Simple3DEncoder as C3D          #  Conv3D
from tcn import TemporalConvNet as TCN              #  TCN
from kmeans_pytorch import kmeans                   #  KNN
from attention_pool import KNNCutDimentsion as KNNC # KNN Cut Dimension

import dataset                                      #  Dataset
from utils import *                                 #  Helper Functions
from config import *                                #  Config

# Device to be used
os.environ['CUDA_VISIBLE_DEVICES'] = GPU            # GPU to be used
device = torch.device('cuda')                       #

def main():
    timestamp = time_tick("Start")
    write_log("Experiment Name: {}\n".format(EXP_NAME))

    # Define Models
    encoder = C3D(in_channels=3) 
    tcn = TCN(245760, [128,128,64,TCN_OUT_CHANNEL])
    knnc = KNNC(CLIP_NUM, TCN_OUT_CHANNEL, KNN_IN_DIM) 
    # rn = RN(CLIP_NUM, RELATION_DIM)
    # rn0 = RN0(CLIP_NUM*(CLASS_NUM+1), RELATION_DIM)
    
    encoder = nn.DataParallel(encoder)
    # tcn = nn.DataParallel(tcn)
    # rn = nn.DataParallel(rn)
    # rn0 = nn.DataParallel(rn0)

    ctc = nn.CTCLoss(reduction='none')
    logSoftmax = nn.LogSoftmax(2)
    cos = nn.CosineSimilarity(dim=1)
    mse = nn.MSELoss()   

    # Move models to computing device
    encoder.to(device)
    tcn.to(device)
    knnc.to(device)
    # rn.to(device)
    # rn0.to(device)

    ctc.to(device)
    mse.to(device)
    logSoftmax.to(device)
    cos.to(device)

    # Define Optimizers
    encoder_optim = torch.optim.AdamW(encoder.parameters(), lr=LEARNING_RATE)
    # rn_optim = torch.optim.AdamW(rn.parameters(), lr=LEARNING_RATE)
    tcn_optim = torch.optim.AdamW(tcn.parameters(), lr=LEARNING_RATE)
    knnc_optim = torch.optim.AdamW(knnc.parameters(), lr=LEARNING_RATE)
    # rn0_optim = torch.optim.AdamW(rn0.parameters(), lr=LEARNING_RATE)

    # Define Schedulers
    encoder_scheduler = StepLR(encoder_optim, step_size=2000, gamma=0.5)
    # rn_scheduler = StepLR(rn_optim, step_size=2000, gamma=0.5)
    tcn_scheduler = StepLR(tcn_optim, step_size=2000, gamma=0.5)
    knnc_scheduler = StepLR(knnc_optim, step_size=2000, gamma=0.5)
    # rn0_scheduler = StepLR(rn0_optim, step_size=2000, gamma=0.5)

    log, timestamp = time_tick("Definition", timestamp)
    write_log(log)

    # Load Saved Models & Optimizers & Schedulers
    if os.path.exists(ENCODER_MODEL):
        encoder.load_state_dict(torch.load(ENCODER_MODEL))
        write_log("ENCODER Loaded")
    if os.path.exists(TCN_MODEL):
        tcn.load_state_dict(torch.load(TCN_MODEL))
        write_log("TCN Loaded")
    if os.path.exists(KNNC_MODEL):
        knnc.load_state_dict(torch.load(KNNC_MODEL))
        write_log("KNNC Loaded")
    # if os.path.exists(RN_MODEL):
    #     rn.load_state_dict(torch.load(RN_MODEL))
    #     write_log("RN Loaded")
    # if os.path.exists(RN0_MODEL):
    #     rn0.load_state_dict(torch.load(RN0_MODEL))
    #     write_log("RN0 Loaded")
    
    if os.path.exists(ENCODER_OPTIM):
        encoder_optim.load_state_dict(torch.load(ENCODER_OPTIM))
    # if os.path.exists(RN_OPTIM):
    #     rn_optim.load_state_dict(torch.load(RN_OPTIM))
    if os.path.exists(TCN_OPTIM):
        tcn_optim.load_state_dict(torch.load(TCN_OPTIM))
    if os.path.exists(KNNC_OPTIM):
        knnc_optim.load_state_dict(torch.load(KNNC_OPTIM))
    # if os.path.exists(RN0_OPTIM):
    #     rn0_optim.load_state_dict(torch.load(RN0_OPTIM))
    
    if os.path.exists(ENCODER_SCHEDULER):
        encoder_scheduler.load_state_dict(torch.load(ENCODER_SCHEDULER))
    # if os.path.exists(RN_SCHEDULER):
    #     rn_scheduler.load_state_dict(torch.load(RN_SCHEDULER))
    if os.path.exists(TCN_SCHEDULER):
        tcn_scheduler.load_state_dict(torch.load(TCN_SCHEDULER))
    if os.path.exists(KNNC_SCHEDULER):
        knnc_scheduler.load_state_dict(torch.load(KNNC_SCHEDULER))
    # if os.path.exists(RN0_SCHEDULER):
    #     rn0_scheduler.load_state_dict(torch.load(RN0_SCHEDULER))

    max_accuracy = MAX_ACCURACY     # Currently the best accuracy
    accuracy_history = []           # Only for logging

    # Prepare output folder
    output_folder = os.path.join("./models", EXP_NAME)
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    
    log, timestamp = time_tick("Models Loading", timestamp)
    write_log(log)
    
    # Some Constant Tensors
    input_lengths = torch.full(size=(QUERY_NUM*CLASS_NUM*CLASS_NUM*SAMPLE_NUM,), fill_value=WINDOW_NUM, dtype=torch.long).to(device)
    # zeros = torch.zeros(QUERY_NUM*CLASS_NUM, WINDOW_NUM).to(device)

    skipped = 0

    # Training Loop
    episode = 0
    while episode < TRAIN_EPISODE:

        print("Train_Epi[{}|{}] Pres_Accu = {:.5}".format(episode, skipped, max_accuracy), end="\t")
        write_log("Training Episode {} | ".format(episode), end="")
        timestamp = time_tick("Restart")

        # Training Mode
        encoder.train()
        tcn.train()
        knnc.train()

        # Zero Grad
        encoder.zero_grad()
        tcn.zero_grad()
        knnc.zero_grad()
        # rn.zero_grad()
        # rn0.zero_grad()

        # Setup Data
        try:
            if DATASET == "haa":
                the_dataset = dataset.HAADataset(DATA_FOLDERS, None, "train", CLASS_NUM, SAMPLE_NUM, INST_NUM, FRAME_NUM, CLIP_NUM, WINDOW_NUM)
            elif DATASET == "kinetics":
                the_dataset = dataset.KineticsDataset(KINETICS_DATA_FOLDERS, "train", (TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT), CLASS_NUM, SAMPLE_NUM, INST_NUM, FRAME_NUM, CLIP_NUM, WINDOW_NUM)
            sample_dataloader = dataset.get_data_loader(the_dataset, True, num_per_class=SAMPLE_NUM, num_workers=0)
            batch_dataloader = dataset.get_data_loader(the_dataset, False, num_per_class=QUERY_NUM,shuffle=True, num_workers=0)
            samples, _ = sample_dataloader.__iter__().next()             # [support*class, window*clip, RGB, frame, H, W]
            batches, batches_labels = batch_dataloader.__iter__().next()   # [query*class, window*clip, RGB, frame, H, W]
        except Exception:
            skipped += 1
            print("Skipped")
            write_log("Data Loading Error | Total Error = {}".format(skipped))
            continue
        
        log, timestamp = time_tick("Data Loading", timestamp)
        write_log("{} | ".format(log), end="")

        # Encoding
        samples = samples.view(CLASS_NUM*SAMPLE_NUM*WINDOW_NUM*CLIP_NUM, 3, FRAME_NUM, 128, 128)
        samples = encoder(Variable(samples).to(device))
        samples = samples.view(CLASS_NUM*SAMPLE_NUM, WINDOW_NUM*CLIP_NUM, -1)    # [support*class, window*clip, feature]

        batches = batches.view(CLASS_NUM*QUERY_NUM*WINDOW_NUM*CLIP_NUM, 3, FRAME_NUM, 128, 128)
        batches = encoder(Variable(batches).to(device))
        batches = batches.view(CLASS_NUM*QUERY_NUM, WINDOW_NUM*CLIP_NUM,-1)       # [query*class, window*clip, feature]

        log, timestamp = time_tick("Encoding", timestamp)
        write_log("{} | ".format(log), end="")

        # TCN Processing
        samples = torch.transpose(samples,1,2)       # [support*class, feature(channel), window*clip(length)]
        samples = tcn(samples)
        samples = torch.transpose(samples,1,2)       # [support*class, window*clip, feature(TCN)]
        samples = samples.reshape(-1, CLIP_NUM, TCN_OUT_CHANNEL)  # [class*sample*window, clip, feature(TCN)]

        batches = torch.transpose(batches,1,2)       # [query*class, feature(channel), window*clip(length)]
        batches = tcn(batches)
        batches = torch.transpose(batches,1,2)       # [query*class, window*clip, feature(TCN)]
        batches = batches.reshape(-1, CLIP_NUM, TCN_OUT_CHANNEL)  # [query*class*window, clip, feature(TCN)]

        batches = batches.detach()

        log, timestamp = time_tick("TCN", timestamp)
        write_log("{} | ".format(log), end="")

        # KNN
        samples = knnc(samples)  # [class*sample*window, feature(KNN)]
        batches = knnc(batches)  # [query*class*window, feature(KNN)]

        cluster_ids, cluster_centers = kmeans(X=samples, num_clusters=CLUSTER_NUM, distance='cosine', device=device) # [cluster, clip*feature]
        if cluster_ids is None:
            print("KNN Skipped")
            write_log("KNN Error")
            continue
        cluster_ids = (cluster_ids+1).reshape(CLASS_NUM, SAMPLE_NUM, WINDOW_NUM) # [class, support, window]

        log, timestamp = time_tick("KNN", timestamp)
        write_log("{} | ".format(log), end="")

        # Compute Relation
        batches_rn = batches.unsqueeze(1).repeat(1,CLUSTER_NUM,1).reshape(-1, KNN_IN_DIM) # [query*class*window*cluster, feature]
        cluster_centers_rn = cluster_centers.unsqueeze(0).repeat(QUERY_NUM*CLASS_NUM*WINDOW_NUM, 1, 1).reshape(-1, KNN_IN_DIM).to(device) # [query*class*window*cluster, feature]
        relations = cos(batches_rn, cluster_centers_rn).reshape(-1, WINDOW_NUM, CLUSTER_NUM) # [query*class, window, cluster]

        # Compute Zero Probability
        blank_prob = torch.full((QUERY_NUM*CLASS_NUM, WINDOW_NUM, 1), 1e-08, layout=torch.strided, device=device, requires_grad=True) # [query*class, window, 1]

        # Generate final probabilities
        final_outcome = torch.cat((blank_prob, relations), 2)             # [query*class, window(length), cluster+1]
        final_outcome = final_outcome.unsqueeze(1).repeat(1, SAMPLE_NUM*CLASS_NUM, 1, 1).reshape(-1, WINDOW_NUM, CLUSTER_NUM+1) # [query*class*class*support, window(length), cluster+1]
        final_outcome = torch.transpose(logSoftmax(final_outcome), 0, 1)  # [window(length), query*class*class*support, cluster+1]

        # Generate Target Labels
        target = cluster_ids.unsqueeze(0).repeat(QUERY_NUM*CLASS_NUM, 1, 1, 1) # [query*class, class, support, window]
        # target = target[torch.arange(target.size(0)), batches_labels-1] # [query*class, support, window]
        target = target.reshape(-1, WINDOW_NUM) # [query*class*class*support, window]

        target_lengths, target_mask = ctc_length_mask(target)
        target = target.reshape(-1) # [query*class*class*support*window]
        target = target[target_mask.reshape(-1)] # [SUM(target_lengths)]

        log, timestamp = time_tick("Relation", timestamp)
        write_log("{} | ".format(log), end="")

        # Compute Loss
        # relations = relations.reshape(QUERY_NUM*CLASS_NUM*WINDOW_NUM, CLASS_NUM)
        # batches_labels = batches_labels.unsqueeze(1).repeat(1,WINDOW_NUM).view(QUERY_NUM*CLASS_NUM*WINDOW_NUM)
        # one_hot_labels = Variable(torch.zeros(QUERY_NUM*CLASS_NUM*WINDOW_NUM, CLASS_NUM).scatter_(1, batches_labels.view(-1,1), 1).to(device))
        # loss = mse(relations,one_hot_labels)
        
        ctc_loss = ctc(final_outcome, target, input_lengths, target_lengths).reshape(-1, CLASS_NUM, SAMPLE_NUM) # [query*class, class, support]
        ctc_loss = torch.mean(ctc_loss, 2) # [query*class, class]

        ctc_loss_correct = ctc_loss[torch.arange(ctc_loss.size(0)), batches_labels-1] # [query*class]
        ctc_loss_correct = torch.mean(ctc_loss_correct)
        ctc_loss_correct.backward(retain_graph=True)
        
        mse_one_hot = Variable(torch.zeros(QUERY_NUM*CLASS_NUM, CLASS_NUM).scatter_(1, (batches_labels-1).view(-1,1), 1).to(device)) # [query*class, class]
        ctc_prob = 1 - nn.functional.softmax(ctc_loss, 1) # [query*class, class]
        mse_loss = mse(ctc_prob, mse_one_hot)
        mse_loss.backward()
        
        
        print("CTC Loss = {:.4} ".format(ctc_loss_correct), end="")
        print("MSE Loss = {:.4} ".format(mse_loss))

        # for name, parms in encoder.named_parameters():	
        #     print('-->name:', name, '-->grad_requirs:',parms.requires_grad, '-->grad_value:', parms.grad)

        # Clip Gradient
        nn.utils.clip_grad_norm_(encoder.parameters(),0.5)
        nn.utils.clip_grad_norm_(tcn.parameters(),0.5)
        nn.utils.clip_grad_norm_(knnc.parameters(),0.5)
        # nn.utils.clip_grad_norm_(rn.parameters(),0.5)
        # nn.utils.clip_grad_norm_(rn0.parameters(),0.5)

        # Update Models
        encoder_optim.step()
        tcn_optim.step()
        knnc_optim.step()
        # rn_optim.step()
        # rn0_optim.step()

        # Update "step" for scheduler
        encoder_scheduler.step()
        tcn_scheduler.step()
        knnc_scheduler.step()
        # rn_scheduler.step()
        # rn0_scheduler.step()

        log, timestamp = time_tick("Loss & Step", timestamp)
        write_log("{} | ".format(log), end="")
        write_log("CTC Loss = {:.6} | ".format(ctc_loss_correct), end="")
        write_log("MSE Loss = {:.6} | ".format(mse_loss))
        episode += 1

        # Validation Loop
        if (episode % VALIDATION_FREQUENCY == 0 and episode != 0) or episode == TRAIN_EPISODE:
            write_log("\n")

            # Validation Mode
            encoder.eval()
            tcn.eval()
            knnc.eval()

            with torch.no_grad():
                accuracies = []

                validation_episode = 0
                while validation_episode < VALIDATION_EPISODE:

                    print("Val_Epi[{}] Pres_Accu = {}".format(validation_episode, max_accuracy), end="\t")
                    write_log("Validating Episode {} | ".format(validation_episode), end="")

                    # Data Loading
                    try:
                        if DATASET == "haa":
                            the_dataset = dataset.HAADataset(DATA_FOLDERS, None, "val", CLASS_NUM, SAMPLE_NUM, INST_NUM, FRAME_NUM, CLIP_NUM, WINDOW_NUM)
                        elif DATASET == "kinetics":
                            the_dataset = dataset.KineticsDataset(KINETICS_DATA_FOLDERS, "val", (TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT), CLASS_NUM, SAMPLE_NUM, INST_NUM, FRAME_NUM, CLIP_NUM, WINDOW_NUM)
                        sample_dataloader = dataset.get_data_loader(the_dataset, True, num_per_class=SAMPLE_NUM, num_workers=0)
                        batch_dataloader = dataset.get_data_loader(the_dataset, False, num_per_class=QUERY_NUM,shuffle=True, num_workers=0)
                        samples, _ = sample_dataloader.__iter__().next()            # [query*class, clip, RGB, frame, H, W]
                        batches, batches_labels = batch_dataloader.__iter__().next()   # [query*class, window*clip, RGB, frame, H, W]
                    except Exception:
                        print("Loading Skipped")
                        write_log("Data Loading Error")
                        continue
                    
                    total_num_covered = CLASS_NUM * QUERY_NUM

                    # Encoding
                    samples = samples.view(CLASS_NUM*SAMPLE_NUM*WINDOW_NUM*CLIP_NUM, 3, FRAME_NUM, 128, 128)
                    samples = encoder(Variable(samples).to(device))
                    samples = samples.view(CLASS_NUM*SAMPLE_NUM, WINDOW_NUM*CLIP_NUM, -1)    # [support*class, window*clip, feature]

                    batches = batches.view(CLASS_NUM*QUERY_NUM*WINDOW_NUM*CLIP_NUM, 3, FRAME_NUM, 128, 128)
                    batches = encoder(Variable(batches).to(device))
                    batches = batches.view(CLASS_NUM*QUERY_NUM, WINDOW_NUM*CLIP_NUM,-1)       # [query*class, window*clip, feature]

                    # TCN Processing
                    samples = torch.transpose(samples,1,2)       # [support*class, feature(channel), window*clip(length)]
                    samples = tcn(samples)
                    samples = torch.transpose(samples,1,2)       # [support*class, window*clip, feature(TCN)]
                    samples = samples.reshape(-1, CLIP_NUM, TCN_OUT_CHANNEL)  # [class*sample*window, clip, feature(TCN)]

                    batches = torch.transpose(batches,1,2)       # [query*class, feature(channel), window*clip(length)]
                    batches = tcn(batches)
                    batches = torch.transpose(batches,1,2)       # [query*class, window*clip, feature(TCN)]
                    batches = batches.reshape(-1, CLIP_NUM, TCN_OUT_CHANNEL)  # [query*class*window, clip, feature(TCN)]

                    # KNN
                    samples = knnc(samples)  # [class*sample*window, feature(KNN)]
                    batches = knnc(batches)  # [query*class*window, feature(KNN)]

                    cluster_ids, cluster_centers = kmeans(X=samples, num_clusters=CLUSTER_NUM, distance='cosine', device=device) # [cluster, clip*feature]
                    if cluster_ids is None:
                        print("KNN Skipped")
                        write_log("KNN Error")
                        continue
                    cluster_ids = (cluster_ids+1).reshape(CLASS_NUM, SAMPLE_NUM, WINDOW_NUM) # [class, support, window]

                    # Compute Relation
                    batches_rn = batches.unsqueeze(1).repeat(1,CLUSTER_NUM,1).reshape(-1, KNN_IN_DIM) # [query*class*window*cluster, feature]
                    cluster_centers_rn = cluster_centers.unsqueeze(0).repeat(QUERY_NUM*CLASS_NUM*WINDOW_NUM, 1, 1).reshape(-1, KNN_IN_DIM).to(device) # [query*class*window*cluster, feature]
                    relations = cos(batches_rn, cluster_centers_rn).reshape(-1, WINDOW_NUM, CLUSTER_NUM) # [query*class, window, cluster]

                    # Compute Zero Probability
                    blank_prob = torch.full((QUERY_NUM*CLASS_NUM, WINDOW_NUM, 1), 1e-08, layout=torch.strided, device=device, requires_grad=True) # [query*class, window, 1]

                    # Generate final probabilities
                    relations = torch.cat((blank_prob, relations), 2)    # [query*class, window, cluster+1]
                    final_outcome = nn.functional.softmax(relations, 2)  # [query*class, window, cluster+1]

                    # Generate Target Labels
                    target = cluster_ids.reshape(-1, WINDOW_NUM) # [class*support, window]
                    # file = open("test.log", "a")
                    # file.write("{}\n\n".format(str(target)))
                    # file.close()
                    target_lengths, target_mask = ctc_length_mask(target)
                    target = target.reshape(-1) # [class*support*window]
                    target = target[target_mask.reshape(-1)] # [SUM(target_lengths)]

                    # Predict
                    # relations = relations.reshape(QUERY_NUM*CLASS_NUM*WINDOW_NUM, CLASS_NUM)
                    # _, predict_labels = torch.max(relations.data,1)
                    # batches_labels = batches_labels.unsqueeze(1).repeat(1,WINDOW_NUM).view(QUERY_NUM*CLASS_NUM*WINDOW_NUM).to(device)
                    predict_labels = ctc_alignment_predict(final_outcome, target, target_lengths, SAMPLE_NUM)
                    batches_labels = batches_labels.numpy()
                    print(predict_labels)
                    print(batches_labels)

                    # Counting Correct Ones #TODO use ctc as score
                    # rewards = [compute_score(prediction, truth) for prediction, truth in zip(predict_labels, batches_labels)]
                    rewards = [1 if predict_labels[i] == batches_labels[i] else 0 for i in range(len(predict_labels))]
                    total_rewards = np.sum(rewards)

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
                # torch.save(rn.state_dict(), os.path.join(folder_for_this_accuracy, "rn.pkl"))
                torch.save(tcn.state_dict(), os.path.join(folder_for_this_accuracy, "tcn.pkl"))
                torch.save(knnc.state_dict(), os.path.join(folder_for_this_accuracy, "knnc.pkl"))
                # torch.save(rn0.state_dict(), os.path.join(folder_for_this_accuracy, "rn0.pkl"))

                torch.save(encoder_optim.state_dict(), os.path.join(folder_for_this_accuracy, "encoder_optim.pkl"))
                # torch.save(rn_optim.state_dict(), os.path.join(folder_for_this_accuracy, "rn_optim.pkl"))
                torch.save(tcn_optim.state_dict(), os.path.join(folder_for_this_accuracy, "tcn_optim.pkl"))
                torch.save(knnc_optim.state_dict(), os.path.join(folder_for_this_accuracy, "knnc_optim.pkl"))
                # torch.save(rn0_optim.state_dict(), os.path.join(folder_for_this_accuracy, "rn0_optim.pkl")

                torch.save(encoder_scheduler.state_dict(), os.path.join(folder_for_this_accuracy, "encoder_scheduler.pkl"))
                # torch.save(rn_scheduler.state_dict(), os.path.join(folder_for_this_accuracy, "rn_scheduler.pkl"))
                torch.save(tcn_scheduler.state_dict(), os.path.join(folder_for_this_accuracy, "tcn_scheduler.pkl"))
                torch.save(knnc_scheduler.state_dict(), os.path.join(folder_for_this_accuracy, "knnc_scheduler.pkl"))
                # torch.save(rn0_scheduler.state_dict(), os.path.join(folder_for_this_accuracy, "rn0_scheduler.pkl"))

                os.system("cp ./config.py '" + folder_for_this_accuracy + "'")
                os.system("cp ./log.txt '" + folder_for_this_accuracy + "'")
    
    print("Training Done")
    print("Final Accuracy = {}".format(max_accuracy))
    write_log("\nFinal Accuracy = {}".format(max_accuracy))

# Program Starts
if __name__ == '__main__':
    main()
