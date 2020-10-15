import os

# Constant (Settings)
TCN_OUT_CHANNEL = 64                        # Num of channels of output of TCN
RELATION_DIM = 32                           # Dim of one layer of relation net
CLASS_NUM = 3                               # <X>-way  | Num of classes
SAMPLE_NUM = 5                              # <Y>-shot | Num of supports per class
QUERY_NUM = 5                               # Num of instances for query per class
TRAIN_EPISODE = 30000                       # Num of training episode
TRAIN_FREQUENCY = 3                         # One data loading every <X> episode
VALIDATION_EPISODE = 50                     # Num of validation episode
VALIDATION_FREQUENCY = 200                  # After each <X> training episodes, do validation once
LEARNING_RATE = 0.0001                      # Initial learning rate
FRAME_NUM = 10                              # Num of frames per clip
CLIP_NUM = 5                                # Num of clips per window
WINDOW_NUM = 3                              # Num of processing window per video
INST_NUM = 15                               # Num of videos selected in each class
GPU = "4,5,6,7"                             # Index of GPU to be used
EXP_NAME = "CTC_Full_HAA_Attention_v2_5S"                # Name of this experiment

# Dataset
##################################################################################################################
DATASET = "haa"             # "kinetics" or "haa"

TRAIN_SPLIT = "/data/ssongad/haa/train.txt"                
VAL_SPLIT = "/data/ssongad/haa/train.txt"
TEST_SPLIT = "/data/ssongad/haa/test.txt"                    

KINETICS_DATA_FOLDERS = ["/data/ssongad/kinetics400/frame/train",
                         "/data/ssongad/kinetics400/frame/test"]

HAA_DATA_FOLDERS = ["/data/jchungaa/haa/frame/train",        
                    "/data/jchungaa/haa/frame/test",      
                    "/data/jchungaa/haa/frame/val"]      
##################################################################################################################

# Saved Models & Optimizers & Schedulers
##################################################################################################################
MAX_ACCURACY = 0.6173333333333334            # Accuracy of the loaded model
                                             # Leave 0 if N/A

CHECKPOINT = "/data/ssongad/codes/ctc_ap_v2/models/CTC_Full_HAA_Attention_v2_5S/Latest"             # Path of a folder, if you put everything in this folder with their DEFAULT names
                            # If you have such a path, paths below are not necessary then
                            # Leave a blank string if N/A

ENCODER_MODEL = ""          # 
RN_MODEL = ""               # Path of saved models
TCN_MODEL = ""              # Leave a blank string if N/A            
RN0_MODEL = ""              # 
AP_MODEL = ""               # 

ENCODER_OPTIM = ""          # 
RN_OPTIM = ""               # Path of saved optimizers                                      
TCN_OPTIM = ""              # Leave a blank string if N/A
RN0_OPTIM = ""              # 
AP_OPTIM = ""               # 

ENCODER_SCHEDULER = ""      # 
RN_SCHEDULER = ""           # Path of saved schedulers
TCN_SCHEDULER = ""          # Leave a blank string if N/A
RN0_SCHEDULER = ""          # 
AP_SCHEDULER = ""          # 


# If CHECKPOINT is given, then use files under CHECKPOINT first
# Only use the specific path of a file when it's missing under CHECKPOINT 
if os.path.exists(CHECKPOINT):
    results = []
    default_names = ("encoder.pkl", "rn.pkl", "tcn.pkl", "rn0.pkl", "encoder_optim.pkl", "rn_optim.pkl", "tcn_optim.pkl",
                     "rn0_optim.pkl", "encoder_scheduler.pkl", "rn_scheduler.pkl", "tcn_scheduler.pkl", "rn0_scheduler.pkl",
                     "ap.pkl", "ap_optim.pkl", "ap_scheduler.pkl")
    for default_name in default_names:
        tmp = os.path.join(CHECKPOINT, default_name)
        results.append(tmp if os.path.exists(tmp) else "")
    
    ENCODER_MODEL = results[0] if results[0] != "" else ENCODER_MODEL
    RN_MODEL = results[1] if results[1] != "" else RN_MODEL
    TCN_MODEL = results[2] if results[2] != "" else TCN_MODEL
    RN0_MODEL = results[3] if results[3] != "" else RN0_MODEL
    AP_MODEL = results[12] if results[12] != "" else AP_MODEL

    ENCODER_OPTIM = results[4] if results[4] != "" else ENCODER_OPTIM
    RN_OPTIM = results[5] if results[5] != "" else RN_OPTIM
    TCN_OPTIM = results[6] if results[6] != "" else TCN_OPTIM
    RN0_OPTIM = results[7] if results[7] != "" else RN0_OPTIM
    AP_OPTIM = results[13] if results[13] != "" else AP_OPTIM

    ENCODER_SCHEDULER = results[8] if results[8] != "" else ENCODER_SCHEDULER
    RN_SCHEDULER = results[9] if results[9] != "" else RN_SCHEDULER
    TCN_SCHEDULER = results[10] if results[10] != "" else TCN_SCHEDULER
    RN0_SCHEDULER = results[11] if results[11] != "" else RN0_SCHEDULER
    AP_SCHEDULER = results[14] if results[14] != "" else AP_SCHEDULER
##################################################################################################################


# Dataset Split
##################################################################################################################
def read_split(file_path):
    result = []
    if os.path.exists(file_path):
        file = open(file_path, "r")
        lines = file.readlines()
        file.close()

        for line in lines:
            result.append(line.rstrip())
            
    return result

TRAIN_SPLIT = read_split(TRAIN_SPLIT)
VAL_SPLIT = read_split(VAL_SPLIT)
TEST_SPLIT = read_split(TEST_SPLIT)

if DATASET == "haa":
    DATA_FOLDERS = HAA_DATA_FOLDERS
elif DATASET == "kinetics":
    DATA_FOLDERS = KINETICS_DATA_FOLDERS
##################################################################################################################
