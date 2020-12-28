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
LEARNING_RATE = 0.0005                      # Initial learning rate
FRAME_NUM = 10                              # Num of frames per clip
CLIP_NUM = 5                                # Num of clips per window
WINDOW_NUM = 3                              # Num of processing window per video
INST_NUM = 15                               # Num of videos selected in each class
GPU = ""                                    # Index of GPU to be used
EXP_NAME = ""                               # Name of this experiment

# Dataset
##################################################################################################################
DATASET = ""             # "haa" or "mit" or "finegym"

TRAIN_SPLIT = ""                
VAL_SPLIT = ""
TEST_SPLIT = ""                    

HAA_DATA_FOLDERS = ["",
                    "",
                    ""]

MIT_DATA_FOLDERS = ["",
                    ""]

FINEGYM_DATA_FOLDER = ""
##################################################################################################################

# Saved Models & Optimizers & Schedulers
##################################################################################################################
MAX_ACCURACY = 0.0            # Accuracy of the loaded model
                                             # Leave 0 if N/A

CHECKPOINT = ""             # Path of a folder, if you put everything in this folder with their DEFAULT names
                            # If you have such a path, paths below are not necessary then
                            # Leave a blank string if N/A

ENCODER_MODEL = ""          # 
RN_MODEL = ""               # Path of saved models
TCN_MODEL = ""              # Leave a blank string if N/A            
AP_MODEL = ""               # 

ENCODER_OPTIM = ""          # 
RN_OPTIM = ""               # Path of saved optimizers                                      
TCN_OPTIM = ""              # Leave a blank string if N/A
AP_OPTIM = ""               # 

ENCODER_SCHEDULER = ""      # 
RN_SCHEDULER = ""           # Path of saved schedulers
TCN_SCHEDULER = ""          # Leave a blank string if N/A
AP_SCHEDULER = ""           # 


# If CHECKPOINT is given, then use files under CHECKPOINT first
# Only use the specific path of a file when it's missing under CHECKPOINT 
if os.path.exists(CHECKPOINT):
    results = []
    default_names = ("encoder.pkl", "rn.pkl", "tcn.pkl", "encoder_optim.pkl", "rn_optim.pkl", "tcn_optim.pkl",
                    "encoder_scheduler.pkl", "rn_scheduler.pkl", "tcn_scheduler.pkl", "ap.pkl", "ap_optim.pkl", 
                    "ap_scheduler.pkl")
    for default_name in default_names:
        tmp = os.path.join(CHECKPOINT, default_name)
        results.append(tmp if os.path.exists(tmp) else "")
    
    ENCODER_MODEL = results[0] if results[0] != "" else ENCODER_MODEL
    RN_MODEL = results[1] if results[1] != "" else RN_MODEL
    TCN_MODEL = results[2] if results[2] != "" else TCN_MODEL
    AP_MODEL = results[9] if results[9] != "" else AP_MODEL

    ENCODER_OPTIM = results[3] if results[3] != "" else ENCODER_OPTIM
    RN_OPTIM = results[4] if results[4] != "" else RN_OPTIM
    TCN_OPTIM = results[5] if results[5] != "" else TCN_OPTIM
    AP_OPTIM = results[10] if results[10] != "" else AP_OPTIM

    ENCODER_SCHEDULER = results[6] if results[6] != "" else ENCODER_SCHEDULER
    RN_SCHEDULER = results[7] if results[7] != "" else RN_SCHEDULER
    TCN_SCHEDULER = results[8] if results[8] != "" else TCN_SCHEDULER
    AP_SCHEDULER = results[11] if results[11] != "" else AP_SCHEDULER
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
elif DATASET == "finegym":
    DATA_FOLDERS = FINEGYM_DATA_FOLDER
elif DATASET == "mit":
    DATA_FOLDERS = MIT_DATA_FOLDERS
##################################################################################################################
INFO_DICT = ["", ""]
d = dict()
for path in INFO_DICT:
    file = open(path, 'r')
    lines = file.readlines()
    file.close()

    for line in lines:
        line = line.strip()
        line = line.split(" ")
        label = line[1]
        name = line[0]

        if label in d.keys():
            d[label].append(name)
        else:
            d[label] = [name]
INFO_DICT = d
