import os

# Constant (Settings)
TCN_OUT_CHANNEL = 64                        # Num of channels of output of TCN
RELATION_DIM = 32                           # Dim of one layer of relation net
CLASS_NUM = 3                               # <X>-way  | Num of classes
SAMPLE_NUM = 5                              # <Y>-shot | Num of supports per class
QUERY_NUM = 3                               # Num of instances for query per class
TRAIN_EPISODE = 30000                       # Num of training episode 
VALIDATION_EPISODE = 70                     # Num of validation episode
VALIDATION_FREQUENCY = 200                  # After each <X> training episodes, do validation once
LEARNING_RATE = 0.0005                      # Initial learning rate
FRAME_NUM = 10                              # Num of frames per clip
CLIP_NUM = 5                                # Num of clips per window
WINDOW_NUM = 3                              # Num of processing window per video
INST_NUM = 10                               # Num of videos selected in each class
GPU = "4,5,8"                               # Index of GPU to be used
EXP_NAME = "CTC_Kinetics400_Trans"                # Name of this experiment

# Dataset
##################################################################################################################
DATASET = "kinetics"             # "kinetics" or "haa"  

DATA_FOLDER = "/data/ssongad/kinetics400/mot_normalized_frame/train"           # 
TRAIN_SPLIT = "/data/ssongad/kinetics400/mot_train.txt"                  # For Kinetics dataset only
TEST_SPLIT = "/data/ssongad/kinetics400/mot_test.txt"                    #

DATA_FOLDERS = ["/data/ssongad/haa/new_normalized_frame/",        # For HAA dataset only
                "/data/ssongad/haa/normalized_frame_scale2",      # Order => [original (1x), 2x, 3x]
                "/data/ssongad/haa/normalized_frame_scale3"]      # If a certain speed is missing, just put the path of 1x there
                                                                  # Don't leave an invalid path, including blank strings
##################################################################################################################

# Saved Models & Optimizers & Schedulers
##################################################################################################################
MAX_ACCURACY = 0            # Accuracy of the loaded model
                            # Leave 0 if N/A

CHECKPOINT = "/data/ssongad/codes/ctc/models/CTC_Kinetics400_AdamW_FN/0.8752834467120181"             # Path of a folder, if you put everything in this folder with their DEFAULT names
                            # If you have such a path, paths below are not necessary then
                            # Leave a blank string if N/A

ENCODER_MODEL = ""          # 
RN_MODEL = ""               # Path of saved models
TCN_MODEL = ""              # Leave a blank string if N/A            
RN0_MODEL = ""              # 
TRANS_EN = ""               #
TRANS_DE = ""               #

ENCODER_OPTIM = ""          # 
RN_OPTIM = ""               # Path of saved optimizers                                      
TCN_OPTIM = ""              # Leave a blank string if N/A
RN0_OPTIM = ""              #
TRANS_EN_OPTIM = ""         #
TRANS_DE_OPTIM = ""         #

ENCODER_SCHEDULER = ""      # 
RN_SCHEDULER = ""           # Path of saved schedulers
TCN_SCHEDULER = ""          # Leave a blank string if N/A
RN0_SCHEDULER = ""          #
TRANS_EN_SCHEDULER = ""     #
TRANS_DE_SCHEDULER = ""     # 


# If CHECKPOINT is given, then use files under CHECKPOINT first
# Only use the specific path of a file when it's missing under CHECKPOINT 
if os.path.exists(CHECKPOINT):
    results = []
    default_names = ("encoder.pkl", "rn.pkl", "tcn.pkl", "rn0.pkl", "encoder_optim.pkl", "rn_optim.pkl", "tcn_optim.pkl",
                     "rn0_optim.pkl", "encoder_scheduler.pkl", "rn_scheduler.pkl", "tcn_scheduler.pkl", "rn0_scheduler.pkl",
                     "trans_encoder.pkl", "trans_encoder_optim.pkl", "trans_encoder_scheduler.pkl", "trans_decoder.pkl",
                     "trans_decoder_optim.pkl", "trans_decoder_scheduler.pkl")
    for default_name in default_names:
        tmp = os.path.join(CHECKPOINT, default_name)
        results.append(tmp if os.path.exists(tmp) else "")
    
    ENCODER_MODEL = results[0] if results[0] != "" else ENCODER_MODEL
    RN_MODEL = results[1] if results[1] != "" else RN_MODEL
    TCN_MODEL = results[2] if results[2] != "" else TCN_MODEL
    RN0_MODEL = results[3] if results[3] != "" else RN0_MODEL
    TRANS_EN = results[12] if results[12] != "" else TRANS_EN
    TRANS_DE = results[15] if results[15] != "" else TRANS_DE

    ENCODER_OPTIM = results[4] if results[4] != "" else ENCODER_OPTIM
    RN_OPTIM = results[5] if results[5] != "" else RN_OPTIM
    TCN_OPTIM = results[6] if results[6] != "" else TCN_OPTIM
    RN0_OPTIM = results[7] if results[7] != "" else RN0_OPTIM
    TRANS_EN_OPTIM = results[13] if results[13] != "" else TRANS_EN_OPTIM
    TRANS_DE_OPTIM = results[16] if results[16] != "" else TRANS_DE_OPTIM

    ENCODER_SCHEDULER = results[8] if results[8] != "" else ENCODER_SCHEDULER
    RN_SCHEDULER = results[9] if results[9] != "" else RN_SCHEDULER
    TCN_SCHEDULER = results[10] if results[10] != "" else TCN_SCHEDULER
    RN0_SCHEDULER = results[11] if results[11] != "" else RN0_SCHEDULER
    TRANS_EN_SCHEDULER = results[14] if results[14] != "" else TRANS_EN_SCHEDULER
    TRANS_DE_SCHEDULER = results[17] if results[17] != "" else TRANS_DE_SCHEDULER
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
TEST_SPLIT = read_split(TEST_SPLIT)
##################################################################################################################
