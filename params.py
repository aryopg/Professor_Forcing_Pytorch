import torch

######################################################################
##################### LANGUAGE-RELATED PARAMS ########################
######################################################################

SOS_token = 0
EOS_token = 1

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)

######################################################################
################### NETWORK ARCHITECTURE PARAMS ######################
######################################################################

N_ITERS = 20
BATCH_SIZE = 32
MAX_LENGTH = 10
HIDDEN_SIZE = 256
LEARNING_RATE = 0.01

######################################################################
##################### HARDWARE-RELATED PARAMS ########################
######################################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
