MIN_VALUE = 1.0
NUM_TARGET = 10
NUM_RESOURCE = 5
NUM_STEP = 5
NUM_ATTACK = 5
THRESHOLD = 1       # defender constraint threshold
GROUPS = 3          # defender constraint groups

BATCH_SIZE_EPISODE = 8
MEMORY_SIZE_EPISODE = 16

BATCH_SIZE_TRANSITION = 32
MEMORY_SIZE_TRANSITION = 64

NUM_FEATURE = 6

NUM_EPISODE = 1001

GAMMA = 0.9
TARGET_UPDATE_EPISODE = 2
TARGET_UPDATE_TRANSITION = 2
OPTIMIZE_UPDATE_EPISODE = 1

NUM_SAMPLE = 100

LSTM_HIDDEN_SIZE = 32
LR_TRANSITION = 0.005
LR_EPISODE = 0.001

ENTROPY_COEFF_TRANS = 1.0
ENTROPY_COEFF_EPS = 1.5
