import torch
#DEVICE = torch.device('cpu')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Specify the GPU device you want to work with (optional)
#torch.cuda.set_per_process_memory_fraction(0.25, DEVICE)
#torch.cuda.set_max_memory_split(512, DEVICE)

#torch.cuda.reset_max_memory_allocated(0.5)
#torch.cuda.set_per_process_memory_growth(True)

MODEL_PARAMETERS = {
    'FedNCF': {
        'mf_dim': 12,
        'layers': [48, 24, 12, 6],#
        'reg_layers': [0, 0, 0, 0],
        'reg_mf': 0
    }, 
}

BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_NEGATIVES = 4
TOPK = 20
VALIDATION_STEPS = 10
NEG_DATA = 99