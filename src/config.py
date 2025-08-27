from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent

RAW_DATA_DIR = Path(__file__).parent.parent / 'data' /'raw'
PROCESS_DATA_DIR = Path(__file__).parent.parent / 'data' /'processed'
LOGS_DIR = ROOT_DIR / 'logs'
MODELS_DIR = ROOT_DIR / 'models'

SEQ_LEN = 32
BATCH_SIZE = 128
EMBEDDING_DIM = 128
ENCODER_HIDDEN_SIZE=256

LEARNING_RATE = 0.001
EPOCHS = 30
ENCODER_LAYERS = 2

DECODER_HIDDEN_SIZE = ENCODER_HIDDEN_SIZE*2