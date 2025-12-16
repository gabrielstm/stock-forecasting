
WINDOW_SIZE = 10
TRAIN_LEN = 80
TEST_LEN = 20
DATASET_NAME = 'data/historico_b3_indicadores.csv'
EPOCHS = 50

def get_split_index(total_rows):
    return int(total_rows * (TRAIN_LEN / (TRAIN_LEN + TEST_LEN)))
