import pickle

from CNN import CNN

if __name__ == '__main__':
    TRAINING_DATA_PATH = '../data/train_preprossed_sampled.pkl'

    with open(TRAINING_DATA_PATH, 'rb') as file:
        train = pickle.load(file)

    cnn = CNN()
    cnn.fit(train[:250])