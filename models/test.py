import pickle

from CNN import CNN

if __name__ == '__main__':
    TEST_DATA_PATH = '../data/test_preprocessed.pkl'

    with open(TEST_DATA_PATH, 'rb') as file:
        test = pickle.load(file)

    cnn = CNN()
    cnn.score(test)
