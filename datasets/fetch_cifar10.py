import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from os.path import isfile
from dlgrad.tensor import Tensor

class CIFAR10:
    def __init__(self) -> None:
        self.idx = 0
        data = []
        labels = []

        if not isfile('datasets/cifar10/cifar10_data.npy'):
            print("Loading the files")
            for batch_file in range(1, 6):
                filename = f'datasets/cifar10/data_batch_{batch_file}'
                with open(filename, 'rb') as f:
                    batch = pickle.load(f, encoding='bytes')
                    data.append(batch[b'data'])
                    labels.extend(batch[b'labels'])
            np.save('datasets/cifar10/cifar10_data.npy', data)
            np.save('datasets/cifar10/cifar10_labels.npy', labels)
        else:
            print("Loading the numpy arrays")
            data = np.load('datasets/cifar10/cifar10_data.npy')
            labels = np.load('datasets/cifar10/cifar10_labels.npy')

        data = np.concatenate(data).astype(np.float32)
        data = data.reshape(-1, 3, 32, 32)
        labels = np.array(labels)
        train_data, self.test_data, train_labels, self.test_labels = train_test_split(data, labels, test_size=0.2)
        self.train_data, self.val_data, self.train_labels, self.val_labels = train_test_split(train_data, train_labels, test_size=0.2)
        self.train_data = self.train_data / 255.0
      
        

    def reset_idx(self):
        self.idx = 0 
    def get_train_data(self):
        return self.train_data, self.train_labels 

    def get_batch_data(self, x_data: np.ndarray, y_data: np.ndarray, BS: int):
        x = Tensor(x_data[self.idx:self.idx+BS])
        y = Tensor(y_data[self.idx:self.idx+BS])
        self.idx += BS
        return x, y

    def num_train_steps(self, BS: int) -> int:
        return self.train_data.shape[0] // BS

    def delete_data(self):
        print("deleting")
        del self.train_data
        del self.val_data
        del self.train_labels
        del self.val_labels
    
    
