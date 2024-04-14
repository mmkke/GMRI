import numpy as np

class Random_Model:
    def __init__(self):
        self.mean =None
        self.std = None
    
    def fit(self, y):
        self.mean = np.mean(y)
        self.std = np.std(y)

    def predict(self, X):
        if self.mean is None or self.std is None:
            raise ValueError('Model needs to be fit')
        

        samples = X.shape[0]
        return np.random.normal(self.mean, self.std, samples).reshape(-1,1)