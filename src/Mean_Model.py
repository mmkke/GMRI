import numpy as np
class Mean_Model:
    '''
    Model that just adds mean of Y for every prediction
    '''
    def  __init__(self):
        self.mean = None

    def fit(self, X, y):
        self.mean = np.mean(y)

    def predict(self, X):
        if self.mean is None:
            raise ValueError("Not fit")
    
        samples = X.shape[0]
        return np.full((samples,1), self.mean)