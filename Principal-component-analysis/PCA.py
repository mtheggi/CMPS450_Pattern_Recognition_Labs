import numpy as np

class PCA():
    def __init__(self,  new_dim:int) -> None:
        # hyperparameter representing the number of dimensions after reduction
        self.new_dim = new_dim
        # for standardization
        self.μ:np.ndarray
        self.σ:np.ndarray
        # for PCA
        self.A:np.ndarray 

    # x_train is (m,n) matrix where each row is an n-dimensional vector of features
    def fit(self, x_train):
        # TODO 1: Find μ and σ of each feature in x_train
        self.μ = None
        self.σ = None
        # if a column has zero std (useless constant) set σ=1 (skip their standardization)
        self.σ = np.where(self.σ == 0, 1, self.σ)
        
        # TODO 2: Standardize the training data
        z_train = None
                
        # TODO 3: Compute the covariance matrix
        Σ = None
        
        # TODO 4: Compute eigenvalues and eigenvectors using Numpy
        λs, U = None
        λs, U = λs.real, U.real           # sometimes a zero imaginary part can appear due to approximations
        
        # TODO 5: Sort eigenvalues and eigenvectors
        # TODO 5.1: Find the sequence of indices that sort λs in descending order
        sorting_inds = None
        # TODO 5.2: Use it to sort λs and U
        λs = None
        U = None
        
        # TODO 6: Select the top L eigenvectors and set A accordingly
        L = self.new_dim
        self.A = None
        
        return self
    
    # x_val is (m,n) matrix where each row is an n-dimensional vector of features
    def transform(self, x_val):
        z_val = (x_val - self.μ) / self.σ
        # TODO 7: Apply the transformation equation
        return None
    
    def inverse_transform(self, z_val):
        # TODO 8: Apply the inverse transformation equation (including destandardization)
        return None

    def fit_transform(self, x_train):
        return self.fit(x_train).transform(x_train)