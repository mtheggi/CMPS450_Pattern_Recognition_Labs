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
        self.μ = np.mean(x_train, axis=0) 
        self.σ = np.std(x_train, axis=0)
        # if a column has zero std (useless constant) set σ=1 (skip their standardization)
        self.σ = np.where(self.σ == 0, 1, self.σ)
        
        # TODO 2: Standardize the training data
        z_train = (x_train - self.μ)/self.σ
        # I should standardize the data because pca depends on variances of the features 
        # and if the features are not standardized, the feature with the largest variance will dominate the first principal component.      
        # TODO 3: Compute the covariance matrix
        Σ = np.cov(z_train, rowvar=False, bias=False)
        
        # TODO 4: Compute eigenvalues and eigenvectors using Numpy
        λs, U = np.linalg.eig(Σ)
        λs, U = λs.real, U.real           # sometimes a zero imaginary part can appear due to approximations
        print("shape of U " , U.shape )
        # TODO 5: Sort eigenvalues and eigenvectors
        # TODO 5.1: Find the sequence of indices that sort λs in descending order
        print("test", λs)
        
        sorting_inds = np.argsort(λs)[::-1] # to reverse it 
        # TODO 5.2: Use it to sort λs and U
        λs = λs[sorting_inds]
        U = U[:, sorting_inds]
        print("shape of U " , U.shape )
        # TODO 6: Select the top L eigenvectors and set A accordingly
        L = self.new_dim
        self.A = U[:, 0:L:1].T
        print(self.A.shape )
        return self
    
    # x_val is (m,n) matrix where each row is an n-dimensional vector of features
    def transform(self, x_val):
        z_val = (x_val - self.μ) / self.σ
        # TODO 7: Apply the transformation equation
        z_val = np.dot(z_val, self.A.T)
        return z_val
    
    def inverse_transform(self, z_val):
        # TODO 8: Apply the inverse transformation equation (including destandardization)
        x_val = np.dot(z_val, self.A) * self.σ + self.μ
        return x_val

    def fit_transform(self, x_train):
        return self.fit(x_train).transform(x_train)