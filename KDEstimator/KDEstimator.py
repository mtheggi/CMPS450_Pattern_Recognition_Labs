import numpy as np
from numpy.linalg import det, inv

class KDEstimator():
    def __init__(self, bump:str='Gauss', bandwidth:str='Silverman') -> None:
        # TODO 0: Set the hyperparameters from input
        self.bump = bump
        self.bandwidth = bandwidth
        
        # set during fit
        self.M, self.N = None, None         # number of observations and features in each
        self.x_train = None                 # store training data to use in inference
        self.h = None                       # the actual value of the bandwidth

        # validation
        assert bump in ['Gauss', 'Rect'], f"Only Gauss and Rect bumps are supported but you passed {bump} to KDEstimator"
        
        # TODO 1: Assert that bandwidth is either 'Silverman' or 'Scott' or an instance of int or float
        bandwidth_is_valid =  bandwidth in ['Silverman', 'Scott'] or isinstance(bandwidth, (int, float))
        assert bandwidth_is_valid, f"bandwidth must be either 'Silverman' or 'Scott' or a number but you passed {bandwidth} to KDEstimator"
        
        
    def fit(self, x_train):
        # TODO 2: Set M and N and x_train
        self.M, self.N = x_train.shape[0], x_train.shape[1] # m points with n features 
        self.x_train = x_train
        
        M, N = self.M, self.N               # to avoid corrupting eqns with self
        
        # TODO 3: compute avg_σ as defined earlier
        # to caclc h_optimum = 1/N * sum of H_opt_i where where I is from 1 to N
        # H_opt_i = σ_i *   [ 4/((N+2 )* M )] ^ (1/N+4)
        # σ_i = sqrt(E (i,i) ) where E is the covariance matrix  ; 
        Silver_const = np.power(4/((N+2)*M), 1/(N+4)) 
        Scott_const = np.power(1/M, 1/(N+4))
        cov_matrix = np.cov(x_train ,rowvar=False , bias=True)
        
        avg_σ = (1/N) * np.sum(np.sqrt(np.diag(cov_matrix))) 
        
        # TODO 4: Set self.h in case of both Silverman and Scott
        if self.bandwidth == 'Silverman':
            self.h = avg_σ * Silver_const
        elif self.bandwidth =='Scott':
            self.h = avg_σ * Scott_const
        else:                           
            # it must be an int or float in this case so we use it directly
            self.h = self.bandwidth

        return self
    # O(M*N)
    def g(self, x): # xvect from 1 to M 
        if self.bump == 'Gauss':
            N = self.N
            π = np.pi
            # TODO 5: Implement the Gaussian bump while using einsum for vectorization
            scale =np.sqrt(np.power(2 * π, N) ) 
            # x trapnpose * x; 
            # x(m*n) x(m*n) for i in range M : for j in range N : c[i][j] += x[i][j]* x[i][j]             
            N = np.einsum('ij,ij->i' , x , x )
            N = np.exp(-0.5 * N)

            return N/scale
        
        elif self.bump == 'Rect':
            # TODO 6: Implement the Rectangular bump                    
            N =np.all( np.logical_and(x>=-0.5, x<=0.5), axis=1) 
            print(N)
            return N 

    #O(M*N)
    def ϕ(self, x):
        h, N = self.h, self.N
        # TODO 7: Implement ϕ as defined earlier
        scale = 1/(h**N)
        return scale * self.g(x/h)
    #O()
    def P(self, x):
        scale = 1/(self.M)
        xₘ = self.x_train
        
        sumt = np.sum(self.ϕ(x-xₘ))
        # TODO 8: Implement P as defined earlier; remember no for loops allowed for this file
        return sumt*scale
    
    def transform(self, x_val):
        # TODO 9: Apply P to each row of (m,n) matrix x_val using np.apply_along_axis
        # if x_val is 1D apply P to x_val directly (same line of code)
        return self.P(x_val) if x_val.ndim ==1 else np.apply_along_axis(self.P, axis=1, arr=x_val)
    
    def fit_transform(self, x_data):
        # fit on x_data then transform
        return self.fit(x_data).transform(x_data)