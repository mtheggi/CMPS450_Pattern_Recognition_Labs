import numpy as np

class BayesClassifier():
    def __init__(self, mode:str='QDA'):
        # the class labels (will be set upon fit)
        self.labels = []
        # model learnable parameters will be set upon fit
        self.means = []
        self.covs = []
        self.priors = []
        self.weighted_cov = None        # in case of LDA only
        
        # TODO 1: Assert that mode is one of ['QDA', 'LDA', 'Naive'] and set it to self.mode
        # assert goes here
        self.mode = None

    def fit(self, x_train: np.ndarray, y_train: np.ndarray):
        # TODO 2: Extract the labels from y_train using Numpy
        self.labels = None
        # TODO 3: Compute the model parameters
        for label in self.labels:
            # TODO 3.1: Extract the data belonging to the current class
            x_train_given_class = None
            # TODO 3.2: Compute the class mean and add it to self.means
            class_mean = None
            self.means.append(class_mean)
            # TODO 3.3: Compute the class covariance using np.cov and add it to self.cov
            # Check the documentation of np.cov and what layout it assumes about the input
            # Set bias=False in case of QDA only and use a one-line if for that
            class_cov = None
            self.covs.append(class_cov)
            # TODO 3.4: Compute the class prior and add it to self.priors
            class_prior = None
            self.priors.append(class_prior)
        
        # TODO 4: Convert the model parameters from lists to numpy arrays
        self.means = None
        self.covs = None
        self.priors = None
        
        # TODO 5: In case of LDA, compute the weighted covariance matrix (self.weighted_cov)
        # In one line, sum the covariance matrices for each class weighted (multiplied) by their priors
        if self.mode == 'LDA':
            self.weighted_cov = None
            # reset self.covs to an empty list to save memory as we no longer need it
            self.covs = []
        # TODO 6: In case of Naive Bayes, diagonalize each of the covariance matrices to enforce independence
        elif self.mode == 'Naive':
            # diagonalize the covariance matrices in one line
            self.covs = None
        # In case of QDA, we maintain the general setup and just return
        else: 
             return

    # TODO 7: Define the normal distribution N(x, μ, Σ) probability density function
    @staticmethod
    def N(x:np.ndarray, μ:np.ndarray, Σ:np.ndarray)->float:
        scale = None                              # start by defining the denominator 
        prob = None
        return prob    
    
    # TODO 8: Given a single x, compute P(C|x) using Bayes rule for each class
    def predict_proba_x(self, x:np.ndarray)->np.ndarray:
        if self.mode != 'LDA':
            # TODO 8.1: Compute P(x|C)P(C) for each class in a numpy array in one line while using covariance matrices in self.covs
            prob_product = None
        else:
            # TODO 8.2: Compute P(x|C)P(C) for each class in a numpy array in one line while using the same covariance matrix self.weighted_cov
            prob_product = None
        return prob_product/np.sum(prob_product) 
    
    def predict_proba(self, x_val:np.ndarray)->np.ndarray:
        # given x_val of dimensions (m,n) apply predict_proba_x to each row (point x) to return array of probabilities (m, k)
        return np.apply_along_axis(self.predict_proba_x, axis=1, arr=x_val)
    
    
    def predict(self, x_val:np.ndarray)->np.ndarray:
        # TODO 9: Get the final predictions by computing argmax over the result from self.predict_proba
        y_pred_prob = None
        y_pred_inds = None
        # replace each prediction with label in self.labels
        y_pred = self.labels[y_pred_inds]
        return y_pred
    
    def score(self, x_val:np.ndarray, y_val:np.ndarray)->float:
        y_pred = self.predict(x_val)
        # TODO 10: compute accuracy in one line by comparing y_val and y_pred 
        acc = None
        return acc