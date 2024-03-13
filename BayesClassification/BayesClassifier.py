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
        assert mode in ['QDA' , 'LDA' , 'Naive'] , "unknow mode , mode shoulde be one of the following ['QDA', 'LDA', 'Naive'] " 
        self.mode = mode

    def fit(self, x_train: np.ndarray, y_train: np.ndarray):
        # TODO 2: Extract the labels from y_train using Numpy
        self.labels = np.unique(y_train)
        # TODO 3: Compute the model parameters
        for label in self.labels:
            # TODO 3.1: Extract the data belonging to the current class
            x_train_given_class = x_train[y_train == label]
            # TODO 3.2: Compute the class mean and add it to self.means
            class_mean = np.mean(x_train_given_class, axis=0)
            self.means.append(class_mean)
            # TODO 3.3: Compute the class covariance using np.cov and add it to self.cov
            # Check the documentation of np.cov and what layout it assumes about the input
            # Set bias=False in case of QDA only and use a one-line if for that
            class_cov = np.cov(x_train_given_class, rowvar=False, bias=(self.mode != 'QDA'))
            self.covs.append(class_cov)
            # TODO 3.4: Compute the class prior and add it to self.priors
            class_prior = len(x_train_given_class)/len(x_train)
            self.priors.append(class_prior)
        
        # TODO 4: Convert the model parameters from lists to numpy arrays
        self.means = np.array(self.means)
        self.covs = np.array(self.covs)
        self.priors = np.array(self.priors)
        
        # TODO 5: In case of LDA, compute the weighted covariance matrix (self.weighted_cov)
        # In one line, sum the covariance matrices for each class weighted (multiplied) by their priors
        if self.mode == 'LDA':
            rows_size = self.covs[0].shape[0]
            self.weighted_cov= np.zeros((rows_size,rows_size))
            for i in range(len(self.priors)):
                self.weighted_cov += self.priors[i] * self.covs[i]

            # reset self.covs to an empty list to save memory as we no longer need it
            self.covs = []
        # TODO 6: In case of Naive Bayes, diagonalize each of the covariance matrices to enforce independence
        elif self.mode == 'Naive':
            self.covs = [np.diag(np.diag(cov)) for cov in self.covs]
        # In case of QDA, we maintain the general setup and just return
        else: 
             return

    # TODO 7: Define the normal distribution N(x, μ, Σ) probability density function
    @staticmethod
    def N(x:np.ndarray, μ:np.ndarray, Σ:np.ndarray)->float:
        det_cov = np.linalg.det(Σ)
        inv_cov = np.linalg.inv(Σ)
        x_minus_μ = x - μ
        x_minus_μ_transpose = x_minus_μ.T
        n = x.shape[0]
        denominator_power = n/2  
        scale = 1 / (np.sqrt(det_cov) * np.power(2 * np.pi , denominator_power))                                # start by defining the denominator 
        prob = np.exp(-0.5 * np.dot(np.dot(x_minus_μ, inv_cov), x_minus_μ_transpose)) * scale
        return prob    
    
    # TODO 8: Given a single x, compute P(C|x) using Bayes rule for each class
    def predict_proba_x(self, x:np.ndarray)->np.ndarray:
        if self.mode != 'LDA':
            # TODO 8.1: Compute P(x|C)P(C) for each class in a numpy array in one line while using covariance matrices in self.covs
            prob_product = np.array([(self.N(x,self.means[i] , self.covs[i])*self.priors[i]) for i in range(len(self.labels))])
        else:
            # TODO 8.2: Compute P(x|C)P(C) for each class in a numpy array in one line while using the same covariance matrix self.weighted_cov
            prob_product = np.array([(self.N(x,self.means[i] , self.weighted_cov)*self.priors[i]) for i in range(len(self.labels))])
        
        return prob_product/np.sum(prob_product) # Review ?? why we divide by the sum of the prob_product
    
    def predict_proba(self, x_val:np.ndarray)->np.ndarray:
        # given x_val of dimensions (m,n) apply predict_proba_x to each row (point x) to return array of probabilities (m, k)
        return np.apply_along_axis(self.predict_proba_x, axis=1, arr=x_val) # array of probabilites 
    
    
    def predict(self, x_val:np.ndarray)->np.ndarray:
        # TODO 9: Get the final predictions by computing argmax over the result from self.predict_proba
        y_pred_prob = self.predict_proba(x_val) # array m * k 
        y_pred_inds = np.apply_along_axis(np.argmax, axis=1, arr=y_pred_prob) 
        # replace each prediction with label in self.labels
        y_pred = self.labels[y_pred_inds]
        return y_pred
    
    def score(self, x_val:np.ndarray, y_val:np.ndarray)->float:
        y_pred = self.predict(x_val)
        # TODO 10: compute accuracy in one line by comparing y_val and y_pred 
        acc = np.mean(y_val == y_pred)
        return acc
    
    
    def __str__(self):
        return f"BayesClassifier(mode={self.mode}, labels={self.labels}, means={self.means}, covs={self.covs}, priors={self.priors}, weighted_cov={self.weighted_cov})"