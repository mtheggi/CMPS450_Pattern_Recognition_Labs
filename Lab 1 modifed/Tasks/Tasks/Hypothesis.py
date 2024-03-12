import numpy as np

class HypothesisFunction:
    def __init__(self, l : int , m: int , k : int ):
        # TODO [1]: Initialize the function's unknown parameters
        self.l = l
        self.m = m
        self.k = k
            
        # TODO [2]: Initialize Wh and Wo matrices from a standard normal distribution
        self.Wh = np.random.randn(self.m , self.l) 
        self.Wo = np.random.randn(self.k , self.m)
        
        # TODO [3]: Initialize bo and bo column vectors as zero
        self.bo = np.zeros((self.k , 1 ))
        self.bh = np.zeros((self.m , 1 ))

    def forward(self, x : int ) -> tuple[np.ndarray, np.ndarray]:
        # Ensure input shape matches input size
        assert x.shape[0] == self.l, f"Your input must be consistent the value l={self.l}"
        
        # TODO [4]: Compute a as mentioned above
        a =  np.tanh(np.dot(self.Wh , x ) + self.bh)
        
        # TODO [5]: Compute output ignoring ReLU
        y = np.dot(self.Wo, a) + self.bo
        
        # TODO [6]: Apply ReLU on the output with numpy boolean masking
        y[y<0] = 0 
        return y, a

    def double_forward(self, x1, x2):
        # Ensure input shape matches input size
            y1, _ = self.forward(x1)
            y2, _ = self.forward(x2)
            
            z = np.concatenate((y1, y2))
            z_bar = (z - np.mean(z)) / np.std(z)
            
            return z_bar


    def count_params(self):
    
        num_params = lambda z: np.prod(z.shape)
        total_params = num_params(self.Wh) + num_params(self.Wo) + num_params(self.bh) + num_params(self.bo)
        
        return total_params