"""An artificial neural network, trainable with multiple activation functions 
and varying number of hidden nodes.

The NN is built using numpy and trained using backpropigation and gradent 
decent. To setup the network chose the number of input, hidden nodes, and 
output. Next, chose the between implemented activation functions. Lastly, 
run the network using the train() function. To make any prediction 
afterward, use predict().  
"""

import numpy as np 
class NeuralNetwork(object): 
    def __init__(self, input_nodes: int, hidden_nodes: int, output_nodes: int, 
    activation_function: str, lr: float= 2e-5, w1=None, w2=None)-> None:
        """class constructur:

        w1: optional starting wieghts for the network, must be a numpy array 
        with the size (input_nodes, hidden_nodes).
        w2: optional starting wieghts for the network, must be a numpy array 
        with the size (hidden_nodes, output_nodes). 
        """
        self.w1 = self.init_wieght(input_nodes, hidden_nodes, w1)
        self.w2 = self.init_wieght(hidden_nodes, output_nodes, w2)
        self.activation_name = activation_function
        self.activation = self.get_active(activation_function)
        self.lr = lr


    #TODO: create checking mech
    def init_wieght(self, in_size, out_size, w=None):
        if w == None: 
            return np.random.normal(0.0, in_size**-0.5, (in_size,  out_size))
        return w


    def get_active(self, name: str= "relu"):
        sigmoid = lambda x : 1/(1+np.exp(-x)) 
        relu = lambda x : np.maximum(0, x)
        activation_dict = {"relu": relu, "sigmoid":sigmoid}
        return activation_dict[name]

    
    def train(self, x, y): 
        """dose a forward pass, then backward pass to update the wieghts
        
        x: is the inputs/featurs.
        y: is the label/target.
        """
        batch_size = y.shape[0]
        delta_w1 = np.zeros_like(self.w1)
        delta_w2 = np.zeros_like(self.w2)

        for i, label in zip(x, y): 
            output, hidden_output = self.forward_pass(i)
            delta_w1, delta_w2 = self.back_prop(output, hidden_output, i, 
                                 label, delta_w1, delta_w2)
        self.update_wieghts(delta_w1, delta_w2, batch_size)

    
    def forward_pass(self, x):
        hidden_output = np.dot(x, self.w1)
        hidden_output = self.activation(hidden_output)
        output = np.dot(hidden_output, self.w2)

        return output, hidden_output


    def back_prop(self, output, hidden_output, x, y, delta_w1, delta_w2): 
        error = y - output
        hidden_error = np.dot(self.w2, error)
        
        output_error_term = error
        hidden_error_term = self.get_hidden_error_term(hidden_output, 
                            hidden_error)
        
        delta_w1 +=  hidden_error_term * x[:, None]
        delta_w2 += output_error_term * hidden_output[:, None]

        return delta_w1, delta_w2


    def get_hidden_error_term(self, hidden_output, hidden_error):
        if self.activation_name == "relu":
            return self.derivative_relu(hidden_error)

        elif self.activation_name == "sigmoid": 
             return hidden_error * (hidden_output * (1-hidden_output))
        

    def update_wieghts(self, delta_w1, delta_w2, batch_size): 
        
        self.w1 += delta_w1*self.lr/batch_size
        self.w2 += delta_w2*self.lr/batch_size


    def predict(self, x): 
        out,_ = self.forward_pass(x)
        return out

    def derivative_relu(self, x):
        x[x<=0] = 0 
        x[x>0] = 1
        return x
