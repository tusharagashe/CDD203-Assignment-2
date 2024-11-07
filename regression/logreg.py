# importing dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class BaseRegressor():
    def __init__(self, num_feats, learning_rate=0.1, tol=0.001, max_iter=100, batch_size=12):
        """
        No need to modify
        """
        # Initializing parameters: This is needed for gradient descent. 
        self.W = np.random.randn(num_feats + 1).flatten()
        
        # Assigning Hyperparameters: You may need to adjust learning rate, batch size, etc for better accuracy 
        self.lr = learning_rate
        self.tol = tol
        self.max_iter = max_iter
        self.batch_size = batch_size
        
        # This will need to be adjusted as you add or remove features
        self.num_feats = num_feats
        
        # Defining list for storing Loss History: Makes it so you can visualize your loss descending, ascending, or not moving
        self.loss_history_train = []
        self.loss_history_val = []
        
    def calculate_gradient(self, X, y):
        # Kept empty as when you are inheriting, you overwrite this with LogisticRegression's calculate_gradient method
        pass
    
    def loss_function(self, y_true, y_pred):
        # Kept empty as when you are inheriting, you overwrite this with LogisticRegression's loss_function method
        pass
    
    def make_prediction(self, X):
        # Kept empty as when you are inheriting, you overwrite this with LogisticRegression's make_prediction method
        pass
    
    def train_model(self, X_train, y_train, X_val, y_val):
        """
        Model training once you've created make_prediction, loss_functino, and calculate_gradient. No need to modify
        """
        # Padding data with vector of ones for bias term
        # Important!!!! Remember workshop 2 Linear regression and Gradient Descent
        X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
        X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
        
        # Defining initial values for while loop
        prev_update_size = 1
        iteration = 1
        
        # Gradient descent
        while prev_update_size > self.tol and iteration < self.max_iter:
            
            # Shuffling the training data for each epoch of training
            shuffle_arr = np.concatenate([X_train, np.expand_dims(y_train, 1)], axis=1)
            
            # In place shuffle: Prevents your gradient from being bias to where data is being presented
            np.random.shuffle(shuffle_arr)
            X_train = shuffle_arr[:, :-1]
            y_train = shuffle_arr[:, -1].flatten()
            num_batches = int(X_train.shape[0]/self.batch_size) + 1
            X_batch = np.array_split(X_train, num_batches)
            y_batch = np.array_split(y_train, num_batches)
            
            # Generating list to save the param updates per batch
            update_size_epoch = []
            
            # Iterating through batches (full for loop is one epoch of training)
            for X_train, y_train in zip(X_batch, y_batch):
                # Making prediction on batch
                y_pred = self.make_prediction(X_train)
                
                # Calculating loss
                loss_train = self.loss_function(X_train, y_train)
                
                # Adding current loss to loss history record
                self.loss_history_train.append(loss_train)
                
                # Storing previous weights and bias
                prev_W = self.W
                # Calculating gradient of loss function with respect to each parameter
                grad = self.calculate_gradient(X_train, y_train)
                
                # Updating parameters
                new_W = prev_W - self.lr * grad 
                self.W = new_W
                
                # Saving step size
                update_size_epoch.append(np.abs(new_W - prev_W))
                
                # Validation pass
                loss_val = self.loss_function(X_val, y_val)
                self.loss_history_val.append(loss_val)
                
            # Defining step size as the average over the past epoch
            prev_update_size = np.mean(np.array(update_size_epoch))
            # Updating iteration number
            iteration += 1
    
    def plot_loss_history(self):
        """
        Plots the loss history after training is complete. No need to modify
        """
        loss_hist = self.loss_history_train
        loss_hist_val = self.loss_history_val
        assert len(loss_hist) > 0, "Need to run training before plotting loss history"
        fig, axs = plt.subplots(2, figsize=(8,8))
        fig.suptitle('Loss History')
        axs[0].plot(np.arange(len(loss_hist)), loss_hist)
        axs[0].set_title('Training Loss')
        axs[1].plot(np.arange(len(loss_hist_val)), loss_hist_val)
        axs[1].set_title('Validation Loss')
        plt.xlabel('Steps')
        axs[0].set_ylabel('Train Loss')
        axs[1].set_ylabel('Val Loss')
        fig.tight_layout()
        

# import required modules
class LogisticRegression(BaseRegressor):
    def __init__(self, num_feats, learning_rate=0.1, tol=0.0001, max_iter=100, batch_size=12):
        """
        Initialization for Logistic Regression

        Args:
            num_feats (int): Number of features you decide to include in your model.
            learning_rate (float, optional): Sets Learning Rate. Defaults to 0.1.
            tol (float, optional): Sets Tolerance. Defaults to 0.0001.
            max_iter (int, optional): Sets Max iteration. Defaults to 100.
            batch_size (int, optional): Sets Batch Size. Defaults to 12.
        """
        
        # super() will be something commonly seen
        # This python built-in function will allow you to access methods from the class you are inheriting from!
        # This is this class will be able to use train_model and plot_loss_history! :) 
        super().__init__(num_feats, learning_rate, tol, max_iter, batch_size)
        
    def calculate_gradient(self, X, y) -> np.ndarray:
        """
        TODO: Write function to calculate gradient of the
        logistic loss function to update the weights 
        
        Refer to Workshop 2 and 3 for some clues on how to start!

        Params:
            X (np.ndarray): feature values
            y (np.array): labels corresponding to X

        Returns: 
            gradients for a given loss function type np.ndarray (n-dimensional array)
        """
        grad = np.dot(X.T,(self.make_prediction(X)-y))
        return grad
    
    def loss_function(self, X, y) -> float:
        """
        TODO: Get y_pred from input X and implement binary cross 
        entropy loss function. Binary cross entropy loss assumes that 
        the classification is either 1 or 0, not continuous, making
        it more suited for (binary) classification.    

        Params:
            X (np.ndarray): feature values
            y (np.array): labels corresponding to X

        Returns: 
            average loss 
        """
        y_pred = self.make_prediction(X)
        loss = (-1 / y_pred.shape[0]) * np.sum(y * np.log(y_pred) + (1-y)*np.log(1-y_pred))
        return loss
        pass
    
    def make_prediction(self, X) -> np.array:
        """
        TODO: implement logistic function to get estimates (y_pred) for input
        X values. The logistic function is a transformation of the linear model W.T(X)+b 
        into an "S-shaped" curve that can be used for binary classification

        Params: 
            X (np.ndarray): Set of feature values to make predictions for

        Returns: 
            y_pred for given X
        """
        y_pred = 1 / (1 + np.exp(-np.dot(X, self.W)))
        return y_pred
        pass



    
