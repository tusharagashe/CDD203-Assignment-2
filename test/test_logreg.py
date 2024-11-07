"""
Write your logreg unit tests here. Some examples of tests we will be looking for include:
* check that fit appropriately trains model & weights get updated
* check that predict is working

More details on potential tests below, these are not exhaustive
"""

import numpy as np
import pandas as pd
from regression import (logreg, utils)
from sklearn.preprocessing import StandardScaler
import pytest

#setup data to use 
X_train, X_val, y_train, y_val = utils.loadDataset(features=['Penicillin V Potassium 500 MG', 'Computed tomography of chest and abdomen', 
																															'Plain chest X-ray (procedure)',  'Low Density Lipoprotein Cholesterol',
																															'Creatinine', 'AGE_DIAGNOSIS'], split_percent=0.8, split_state=42)
# scale data since values vary across features
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_val = sc.transform (X_val)

def test_gradient_calculation():
	X_train_ones = np.ones((3,3))
	y_train_zeros = np.zeros((3,3))
	log_model = logreg.LogisticRegression(num_feats=2, max_iter=10, tol=0.01, learning_rate=0.001, batch_size=12)
	log_model.W = np.array([0.5,0.5,0.5])
	grad = log_model.calculate_gradient(X_train_ones, y_train_zeros)
	assert np.allclose(grad, np.full((3, 3), 2.452723428580931))
 
def test_exploding_gradient():
	log_model = logreg.LogisticRegression(num_feats=6, max_iter=10, tol=0.001, learning_rate=0.1, batch_size=100)

	log_model.train_model(X_train, y_train, X_val, y_val)
	relative_changes = np.abs(np.diff(log_model.loss_history_train) / log_model.loss_history_train[:-1])
    
  # Check if any relative change exceeds the threshold
	assert np.all(relative_changes <= 1.5)
 
def test_vanishing_gradient():
	log_model = logreg.LogisticRegression(num_feats=6, max_iter=10, tol=0.00001, learning_rate=0.0001, batch_size=12)

	log_model.train_model(X_train, y_train, X_val, y_val)
	relative_changes = np.abs(np.diff(log_model.loss_history_train) / log_model.loss_history_train[:-1])

	# Check if any relative change exceeds the threshold
	assert np.all(relative_changes >= 1e-5)

def test_loss_decreasing():
  log_model = logreg.LogisticRegression(num_feats=6, max_iter=10, tol=0.001, learning_rate=0.01, batch_size=100)
  log_model.train_model(X_train, y_train, X_val, y_val)
  assert np.all(np.diff(log_model.loss_history_train)) 
  
def test_loss_converging():
  log_model = logreg.LogisticRegression(num_feats=6, max_iter=100, tol=0.001, learning_rate=0.01, batch_size=100)
  log_model.train_model(X_train, y_train, X_val, y_val)
  max_allowed_change = 0.01
  #check if the loss in the last 10 iterations is varying by lessthan max allowed to change meaning more stability and convergence occuring 
  assert np.all(np.abs(np.diff(log_model.loss_history_train[-10:])) <= max_allowed_change) 

def test_weight_change():
  log_model = logreg.LogisticRegression(num_feats=6, max_iter=100, tol=0.001, learning_rate=0.01, batch_size=100)
  old_weights = log_model.W
  log_model.train_model(X_train, y_train, X_val, y_val)
  assert not np.array_equal(old_weights, log_model.W)

def test_make_prediction():
  X_train_ones = np.ones((3,3))
  y_train_zeros = np.zeros((3,3))
  log_model = logreg.LogisticRegression(num_feats=2, max_iter=10, tol=0.01, learning_rate=0.001, batch_size=12)
  y_pred = log_model.make_prediction(X_train_ones)
  assert np.all(y_pred >= 0) and np.all(y_pred <= 1) and y_pred.shape[0] == X_train_ones.shape[0]
 
def test_accuracy_post_training():
  log_model = logreg.LogisticRegression(num_feats=6, max_iter=100, tol=0.001, learning_rate=0.01, batch_size=100)
  log_model.train_model(X_train, y_train, X_val, y_val)
  y_pred = log_model.make_prediction(np.hstack([X_val, np.ones((X_val.shape[0], 1))]))
  y_pred_binary = (y_pred >= 0.5).astype(int)
  #converts the arrays into booleans and calcualtes mean between T/F
  accuracy = np.mean(y_pred_binary == y_val)
  assert accuracy > 0.5