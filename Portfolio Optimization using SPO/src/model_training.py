import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
import sys
import os
import itertools
from concurrent.futures import ThreadPoolExecutor
import math
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import missingno
import random
from statsmodels.tsa.stattools import acf, pacf
import gurobipy as gp
from gurobipy import GRB
import tensorflow as tf
from tensorflow.keras import initializers
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression
from tqdm import tqdm
import shap


''' This file contains all the necessary utilities and functions needed for
    model training with SPO framework.
'''


def oracle(cost_vec, sigma, gamma):
    """ 
        Oracle is an mixed integer programming problem 
        that solves portfolio optimization problem as per the given cost_vec
        covariance matrix (sigma) and risk factor (gamma)
        
        Args:
            cost_vec (:obj: `list` of :obj: `float`): A list of returns of all the securities to be optimized
            sigma (:obj: `list` of :obj: `list` of :obj: `float`): A covariance matrix of all the securities
            gamma (float): Maximum risk (or variance): that the optimized portfolio is allowed to have
        
        Returns:
            obj_value (float): The overall cost of optimized portfolio
            values (:obj: `list` of :obj: `float`): Optimum percentage share of assets allocated to each of the security
    """
    
    # Create an empty model
    model = gp.Model('portfolio')
    model.Params.LogToConsole = 0
    
    # Add matrix variable for the stocks
    numbr_vars = len(cost_vec)
    x = model.addMVar(numbr_vars)
    
    # Objective is to minimize risk (squared).  This is modeled using the
    # covariance matrix, which measures the historical correlation between stocks
    portfolio_risk = cost_vec @ x
    model.setObjective(portfolio_risk, GRB.MINIMIZE)
    
    # Fix budget with a constraint
    model.addConstr(x.sum() <= 1, 'budget')
    model.addConstr(x @ sigma @ x <= gamma, 'maximum variability')
    
    # Verify model formulation
    model.write('portfolio_selection_optimization.lp')
    
    # Optimize model to find the minimum risk portfolio
    model.optimize()
    
    # Return objective value and weights of variables
    all_vars = model.getVars()
    values = model.getAttr("x", all_vars)
    
    obj = model.getObjective()
    obj_value = obj.getValue()
    
    return obj_value, values


def spo_plus_loss(c_hat_mat, cost_mat, sigma, gamma, batch_size=16):
    """
        Calculates SPO+ loss using c_hat (prediction) and cost_vec (real values)
        z_star is optimal objective value and w_star is optimal weights of the optimization model w.r.t cost_vec
    
        Args:
            c_hat_mat (:obj: `list` of :obj: `list` of :obj: `float`): Matrix of predicted returns of selected 
                                                                        securities over a batch of data
            cost_mat (:obj: `list` of :obj: `list` of :obj: `float`): Matrix of observed returns of selected 
                                                                        securities over a batch of data
            sigma (:obj: `list` of :obj: `list` of :obj: `float`): A covariance matrix of all the securities
            gamma (float): Maximum risk (or variance): that the optimized portfolio is allowed to have
            batch_size (int): Number of datapoints in a batch
            
        Returns:
            avg_spo_plus_loss (float): Average loss over the batch
            loss_calculations (dict of str: int): A dictionary logging objective value, optimal shares and total loss of every 
                                                    datapoint in the batch
    
    """
    
    
    total_spo_loss = 0
    loss_calculations = {"c_hat": [],
                       "cost_vec": [],
                       "z_oracle":[],
                       "w_oracle":[],
                       "z_star":[],
                       "w_star":[],
                       "total_spo_loss":[]}
    for i in range(batch_size):
        c_hat = c_hat_mat[i]
        cost_vec = cost_mat[i]
        spo_plus_cost_vec = (2*c_hat) - cost_vec
        z_oracle, w_oracle = oracle(spo_plus_cost_vec, sigma, gamma)
        z_star, w_star = oracle(cost_vec, sigma, gamma)
        spo_plus_cost = -z_oracle + 2*(c_hat @ w_star) - z_star
        total_spo_loss = total_spo_loss + spo_plus_cost
        loss_calculations["c_hat"].append(c_hat)
        loss_calculations["cost_vec"].append(cost_vec)
        loss_calculations["z_oracle"].append(z_oracle)
        loss_calculations["w_oracle"].append(w_oracle)
        loss_calculations["z_star"].append(z_star)
        loss_calculations["w_star"].append(w_star)
        loss_calculations["total_spo_loss"].append(total_spo_loss)
        
    avg_spo_plus_loss = total_spo_loss/batch_size

    return avg_spo_plus_loss, loss_calculations


def spo_plus_subgrad_new(features_vec, cost_vec, c_hat_vec, sigma, gamma):
    """ 
        Calculates sub-gradients with respect to the SPO+ loss
    
        Args:
            features_vec (:obj: list of float): Values of features for a particular target
            cost_vec (:obj: list of float): Vector containing observed cost of all the securities
            c_hat_vec (:obj: list of float): Vector containing predicted cost of all the securities
            sigma (:obj: `list` of :obj: `list` of :obj: `float`): A covariance matrix of all the securities
            gamma (float): Maximum risk (or variance): that the optimized portfolio is allowed to have
            
        Returns:
            G_new (:obj: list of float): Vector containing gradient value for each feature
    """
    n_socks = cost_vec.shape[0]
    n_rows , n_features = features_vec.shape
    
    # Define a zero tensor of shape (1,17)
    G_new = tf.zeros(shape= (1, n_features))
    
    
    # calculate optimal values for the difference of the predicted and real cost vectors
    spo_plus_cost_vec = (2*c_hat_vec) - cost_vec
    z_oracle, w_oracle_list = oracle(spo_plus_cost_vec, sigma, gamma)
            
    # calculate optimal values for real cost vectors
    z_star, w_star_list = oracle(cost_vec, sigma, gamma)
            
    # takinf difference between optimal values from last two steps
    w_oracle = tf.convert_to_tensor(w_oracle_list)
    w_star = tf.convert_to_tensor(w_star_list)
    w_star_diff = w_star - w_oracle
    w_star_diff = tf.reshape(w_star_diff, shape = (n_socks, 1))
            
            
    # reshaping training data to (1,17)
    feats = tf.cast(features_vec, dtype=tf.float32)
            
    # define a multiplier as per the paper and calculate gradient for current run
    multiplier = tf.constant(2, dtype=tf.float32)
    w_x = w_star_diff*(feats)
    w_x_zero = tf.reshape(w_x[0], (1,n_features))
            
    # cummumate the gradient with previous run
    G_new = G_new + multiplier*w_x_zero
    
    return G_new


def get_c_hat_vec(df_cost_vec, c_hat, ticker = "AAPL"):
    """
        Add a predicted value c_hat to a cost vector to get predicted cost vector i.e. c_hat_vec
        
        Args:
            df_cost_vec (pd.dataframe): A pandas dataframe with just one row containing cost of securities
            c_hat (float): Predicted value for a particulat security e.g. AAPL
            ticker (str, optional): Name of ticker to for wich the value c_hat has to be assigned in the df_cost_vec
            
        Returns:
            c_hat_vec (:obj: list of float): Vector containing predicted cost of the securities
    """
    df_c_hat_vec = df_cost_vec.copy(deep = True)
    df_c_hat_vec[ticker] = c_hat.numpy()
    c_hat_vec = df_c_hat_vec.values
    
    return c_hat_vec


def SGD_regressor(training_dataframe, model, cost_df, sigma, gamma, learning_rate= 0.1, decay_rate = 1.02, n_epochs=50, batch_size = 16, decay = False):
    """ 
        This finction performs minibatch stochastic gradient descent to optimise model parameters
        
        Args:
            training_dataframe (pd.dataframe): Pandas dataframe for training
            model (:obj: Sequential): Keras sequential model to be optimized
            cost_df (pd.dataframe): pandas dataframe containing periodic cost of all the securities
            sigma (:obj: `list` of :obj: `list` of :obj: `float`): A covariance matrix of all the securities
            gamma (float): Maximum risk (or variance): that the optimized portfolio is allowed to have
            learning_rate (float, optional): Learning rate for gradient descent, default value is 0.1
            decay_rate (float, optional): The learning rate will be diided by this amount, default value is 1.02
            n_epochs (int, optional): Number of training epochs, default value is 50
            batch_size (int, optional): Number of datapoints in a batch, default value is 16
            decay (bool, optional): If true the learning would be decayed at every epoch otherwise the model will be trained on 
                            a constant learning rate, by default it is false
            
        Returns:
            model (:obj: Sequential): Optimized keras sequential model
            epoch_loss (:obj: list of float): List of values of loss at every epoch
            
    """
    
    
    n_rows, n_cols = training_dataframe.shape
    n_features = n_cols-1
    epoch_loss = []
    for epoch in range(n_epochs):
        
        c_hat_mat = []
        cost_mat = []
        
        batch_dataframe = training_dataframe.sample(batch_size)
        batch_features = batch_dataframe.iloc[:,0:n_features].values
        batch_target = batch_dataframe.iloc[:,-1].values
        batch_cost_df = cost_df.iloc[batch_dataframe.index]
        
        # calculate gradients
        for idx in range(batch_size):
            gradient_list = []
            df_cost_vec = batch_cost_df.iloc[idx]
            features_vec = tf.reshape(batch_features[idx], shape = (1,n_features))
            c_hat = model(features_vec)
            
            c_hat_vec = get_c_hat_vec(df_cost_vec, c_hat)
            cost_vec = df_cost_vec.values
            
            '''Call to spo+ sub gradient function'''
            gradients = spo_plus_subgrad_new(features_vec, cost_vec, c_hat_vec, sigma, gamma)
            
            grads_reshaped = tf.reshape(gradients, shape = (n_features, 1))
            trainable_weights = model.get_weights()
            trainable_weights[0] = trainable_weights[0] - (learning_rate* grads_reshaped)
            model.set_weights(trainable_weights)
            
            c_hat_new = model(features_vec)
            c_hat_vec_new = get_c_hat_vec(df_cost_vec, c_hat_new)
            
            c_hat_mat.append(c_hat_vec_new)
            cost_mat.append(cost_vec)
        
        epoch_spo_plus_loss, loss_params = spo_plus_loss(c_hat_mat, cost_mat, sigma, gamma, batch_size)
        epoch_loss.append(epoch_spo_plus_loss)
        
        if decay:
            learning_rate = learning_rate/decay_rate
            
    return model, epoch_loss


def get_model(n_feats = 1, units= 1, use_bias=True):
    """
        This finction initializes a keras sequential model
        
        Args:
            n_feats (int, optional): number of features in your training data set
            units (int, optional): number of units in single dense layer
            use_bias (bool, optional): If True uses bias term too 
            
        Returns:
            linear_model (:obj: Sequential): Keras sequential model
    """
    
    # Input layer
    input_layer = tf.keras.Input(shape= (n_feats,))
    
    # Dense layer
    if use_bias:
        layer_1 = tf.keras.layers.Dense(units=units,
                                kernel_initializer=initializers.Zeros(),
                                bias_initializer=initializers.Zeros(),
                                trainable=True)
    else:
        layer_1 = tf.keras.layers.Dense(units=units,
                                kernel_initializer=initializers.Zeros(),
                                bias_initializer=initializers.Zeros(),
                                trainable=True,
                                use_bias=False)
    
    # Linear regression model
    linear_model = tf.keras.Sequential([
        input_layer,
        layer_1
    ])
    
    return linear_model


def select_features(features, target, k):
    """
        Selects k-best features for a given dataset
        
        Args:
            features (pd.dataframe): Containing only features
            target (pd.dataframe): Containing only target variable
            k (int): Number of important features to be calculated, e.g. if k=5 then top 5 most important features will be returned
            
        Returns:
            features_fs (:obj: `list` of :obj: `list` of :obj: `float`): Most importane feature values
            fs (:obj: `SelectKBest`): Object containing all the information of best features including feature importance values
    """
    
    # configure to select all features
    fs = SelectKBest(score_func=mutual_info_regression, k=k)
    # learn relationship from training data
    fs.fit(features, target)
    # transform train input data
    features_fs = fs.transform(features)
    return features_fs, fs


def grid_search(train_df, returns_df, sigma, gamma, n_epoch, GridSearchParams):
    """
        Performs grid search hyperparameter tuning and finds out best model based on SPO+ loss
        
        Args:
            train_df (pd.dataframe): Training set
            returns_df (pd.dataframe): Dataframe containing returns of all the securities over a period
            sigma (:obj: `list` of :obj: `list` of :obj: `float`): A covariance matrix of all the securities
            gamma (float): Maximum risk (or variance): that the optimized portfolio is allowed to have
            n_epoch (int, optional): Number of training epochs
            GridSearchParams (dict of str: :obj: `list' of `float`): Adictionary containing all the parameters and their expected values
            
       Returns:
           Results (:obj: `list`): List of loss of all the parameter combinations
           BestParams (dict of str: float): Dictionary of best parameters
           MinError (float): Aberage validation loss with the best parameters
    """
    
    Results = []
    Keys = GridSearchParams.keys()
    MinError = sys.maxsize
    BestParams = {}
    ParamSets = [dict(zip(Keys, values)) for values in itertools.product(*GridSearchParams.values())]
    
    df_train, df_test = train_test_split(train_df, test_size=0.2, random_state=42, shuffle=False)
    df_returns_train, df_returns_test = train_test_split(returns_df, test_size=0.2, random_state=42, shuffle=False)
    test_n_rows, test_n_cols = df_returns_test.shape
    
    for params in ParamSets:
        decay_rate = params.get("decay_rate")
        batch_size = params.get("batch_size")
        learning_rate = params.get("learning_rate")
        
        n_rows, n_cols = df_train.shape
        n_feats = n_cols-1
        
        # Instantiate the model
        model = get_model(n_feats = n_feats)
        
        # Train the model
        trained_model, epoch_loss_list = SGD_regressor(df_train, model, df_returns_train, 
                                                               sigma, gamma, learning_rate= learning_rate, decay_rate = decay_rate, 
                                                               n_epochs=n_epoch, batch_size = batch_size)
        
        # Predict using the trained model
        y_pred = trained_model(df_test.iloc[:,0:n_feats].values)
        
        c_hat_mat = []
        start_idx = df_train.index[-1]
        for idx, c_hat in enumerate(y_pred):
            df_cost_vec = df_returns_test.iloc[idx]
            c_hat_vec_new = get_c_hat_vec(df_cost_vec, c_hat)
            c_hat_mat.append(c_hat_vec_new)
            
        cost_mat = df_returns_test.values

        # Calculate the error using the SPO+ loss function
        spo_plus_val_loss, loss_params = spo_plus_loss(c_hat_mat, cost_mat, sigma, gamma, test_n_rows)
        
        print(f'Validation loss = {spo_plus_val_loss}')
        Results.append((params, spo_plus_val_loss))

        if spo_plus_val_loss < MinError:
            MinError = spo_plus_val_loss
            BestParams = params
            
    return Results, BestParams, MinError


def get_SPO_plus_testing_loss(df_train, df_returns_test, y_pred, sigma, gamma):
    """
        Returns spo+ loss on testing set
        
        Args:
            df_train (pd.dataframe): Datframe containing observed values of the target as last column
            df_returns_test (pd.dataframe): Dataframe with cost matrix for test set
            y_pred (list): Predicted value of returns for test dataset
            sigma (:obj: `list` of :obj: `list` of :obj: `float`): A covariance matrix of all the securities
            gamma (float): Maximum risk (or variance): that the optimized portfolio is allowed to have
            
        Returns:
            spo_plus_test_loss (float): Average test SPO+ loss
    """
    
    
    test_n_rows, test_n_cols = df_returns_test.shape
    c_hat_mat = []
    start_idx = df_train.index[-1]
    for idx, c_hat in enumerate(y_pred):
        df_cost_vec = df_returns_test.iloc[idx]
        c_hat_vec_new = get_c_hat_vec(df_cost_vec, c_hat)
        c_hat_mat.append(c_hat_vec_new)
            
    cost_mat = df_returns_test.values

    # Calculate the error using the SPO+ loss function
    spo_plus_test_loss, loss_params = spo_plus_loss(c_hat_mat, cost_mat, sigma, gamma, test_n_rows)
    
    return spo_plus_test_loss