# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
import numpy as np
#import pandas as pd
import cyipopt
import copy
import polars as pl
import scipy.stats as stats
import pyarrow as pa
import pandas as pd
os.chdir('/home/bizmia/Desktop/Joonhwi/NewProblemSet2')

# %%
## (d) Read the data into memory.
df = pd.read_csv('demand_data.csv')
df['consumer'] = (np.arange(df.shape[0])) % 15 + 1
df.set_index(['marketindex', 'consumer', 'choice'], inplace = True)

## from "x1_prod1" to ("x1", "prod1")
df.columns = df.columns.str.split("_", expand = True)

## choose "alternative" as (one of) the indices 
## leave "attribute" as the main columns
## Convert the data from "wide" to "long"
data = df.stack(level=1).rename_axis(['marketindex', 'consumer', 'choice', 'alternative']).reset_index()

## convert "prod1" (str) to 1 (int)
data['alt'] = data['alternative'].str[4].astype("str").astype("int")
data['choice_alt'] = data['choice'] == data['alt'].astype("int")
data = data.drop('alternative', 1)
# data = data.drop('choice', 1)

data['const_0'] = [1, 0, 0, 0] * (int(round(data.shape[0] / 4)))
data['const_1'] = [0, 1, 0, 0] * (int(round(data.shape[0] / 4)))
data['const_2'] = [0, 0, 1, 0] * (int(round(data.shape[0] / 4)))
data['const_3'] = [0, 0, 0, 1] * (int(round(data.shape[0] / 4)))

del df

# %%
# Derive the market shares by aggregation
data_agg = data.groupby(["marketindex","choice"]).size()

idx = pd.MultiIndex.from_product([data_agg.index.levels[0], data_agg.index.levels[1]])
data_agg = data_agg.reindex(idx, fill_value = 0)

data_agg = data_agg.to_frame(name = 'size').reset_index()
data_agg['size'] = data_agg['size'] / 4
# data_agg.set_index(['marketindex'], inplace = True)

## Remove the markets where the share for s_0 is 0
data_agg = data_agg.groupby('marketindex').filter(lambda x: x.iloc[0]['size'] > 0)
data_agg.rename({'choice': 'alt'}, axis=1, inplace=True)

## Create columns for log-share
data_agg['share'] = data_agg['size'] / data_agg.groupby('marketindex')['size'].transform('first')
data_agg['log_share'] = np.log(data_agg['share'])

## Remove rows where share == 0 
data_agg = data_agg.loc[data_agg['share'] > 0]
data_agg = data_agg.drop('size', 1).drop('share', 1)

## Merge with Attributes 
data_att = data.drop('consumer', 1).drop('choice', 1)
# data_att.drop_duplicates(keep = False, inplace = True)
## data_att.set_index(['marketindex'], inplace = True)
## Remove duplicate rows
## data_att.drop_duplicates(subset = ['marketindex'], keep = "last")
data_att = data_att.groupby('marketindex').head(4)

data_merge = data_agg.merge(data_att, on = ['marketindex', 'alt'], how = 'left')

# %%
## (e) write function of \beta which returns the score vector and the exact Jacobian of the score vector
##### The multi_nomial logit class
class mnl:
    def __init__(self, X, y):
        # The input must be a numpy array
        # Declare the data as the "global" variable within the class
        self.X = X
        self.y = y
        
        ## Find the size of data
        self.n = np.shape(self.X)[0]
    
    def objective(self, beta):
        # The callback for calculating the objective function
        likelihood = [0] * self.n
        for i in range(self.n / 4):
            X_tmp = self.X[(i*4):(i*4+4)]
            y_tmp = self.y[(i*4):(i*4+4)]
            # G_tmp = np.exp(X_tmp[y_tmp] @ beta) / np.sum(np.exp(X_tmp @ beta))
            
            A = np.exp(X_tmp @ beta) 
            B = np.sum(np.exp(X_tmp @ beta))
            
            G_tmp = np.log(A / B) * y_tmp.T
            likelihood[i * 4 : (i * 4 + 4)] = G_tmp
            
        obj = np.sum(likelihood)    
        return(obj)

    def gradient(self, beta):
        ## (e) The function that calculates the score function
        score = [[0] * 7] * round(self.n / 4)
        for i in range(self.n / 4):
            X_tmp = self.X[(i*4):(i*4+4)]
            y_tmp = self.y[(i*4):(i*4+4)]
            # G_tmp = np.exp(X_tmp[y_tmp] @ beta) / np.sum(np.exp(X_tmp @ beta))
            
            A = np.exp(y_tmp.T @ X_tmp @ beta) 
            B = np.sum(np.exp(X_tmp @ beta))
            
            C = (X_tmp - y_tmp.T @ X_tmp).reshape([4,3])
            D = np.exp(X_tmp @ beta)
            E = np.transpose(C) @ D
            
            G_tmp = A / B
            g_tmp = A * E / B**2
            score[i] = g_tmp / G_tmp
            
        return(np.sum(score, axis = 0))
    
    def hessianstructure(self):
            # The structure of the Hessian
        self.k = self.X.shape[1]
        hess_struc = np.tril(np.ones((self.k, self.k)))
            # Create a sparse matrix to hold the hessian structure
        hess_struc = np.nonzero(hess_struc) # This will return two set of indices
                 
        print(hess_struc)
        return(hess_struc)

    def hessian(self, beta, lagrange, obj_factor):
            # Redefine parameter as column vector to use matrix multiplication
        beta = beta.reshape(len(beta),1)
            # The callback for calculating the Hessian
        jacobian = [[[0] * 7] * 7] * round(self.n / 4)
        for i in range(self.n / 4):
            X_tmp = self.X[(i*4):(i*4+4)]
            y_tmp = self.y[(i*4):(i*4+4)]
            # G_tmp = np.exp(X_tmp[y_tmp] @ beta) / np.sum(np.exp(X_tmp @ beta))
            
            A = np.exp(y_tmp.T @ X_tmp @ beta) 
            B = np.sum(np.exp(X_tmp @ beta))
            
            C = (X_tmp - y_tmp.T @ X_tmp).reshape([4,3])
            D = np.exp(X_tmp @ beta)
            E = np.transpose(C) @ D
            
            # G_tmp = A / B
            # g_tmp = A * E / B**2
            
            F = C[0].reshape([3,1]) @ X_tmp[0].reshape([1,3]) * D[0] + C[1].reshape([3,1]) @ X_tmp[1].reshape([1,3]) * D[1] + C[2].reshape([3,1]) @ X_tmp[2].reshape([1,3]) * D[2] + C[3].reshape([3,1]) @ X_tmp[3].reshape([1,3]) * D[3] 
            jacobian[i] = A / (B)**3 * (E.reshape([3,1]) @ E.reshape([1,3]) + F)
            
        hess = -1 * obj_factor * np.sum(jacobian, axis = 0)
        hess = np.tril(hess)
            
        row, col = self.hessianstructure()
        return(hess[row, col])
        
    def intermediate(
            self,
            alg_mod,
            iter_count,
            obj_value,
            inf_pr,
            inf_du,
            mu,
            d_norm,
            regularization_size,
            alpha_du,
            alpha_pr,
            ls_trials
            ):
        print("Objective value at iteration #%d is - %g" % (iter_count, obj_value))

# %%
##### Execute the nonlinear optimization to find the OLS estimator
# Define the problem
# Initialization value
X = data[['x1', 'x2', 'x3', 'const_0', 'const_1', 'const_2', 'const_3']].to_numpy()
y = data[['choice_alt']].to_numpy()
beta0 = [1, 2, 3, 4, 5, 6, 7]
# Parameter lower and upper bounds
# lb = None#[1.0, 1.0, 1.0, 1.0]
# ub = None#[5.0, 5.0, 5.0, 5.0]
   
# Constraint lower and upper bounds
# cl = None
# cu = None
   
# Initialize the problem
mnl_nonlinear = cyipopt.Problem(
    n = len(beta0),        # Dimension of parameter
    m = 0,                  # Dimension of constraints
    problem_obj = mnl(X, y), # Problem
    # lb = lb,                # Parameter lower bound
    # ub = ub,                # Parameter upper bound
    # cl = cl,                # Constraint lower bound
    # cu = cu                 # Constraint upper bound             
       )
          
   # IPOPT options
mnl_nonlinear.add_option('print_level', 5)
mnl_nonlinear.add_option('linear_solver', 'ma57')
   #OLS_nonlinear.add_option('derivative_test', 'second-order')
# mnl_nonlinear.add_option('derivative_test', 'none')
# mnl_nonlinear.add_option('jac_d_constant', 'no')
# mnl_nonlinear.add_option('hessian_constant', 'no')
mnl_nonlinear.add_option('hessian_approximation', 'limited-memory') # Approximate Hessian
# mnl_nonlinear.add_option('hessian_approximation', 'exact')
mnl_nonlinear.add_option('mu_strategy', 'adaptive')
mnl_nonlinear.add_option('max_iter', 10000)
mnl_nonlinear.add_option('tol', 1e-8)
mnl_nonlinear.add_option('acceptable_tol', 1e-8)

    # Execute the nonlinear optimization
beta_hat_nonlinear, info = mnl_nonlinear.solve(beta0)
print("beta_hat using IPOPT:" + str(beta_hat_nonlinear))


# %%
##### (g) Market-level Analysis using OLS
##### The lm_regression class
class lm_regression:
    def __init__(self, X, y):
        # This function takes X and y vector and runs OLS
        # The input must be a numpy array
        # Declare the data as the "global" variable within the class
        self.X = X
        self.y = y
        
        ## Find the size of data
        self.n = np.shape(self.X)[0]
        self.k = np.shape(self.X)[1]
        
        ## Calculate OLS
        self.Xty = self.X.T @ self.y
        self.XtX = self.X.T @ self.X
        self.beta_hat = np.linalg.inv(self.XtX) @ self.Xty
    
    ## The function that calculates the OLS prediction
    def predictions_f(self):
        self.y_hat = self.X @ self.beta_hat
        return(self.y_hat)
        
    ## The function that calculates the OLS residuals        
    def residuals_f(self):
        #self.u_hat = self.y - self.predictions_f()
        self.u_hat = self.y - self.y_hat
        return(self.u_hat)
    
    ## The function that calculates s^2    
    def s_squared_f(self):
        u_hat = self.u_hat
        self.RSS = u_hat.T @ u_hat
        self.s_sq = self.RSS / (self.n-self.k)
        return(self.s_sq)
    
    ## The function that calculates var(beta_hat)
    def var_beta_hat_f(self):
        self.var_beta_hat = self.s_sq * np.linalg.inv(self.XtX)
        #return(self.var_beta_hat)
        return(self.var_beta_hat)
    
    ## The function that calculates TSS, RSS, R_sq
    def r_sq_f(self):
        # Demean the variables first
        y_variation = self.y - np.average(self.y)
        yhat_variation = self.y_hat - np.average(self.y)
                
        self.TSS = np.matmul(y_variation.T, y_variation) 
        self.ESS = np.matmul(yhat_variation.T, yhat_variation)
        R_sq1 = 1 - (self.RSS / self.TSS)
        R_sq2 = self.ESS / self.TSS
        
        return(R_sq2, R_sq1)
    
    ## The function that constructs the t statistics and executes it
    def t_test_f(self, cons_coeff, cons_const):
        test_stat = ((cons_coeff.T @ self.beta_hat) / 
                               np.sqrt(cons_coeff.T @ self.var_beta_hat @ cons_coeff))
        test_stat = test_stat - cons_const
        # Two-sided p value
        p_val = 2*(1-stats.norm.cdf(np.abs(test_stat), loc=0, scale=1))
        
        return(test_stat, p_val)    

# %%
    # Subset y
    y_ols = data_merge[['log_share']].to_numpy()
    
    # Subset X
    X_ols = data_merge[['x1', 'x2', 'x3', 'const_0', 'const_1', 'const_2', 'const_3']].to_numpy()
    
    ### Execute the regression
    # The following need to be run sequentially due to the structure of the 
    # global variable defined in the class
    lm_result = lm_regression(X_ols, y_ols)
    beta_hat = lm_result.beta_hat
    y_hat = lm_result.predictions_f()
    u_hat = lm_result.residuals_f()
    s_sq = lm_result.s_squared_f()
    var_beta_hat = lm_result.var_beta_hat_f()
    R_sq1, R_sq2 = lm_result.r_sq_f()
    
    ### Print the results to console
    print("beta_hat: \n" + str(beta_hat))
    print("y_hat: \n" + str(y_hat))
    print("sum of residuals: \n" + str(np.sum(u_hat)))
    print("sum of x_i u_i: \n" + str(X_ols.T.dot(u_hat)))
    print("s_squared: \n" + str(s_sq))
    print("var(beta_hat): \n" + str(var_beta_hat))
    print("R_squared using different formulas: \n" + str(R_sq1) + str(R_sq2))











