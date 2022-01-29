#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 22:25:25 2022

@author: bizmia
"""

import numpy as np
import cyipopt
import pandas as pd
data = pd.read_csv("/home/bizmia/Desktop/Joonhwi/Task1/cps09mar.csv")
data['ones'] = 1
data['log_wage'] = np.log(data.earnings)
y = data.log_wage.to_numpy()
X = data[['ones','female','education', 'hours']].to_numpy()

class ols:

    def __init__(self):
        pass

    def objective(self, beta):
        #
        # The callback for calculating the objective
        #
        return np.transpose(y - X @ beta) @ (y - X @ beta)

    def gradient(self, beta):
        #
        # The callback for calculating the gradient
        #
        return - 2 * np.transpose(X) @ y + 2 * np.transpose(X) @ X @ beta

    def hessianstructure(self):
        return np.nonzero(np.tril(np.ones((4, 4))))

    def hessian(self, beta, lagrange, obj_factor):
        #
        # The callback for calculating the Hessian
        #
        
        H = 2 * np.transpose(X) @ X 
        row, col = self.hessianstructure()
        return H[row, col]

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

        #
        # Example for the use of the intermediate callback.
        #
        print("Objective value at iteration #%d is - %g" % (iter_count, obj_value))


def main():
    #
    # Define the problem
    #
    x0 = [0.0, 0.0, 0.0, 0.0]

    # lb = [1.0, 1.0, 1.0, 1.0]
    # ub = [5.0, 5.0, 5.0, 5.0]

    cl = []
    # cu = []

    nlp = cyipopt.Problem(
        n=len(x0),
        m=len(cl),
        problem_obj=ols(),
        # lb=lb,
        # ub=ub,
        # cl=cl,
        # cu=cu
        )

    #
    # Set solver options
    #
    #nlp.addOption('derivative_test', 'second-order')
    nlp.add_option('mu_strategy', 'adaptive')
    nlp.add_option('tol', 1e-7)
    nlp.add_option('linear_solver', 'ma57')
    nlp.add_option('hessian_approximation', 'exact')
    #
    # Scale the problem (Just for demonstration purposes)
    #
    nlp.set_problem_scaling(
        obj_scaling=2,
        x_scaling=[1, 1, 1, 1]
        )
    nlp.add_option('nlp_scaling_method', 'user-scaling')

    #
    # Solve the problem
    #
    x, info = nlp.solve(x0)

    print("Solution of the primal variables: x=%s\n" % repr(x))

    print("Solution of the dual variables: lambda=%s\n" % repr(info['mult_g']))

    print("Objective=%s\n" % repr(info['obj_val']))


if __name__ == '__main__':
    main()
