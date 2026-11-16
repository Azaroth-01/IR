import pgmpy

import numpy as np
import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

# 1. Load data and standardize column names to lowercase
heartDisease = pd.read_csv('./heart.csv')
heartDisease.columns = map(str.lower, heartDisease.columns)

# 2. Handle missing values
heartDisease = heartDisease.replace('?', np.nan)
heartDisease = heartDisease.dropna()

print('Sample instances from the dataset are given below:')
print(heartDisease.head())
print('\nAttributes and datatypes:')
print(heartDisease.dtypes)

# 3. Define network structure
model = DiscreteBayesianNetwork([
    ('age', 'heartdisease'), 
    ('sex', 'heartdisease'), 
    ('exang', 'heartdisease'), 
    ('cp', 'heartdisease'), 
    ('heartdisease', 'restecg'), 
    ('heartdisease', 'chol')
])

print('\nLearning CPD using Maximum Likelihood Estimators...')
model.fit(heartDisease, estimator=MaximumLikelihoodEstimator)

print('\nInferencing with Bayesian Network:')
heartDisease_infer = VariableElimination(model)

print('\n1. Probability of HeartDisease given evidence = restecg')
q1 = heartDisease_infer.query(variables=['heartdisease'], evidence={'restecg': 1})
print(q1)

print('\n2. Probability of HeartDisease given evidence = cp')
q2 = heartDisease_infer.query(variables=['heartdisease'], evidence={'cp': 2})
print(q2)
