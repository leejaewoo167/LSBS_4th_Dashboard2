import os
import shutil
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import scipy as sp
import scipy.stats as st
import statsmodels.api as sm
from statsmodels.formula.api import ols
import statsmodels.formula.api as smf
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso, RidgeCV, LassoCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, root_mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.datasets import load_iris
from sklearn import linear_model
from tqdm import tqdm
warnings.filterwarnings('ignore')

os.chdir('./src')
from dataloader import DataLoader

# load data 

dataloader = DataLoader()
dataset = dataloader.load_data()
