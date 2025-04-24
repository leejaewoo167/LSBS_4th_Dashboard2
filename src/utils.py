import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import scipy as sp
import scipy.stats as st
import statsmodels.api as sm
warnings.filterwarnings('ignore')


def risk_material(material, 
                  mat_risk: dict):
    
    if material in mat_risk.keys():
        return mat_risk[material]
    else:
        return 0
    
def merge_dict(left: dict, 
               right:dict):
    
    for k,v in right.items():
        if k in left.keys():
            if v == left[k]:
                pass
            else:
                raise "confused component!"
        else:
            left[k] = v
    
    return left

def risk_ranges(sq_ft):
    
    if sq_ft < 1:
        return 1
    elif 1 <= sq_ft and sq_ft <= 100:
        return 2
    elif 100 < sq_ft and sq_ft <= 250:
        return 3
    elif 251 < sq_ft and sq_ft <= 400:
        return 4
    else:
        return 5