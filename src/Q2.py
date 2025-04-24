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
from dataloader import DataLoader
warnings.filterwarnings('ignore')

os.chdir('../src')
dataloader = DataLoader()

dataset = dataloader.load_data()

# ---------------------------
# 💰 지역별 '평단가' 기반 등급 분류 (5단계)
# ---------------------------

#  위험도 평균 열 생성
dataset['Risk_Avg'] = (
    dataset['Risk_RoofMatl'] * 0.30 +
    dataset['Risk_Exterior1st'] * 0.30 +
    dataset['Risk_Exterior2nd'] * 0.10 +
    dataset['Risk_MasVnrType'] * 0.10 +
    dataset['Risk_WoodDeckSF'] * 0.2
)

# 위험도 평균을 5단계로 그룹화
dataset['Risk_Level'] = dataset['Risk_Avg'].round()
dataset['Risk_Level'].shape

# 페이지 2 내용 필요없음

# # 위험도별 주택 개수 확인
# # 위험도 5는 1개 밖에없어서 제거한다고 설명할때 사용 가능!
# cnt_RiskLevel = dataset['Risk_Level'].value_counts().sort_index()

# plt.figure(figsize=(6, 4))
# cnt_RiskLevel.sort_index().plot(kind='bar', color='salmon', edgecolor='black')
# plt.xlabel('Risk_Level')
# plt.ylabel('# of house by risk level')
# plt.title('Risk_level_house_cnt')
# plt.grid(axis='y', linestyle='--', alpha=0.5)
# plt.xticks(rotation=0)

# 화재 위험도별 평단가 두개다 막대그래프로 그래프 그리기 필요 코드
mean_RiskLevel = dataset.groupby('Risk_Level')['BuildingPricePerTotalSF'].mean()

# 중앙값 평단가 바 그래프 그리기 위해 필요 코드
median_RiskLevel = dataset.groupby('Risk_Level')['BuildingPricePerTotalSF'].median()


# 결측값 제거 및 위험도 5 제거 (분석을 위해)
dataset = dataset.dropna(subset=['BuildingPricePerTotalSF'])

#########################################################################3
##################################################################3333333333333333333333

dataset = dataset[dataset['Risk_Level'] != 5]

# 분산분석 과정
import statsmodels.api as sm
from statsmodels.formula.api import ols


model = ols('BuildingPricePerTotalSF ~ C(Risk_Level)',data=dataset).fit()
anova_results = sm.stats.anova_lm(model, typ=2)

# 아노바 검정 결과
print(anova_results)
# 해석
# 분산분석 결과 위험도별 가격이 차이가 있다는 것을 확인 (단 분산분석을 믿을수 있는가?)
# 아래 잔차의 정규성 검정 및 잔차의 등분산성 검정으로 확인

import scipy.stats as sp
W, p = sp.shapiro(model.resid)
# 아노바 검정 결과
# 잔차 정규성 검정 결과 출력해야하는 내용 !!!!!!
print(f'검정통계량: {W:.3f}, 유의확률: {p:.3f}')
# 해석
# 잔차의 정규성 검정 방법중 샤피로 위크 검정결과 잔차의 정규성이 성립한다는 귀무가설 기각
# 잔차의 정규성 만족안함

# 아노바 검정 결과
# 잔차 정규성 검정 결과 출력해야하는 내용 !!!!!!
from scipy.stats import probplot
plt.figure(figsize=(6, 6))
probplot(model.resid, dist="norm", plot=plt)
# 해석 잔차 정규성 만족안함



# bartlett을 사용한 잔차의 등분산성 검증 결과 등분산성 역시 성립하지 않음
from scipy.stats import bartlett
from scipy.stats import kruskal
groups = [1, 2, 3, 4]
grouped_residuals = [model.resid[dataset['Risk_Level'] == group] for group in groups]
test_statistic, p_value = bartlett(*grouped_residuals)
# 아노바 검정 결과
# 출력해야 하는 내용!!!
print(f"검정통계량: {test_statistic}, p-value: {p_value}")


# 아노바 검정결과 x 여기부터 비모수 검정 부분 step12 나누는거로 해야할듯
# 따라서 Kruskal-Wallis 검정 (비모수 검정)을 통해 위험도별 주택 평단가의 차이가 통계적으로 유의미한지 확인
grouped = [group['BuildingPricePerTotalSF'].values for name, group in dataset.groupby('Risk_Level')]

kruskal_stat, kruskal_p = kruskal(*grouped)

# Kruskal-Wallis 검정결과
kruskal_result = {
    "검정통계량 (H)": kruskal_stat,
    "p-value": kruskal_p,
    "결론": "✔️ 그룹 간 차이가 유의함 (p < 0.05)" if kruskal_p < 0.05 else "❌ 유의한 차이 없음 (p ≥ 0.05)"
}
# 위험도별 주택 평단가 차이가 하나 이상은 존재함을 확인
# 따라서 사후검정을 통해 어떤 위험도끼리 차이가 있는지 확인

# 출력해야하는 결과
kruskal_result


# dunn-test(비모수 사후검정)
# 이것도 크루스칼이랑 같이 두번째 step에
import scikit_posthocs as sp
posthoc = sp.posthoc_dunn(dataset, val_col='BuildingPricePerTotalSF', group_col='Risk_Level', p_adjust='bonferroni')
# 비모수 사후검정 실시 결과 위험도 2말고는 차이가 있음을 확인 불가
# 결과보여주기 위해 출력해야하는 부분
posthoc

# 위험도 2에 해당하는 평단가가 다른 위험도에 비해 높을 수 있다.
# 단 이것이 화재 안정성이 높은 자재가 집값을 비싸게 만든다고 볼수없다.

# import pandas as pd
# import plotly.graph_objects as go

# # 색상 설정
# color_map = {
#     1: 'white', 2: 'gray', 3: 'yellow', 4: 'orange', 5: 'red'
# }

# # 지도 레이아웃
# layout_mapbox = dict(
#     mapbox=dict(style="open-street-map", center=dict(lat=42.0345, lon=-93.62), zoom=11),
#     margin={"r": 0, "t": 40, "l": 0, "b": 0},
#     title='Ames 시 위험도 기반 주택 시각화 & 소방서 위치'
# )

# # 주택 마커
# traces = []
# for level, color in color_map.items():
#     df = dataset[dataset['Risk_Level'] == level]
#     traces.append(go.Scattermapbox(
#         lat=df['Latitude'], lon=df['Longitude'],
#         mode='markers',
#         marker=dict(size=7, color=color, opacity=0.6),
#         text='가격: $' + df['BuildingPricePerTotalSF'].astype(str) + '<br>위험도: ' + df['Risk_Level'].astype(str),
#         name=f'위험도 {level}'
#     ))

# # 시각화
# fig = go.Figure(data=traces, layout=layout_mapbox)
# fig.show()


# 위험도별 주택가격 Box Plot에 내야하는 부분
# 리스크 대비 평균 가격 
fig, ax = plt.subplots(figsize=(6, 4))
mean_RiskLevel.sort_index().plot(kind='bar', color='salmon', edgecolor='black')
_ = ax.set_xlabel('Risk_Level')
_ = ax.set_ylabel('# of house price by risk level')
_ = ax.set_title('Risk_level_house_price_mean')
_ = ax.set_grid(axis='y', linestyle='--', alpha=0.5)
_ = plt.xticks(rotation=0)
plt.show()

# 리스크 대비 중앙값 가격
fig, ax = plt.subplots(figsize=(6, 4))
median_RiskLevel.sort_index().plot(kind='bar', color='salmon', edgecolor='black')
_ = ax.set_xlabel('Risk_Level')
_ = ax.set_ylabel('# of house price by risk level')
_ = ax.set_title('Risk_level_house_price_median')
_ = ax.set_grid(axis='y', linestyle='--', alpha=0.5)
_ = plt.xticks(rotation=0)
plt.show()