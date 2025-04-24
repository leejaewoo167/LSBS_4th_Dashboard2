import os
import shutil
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import scipy.stats as st
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import plotly.express as px
import plotly.graph_objects as go
import json

warnings.filterwarnings('ignore')


# 주택 가격 분류
# ---------------------------
# 📂 데이터 로딩
# ---------------------------
# DataLoader 사용
from dataloader import DataLoader
dataloader = DataLoader()
dataset = dataloader.load_data()

# ---------------------------
# 🚒 소방서 위치 데이터
# ---------------------------
fire_stations = pd.DataFrame({
    'Name': ['Station 1', 'Station 2', 'Station 3'],
    'Address': [
        '1300 Burnett Ave, Ames, IA 50010',
        '132 Welch Ave, Ames, IA 50014',
        '2400 S Duff Ave, Ames, IA 50010'
    ],
    'Latitude': [42.034862, 42.021596, 42.001115],
    'Longitude': [-93.615031, -93.649759, -93.609166]
})

# ---------------------------
# 💰 가격 등급 분류 (5단계)
# ---------------------------
price_by_neigh = dataset.groupby('Neighborhood')['SalePrice'].mean()
q20 = price_by_neigh.quantile(0.20)
q40 = price_by_neigh.quantile(0.40)
q60 = price_by_neigh.quantile(0.60)
q80 = price_by_neigh.quantile(0.80)

def classify_price_grade(neighborhood):
    price = price_by_neigh[neighborhood]
    if price <= q20:
        return 'Very Low'
    elif price <= q40:
        return 'Low'
    elif price <= q60:
        return 'Normal'
    elif price <= q80:
        return 'High'
    else:
        return 'Very High'

dataset['PriceGrade'] = dataset['Neighborhood'].apply(classify_price_grade)

# ---------------------------
# 🎨 색상 매핑
# ---------------------------
color_map = {
    'Very Low': 'indigo',
    'Low': 'purple',
    'Normal': 'gray',
    'High': 'blue',
    'Very High': 'navy'
}

# ---------------------------
# 🧭 지도 설정
# ---------------------------
layout_mapbox = dict(
    mapbox=dict(
        style="open-street-map",
        center=dict(lat=42.034534, lon=-93.620369),
        zoom=11
    ),
    margin={"r": 0, "t": 40, "l": 0, "b": 0},
    title='Ames 시 거래가격 & 소방서 위치'
)

# ---------------------------
# 🏡 주택 마커 (등급별 분리)
# ---------------------------
home_traces = []
for grade, color in color_map.items():
    subset = dataset[dataset['PriceGrade'] == grade]
    trace = go.Scattermapbox(
        lat=subset['Latitude'],
        lon=subset['Longitude'],
        mode='markers',
        marker=dict(size=7, color=color, opacity=0.6),
        text='$' + subset['SalePrice'].astype(int).astype(str) +
             '<br>평단가:$',
        name=f"{grade} Area"
    )
    home_traces.append(trace)


# ---------------------------
# 🚒 소방서 마커
# ---------------------------
fire_trace = go.Scattermapbox(
    lat=fire_stations['Latitude'],
    lon=fire_stations['Longitude'],
    mode='markers+text',
    marker=dict(size=12, color='red'),
    text=fire_stations['Name'],
    name='소방서',
    textposition='top right'
)

# ---------------------------
# 🧱 GeoJSON 경계선
# ---------------------------
with open('../data/ames_boundary.geojson') as f:
    geojson = json.load(f)

# 지도 객체 생성
fig = go.Figure(data=home_traces + [fire_trace], layout=layout_mapbox)

# GeoJSON 레이어 추가
fig.update_layout(
    mapbox_layers=[
        {
            "source": {
                "type": "FeatureCollection",
                "features": [geojson] if geojson["type"] != "FeatureCollection" else geojson["features"]
            },
            "type": "line",
            "color": "black",
            "line": {"width": 2}
        }
    ]
)

# 지도 출력
fig.show()


import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import json
import warnings
warnings.filterwarnings('ignore')

# 📂 데이터 로딩 (경로 수정 가능)
from dataloader import DataLoader
dataloader = DataLoader()
dataset = dataloader.load_data()

# ✅ 유효한 데이터 필터링
dataset = dataset[(dataset['LotArea'] > 0) & 
                  (dataset['TotalBsmtSF'] + dataset['1stFlrSF'] + dataset['2ndFlrSF'] > 0)]

# ✅ 총 건물 면적 계산
# TotalSF = TotalBsmtSF + 1stFlrSF + 2ndFlrSF
# → 지하 + 1층 + 2층을 합친 총 연면적 (평단가 계산 기준)
dataset['TotalSF'] = dataset['TotalBsmtSF'] + dataset['1stFlrSF'] + dataset['2ndFlrSF']

# ✅ 가중 평균 기반 LandValue, BuildingValue 계산
alpha = 0.4  # 땅과 건물의 상대적 중요도

# denominator = α × LotArea + (1 - α) × TotalSF
# → 전체 면적 중 땅과 건물이 차지하는 가중합 (가격 분배 기준)
denominator = alpha * dataset['LotArea'] + (1 - alpha) * dataset['TotalSF']

# LandValue = (α × LotArea) / (denominator) × SalePrice
# → 전체 주택 가격 중 땅 면적이 차지하는 비중만큼을 땅값으로 분배
dataset['LandValue'] = (alpha * dataset['LotArea']) / denominator * dataset['SalePrice']

# BuildingValue = SalePrice - LandValue
# → 전체 집값에서 땅값을 빼고 남은 것이 건물값 (즉, 피해 대상)
dataset['BuildingValue'] = dataset['SalePrice'] - dataset['LandValue']

# ✅ 건물 평단가 계산
# BuildingPricePerTotalSF = BuildingValue / TotalSF
# → 건물 1평당 단가 = 실제 화재 피해 추정 단가    
dataset['BuildingPricePerTotalSF'] = (dataset['BuildingValue'] / dataset['TotalSF'])* 35.5832  # (1평 = 3.305 m², 1m² = 35.5832$)

# ✅ 등급화 (5분위)
q20 = dataset['BuildingPricePerTotalSF'].quantile(0.20)
q40 = dataset['BuildingPricePerTotalSF'].quantile(0.40)
q60 = dataset['BuildingPricePerTotalSF'].quantile(0.60)
q80 = dataset['BuildingPricePerTotalSF'].quantile(0.80)

def classify_price_grade(x):
    if x <= q20:
        return 'Very Low'
    elif x <= q40:
        return 'Low'
    elif x <= q60:
        return 'Normal'
    elif x <= q80:
        return 'High'
    else:
        return 'Very High'

dataset['PriceGrade'] = dataset['BuildingPricePerTotalSF'].apply(classify_price_grade)

# ✅ 색상 매핑
color_map = {
    'Very Low': 'indigo',
    'Low': 'purple',
    'Normal': 'gray',
    'High': 'blue',
    'Very High': 'navy'
}

# ✅ 소방서 위치 데이터
fire_stations = pd.DataFrame({
    'Name': ['Station 1', 'Station 2', 'Station 3'],
    'Latitude': [42.034862, 42.021596, 42.001115],
    'Longitude': [-93.615031, -93.649759, -93.609166]
})

# ✅ 지도 설정
layout_mapbox = dict(
    mapbox=dict(
        style="open-street-map",
        center=dict(lat=42.034534, lon=-93.620369),
        zoom=11
    ),
    margin={"r": 0, "t": 40, "l": 0, "b": 0},
    title='Ames 시 건물 피해 단가 기준 주택 등급 & 소방서 위치'
)

# ✅ 주택 마커
home_traces = []
for grade, color in color_map.items():
    subset = dataset[dataset['PriceGrade'] == grade]
    trace = go.Scattermapbox(
        lat=subset['Latitude'],
        lon=subset['Longitude'],
        mode='markers',
        marker=dict(size=7, color=color, opacity=0.6),
        text='총가:$' + subset['SalePrice'].astype(int).astype(str) +
             '<br>건물평단가:$' + subset['BuildingPricePerTotalSF'].round(1).astype(str),
        name=f"{grade} Area"
    )
    home_traces.append(trace)

# ✅ 소방서 마커
fire_trace = go.Scattermapbox(
    lat=fire_stations['Latitude'],
    lon=fire_stations['Longitude'],
    mode='markers+text',
    marker=dict(size=12, color='red'),
    text=fire_stations['Name'],
    name='소방서',
    textposition='top right'
)

# ✅ GeoJSON 경계선 로딩
with open('../data/ames_boundary.geojson') as f:
    geojson = json.load(f)

# ✅ 지도 객체 생성
fig = go.Figure(data=home_traces + [fire_trace], layout=layout_mapbox)
fig.update_layout(
    mapbox_layers=[{
        "source": {
            "type": "FeatureCollection",
            "features": [geojson] if geojson["type"] != "FeatureCollection" else geojson["features"]
        },
        "type": "line",
        "color": "black",
        "line": {"width": 2}
    }]
)

# ✅ 지도 출력
fig.show()