import os
from dataloader import DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

import warnings
warnings.filterwarnings('ignore')

# os.chdir('./src')

# 1. 전처리 수행
def preprocess_data(ames):
    # 수치형 변수와 범주형 변수 분리
    numeric_cols = ames.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = ames.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if ames[col].isnull().sum() > 0:
            # 최빈값 직접 계산 후 할당
            mode_value = ames[col].mode()[0]
            ames[col].fillna(mode_value, inplace=True)
    
    numeric_data = ames[numeric_cols].copy()

    # 수치형 변수 결측치 대체 (by.회귀)
    mice_imputer = IterativeImputer(max_iter=10, random_state=42)
    ames[numeric_cols] = mice_imputer.fit_transform(numeric_data)
    
    # 범주형 변수 결측치 대체 (by.최빈값)
    for col in categorical_cols:
        if ames[col].isnull().sum() > 0:
            cat_imputer = SimpleImputer(strategy='most_frequent')
            ames[col] = cat_imputer.fit_transform(ames[[col]])
    
    # 이상치 탐색
    Q1 = ames['SalePrice'].quantile(0.25)
    Q3 = ames['SalePrice'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # 이상치 경계값으로 대체
    ames.loc[ames['SalePrice'] < lower_bound, 'SalePrice'] = lower_bound
    ames.loc[ames['SalePrice'] > upper_bound, 'SalePrice'] = upper_bound
    
    return ames

# 2. 위험 등급 설정
def create_risk_categories(ames):
    """외장재, 판매액, 내장재에 대한 위험 등급 설정"""
    
    # 종합 위험 점수 계산 (각 요소별 가중치 적용)
    ames['TotalRiskScore'] = (
        ames['Risk_RoofMatl'] * 0.3 +
        ames['Risk_Exterior1st'] * 0.3 +
        ames['Risk_Exterior2nd'] * 0.1 +
        ames['Risk_MasVnrType'] * 0.1 +
        ames['Risk_WoodDeckSF'] * 0.2
    )
    
    # 피해액 추정 (기본 공식: 위험점수 x 판매액 x 0.1)
    # 화재 시 건물 가치의 일정 비율이 손실된다고 가정
    # 
    ames['EstimatedDamage'] = ames['TotalRiskScore'] * ames['SalePrice'] * 0.1
    
    return ames

# 3. 표본 대표성 검증 (Z-test)
def sample_representation_test(ames, sample_indices):
    """표본(판매된 집)과 모집단(전체)의 평균 비교를 통한 대표성 검증"""
    # 표본 선택 (예시로 랜덤 선택)
    if sample_indices is None:
        sample = ames.sample(frac=0.3, random_state=42)
    else:
        sample = ames.iloc[sample_indices]
    
    population = ames
    
    # 주요 변수에 대한 Z-test 수행
    test_variables = ['SalePrice', 'GrLivArea', 'TotalBsmtSF', 'EstimatedDamage']
    test_results = {}
    
    for var in test_variables:
        # 모집단 통계
        pop_mean = population[var].mean()
        pop_std = population[var].std()
        pop_n = len(population)
        
        # 표본 통계
        sample_mean = sample[var].mean()
        sample_n = len(sample)
        
        # Z-test 계산
        z_score = (sample_mean - pop_mean) / (pop_std / np.sqrt(sample_n))
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))  # 양측 검정
        
        # 결과 저장
        test_results[var] = {
            'population_mean': pop_mean,
            'sample_mean': sample_mean,
            'z_score': z_score,
            'p_value': p_value,
            'significant': p_value < 0.05  # 유의수준 0.05
        }
    
    # 결과 출력
    print("\n===== 표본 대표성 검증 결과 =====")
    for var, result in test_results.items():
        print(f"\n변수: {var}")
        print(f"모집단 평균: {result['population_mean']:.2f}")
        print(f"표본 평균: {result['sample_mean']:.2f}")
        print(f"Z-score: {result['z_score']:.4f}")
        print(f"P-value: {result['p_value']:.4f}")
        if result['significant']:
            print("결론: 통계적으로 유의한 차이가 있음 (대표성 부족)")
        else:
            print("결론: 통계적으로 유의한 차이가 없음 (대표성 있음)")
    
    # 대표성 있는지 종합 판단
    has_representation = not any([result['significant'] for result in test_results.values()])
    
    return has_representation, test_results

# 4. 피해액 예측 모델 구축
def build_prediction_models(ames):
    """라쏘 및 릿지 회귀를 활용한 피해액 예측 모델 구축"""
    # 피해액 예측을 위한 특성 선택
    features = [
        'SalePrice', 'ExteriorRisk', 'InteriorRisk', 'PriceRisk',
        'OverallQual', 'GrLivArea', 'TotalBsmtSF', 'GarageCars',
        'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'LotArea'
    ]
    
    # 목표 변수
    target = 'EstimatedDamage'
    
    # 학습/테스트 데이터 분할
    X = ames[features]
    y = ames[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # 수치형 변수 전처리
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    # 전체 전처리
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features)
        ]
    )
    
    # Lasso 모델
    lasso_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', Lasso(random_state=42))
    ])
    
    # Ridge 모델
    ridge_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', Ridge(random_state=42))
    ])
    
    # 알파값 설정
    lasso_param_grid = {
        'regressor__alpha': [0.001, 0.01, 0.1, 1, 10, 100]
    }
    
    ridge_param_grid = {
        'regressor__alpha': [0.001, 0.01, 0.1, 1, 10, 100]
    }
    
    # K-폴드 교차 검증
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Lasso 모델 튜닝
    lasso_grid = GridSearchCV(
        lasso_pipeline, 
        lasso_param_grid, 
        cv=kfold, 
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    
    lasso_grid.fit(X_train, y_train)
    best_lasso = lasso_grid.best_estimator_
    
    # Ridge 모델 튜닝
    ridge_grid = GridSearchCV(
        ridge_pipeline, 
        ridge_param_grid, 
        cv=kfold, 
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    
    ridge_grid.fit(X_train, y_train)
    best_ridge = ridge_grid.best_estimator_
    
    # 모델 평가
    models = {
        'Lasso': best_lasso,
        'Ridge': best_ridge
    }
    
    results = {}
    
    for name, model in models.items():
        # 테스트 세트 예측
        y_pred = model.predict(X_test)
        
        # 평가 지표 계산
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'model': model
        }
        
        print(f"\n===== {name} 회귀 모델 평가 =====")
        print(f"MSE: {mse:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"R^2: {r2:.4f}")
        
        if name == 'Lasso':
            lasso_coefs = model.named_steps['regressor'].coef_
            lasso_feature_importance = pd.Series(lasso_coefs, index=features)
            print("\nLasso 특성 중요도 (0이면 제외된 특성):")
            print(lasso_feature_importance.sort_values(ascending=False))
    
    # 시각화: 실제 vs 예측값 비교
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, models['Lasso'].predict(X_test), alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('실제 피해액')
    plt.ylabel('예측 피해액')
    plt.title('Lasso: 실제 vs 예측 피해액')
    
    plt.subplot(1, 2, 2)
    plt.scatter(y_test, models['Ridge'].predict(X_test), alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('실제 피해액')
    plt.ylabel('예측 피해액')
    plt.title('Ridge: 실제 vs 예측 피해액')
    
    plt.tight_layout()
    plt.savefig('prediction_comparison.png')
    plt.close()
    
    return results

# 5. 최종 모델 선택 및 설명
def explain_model_selection(results):
    # 성능 비교 및 최종 모델 선택
    best_model_name = min(results, key=lambda x: results[x]['rmse'])
    best_model = results[best_model_name]
    
    print("\n===== 최종 모델 선택 =====")
    print(f"최적 모델: {best_model_name}")
    print(f"RMSE: {best_model['rmse']:.2f}")
    print(f"R^2: {best_model['r2']:.4f}")
    
    if best_model_name == 'Lasso':
        print("\nLasso 모델의 성능이 더 우수")
        print("1. 화재 피해액 예측에 일부 특성만이 중요한 영향을 미침")
        print("2. 불필요한 특성들이 제거되었을 때 예측 정확도가 향상됨")
    else:
        print("\nRidge 모델의 성능이 우수")
        print("1. 상관관계가 높음")
        print("2. 모든 특성이 예측에 기여함")
    
    return best_model_name, best_model


def main(ames, sample_indices=None):
    print("===== 화재 발생 시 예상 피해액 모델링 =====\n")
    
    # 1. 데이터 전처리
    print("\n1. 데이터 전처리 중...")
    ames = preprocess_data(ames)
    
    # 2. 위험 등급 설정
    print("\n2. 위험 등급 설정 중...")
    ames = create_risk_categories(ames)
    print("위험 등급 설정 완료!")
    print("- 외장재 위험 등급 (1-5): 낮을수록 안전")
    print("- 판매액 위험 등급 (1-5): 높을수록 피해액 큼")
    print("- 내장재 위험 등급 (1-5): 낮을수록 안전")
    
    # 3. 표본 대표성 검증
    print("\n3. 표본 대표성 검증 중...")
    has_representation, test_results = sample_representation_test(ames, sample_indices)
    
    if not has_representation:
        print("\n경고: 표본이 모집단을 대표하지 않음. 모델링 결과 신중한 해석 요망")
    else:
        print("\n표본이 모집단을 적절히 대표, 모델링 진행 가능")
    
    # 4. 피해액 예측 모델 구축
    print("\n4. 피해액 예측 모델 구축 중...")
    model_results = build_prediction_models(ames)
    
    # 5. 최종 모델 선택 및 설명
    print("\n5. 최종 모델 선택 및 설명...")
    best_model_name, best_model = explain_model_selection(model_results)
    
    return ames, model_results, best_model_name, best_model

if __name__ == "__main__":

    dataloader = DataLoader()
    dataset = dataloader.load_data()
    
    risk_columns = [c for c in dataset.columns if c.split('_')[0] == 'Risk']
    risk_columns
    
    dataset, model_results, best_model_name, best_model = main(dataset)
    
    print("\n===== 모델링 완료 =====")
    print(f"최종 선택된 모델: {best_model_name}")
    print(f"RMSE: {best_model['rmse']:.2f}")
    print(f"R^2: {best_model['r2']:.4f}")