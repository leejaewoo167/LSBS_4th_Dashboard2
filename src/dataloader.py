import pandas as pd
import warnings

from utils import risk_material, merge_dict, risk_ranges
warnings.filterwarnings('ignore')

class DataLoader:
    def __init__(self):
        self.data = pd.read_csv('../data/ames.csv')
        
    def make_risk_point(self, 
                        data: pd.DataFrame):
        RoofMatl_materials = ['CompShg', 'Tar&Grv', 'WdShake', 'WdShngl', 'Metal', 'Roll', 'Membran']
        RoofMatl_risk_point = [2,2,5,4,1,3,3]
        RoofMatl_dict = {m:p for m,p in zip(RoofMatl_materials, RoofMatl_risk_point)}
        
        Exterior1st_materials = ['Wd Sdng', 'HdBoard', 'MetalSd', 'VinylSd', 'WdShing', 'Plywood',
            'Stucco', 'CemntBd', 'BrkFace', 'AsbShng', 'BrkComm', 'ImStucc',
            'AsphShn', 'CBlock', 'PreCast']
        Exterior1st_risk_point = [5,4,1,3,5,5,2,1,1,1,1,2,4,1,1]
        Exterior1st_dict = {m:p for m,p in zip(Exterior1st_materials, Exterior1st_risk_point)}
        
        Exterior2nd_materials = ['Wd Sdng', 'HdBoard', 'MetalSd', 'VinylSd', 'Wd Shng', 'Plywood',
            'Stucco', 'CmentBd', 'AsbShng', 'ImStucc', 'BrkFace', 'Brk Cmn',
            'CBlock', 'AsphShn', 'Stone', 'PreCast']
        Exterior2nd_risk_point = [5,4,1,3,5,5,2,1,1,2,1,1,1,4,1,1]
        Exterior2nd_dict = {m:p for m,p in zip(Exterior2nd_materials, Exterior2nd_risk_point)}
        
        MasVnrType_materials = ['BrkFace', 'Stone', 'BrkCmn']
        MasVnrType_risk_point = [1,1,1]
        MasVnrType_dict = {m:p for m,p in zip(MasVnrType_materials, MasVnrType_risk_point)}
        
        params = ['RoofMatl', 'Exterior1st', 'Exterior2nd','MasVnrType']
        
        # merged_dict
        merged_dict = merge_dict(left = RoofMatl_dict,
                                right = Exterior1st_dict)
        merged_dict = merge_dict(left = merged_dict,
                                right = Exterior2nd_dict)
        merged_dict = merge_dict(left = merged_dict,
                                right = MasVnrType_dict)
        
        # make derived parameters about risk of materials
        for param in params:
            data[f'Risk_{param}'] = data[param].apply(lambda x: risk_material(material = x, mat_risk = merged_dict)) 
        data['Risk_WoodDeckSF'] = data['WoodDeckSF'].apply(risk_ranges)
        
        data['Risk_Avg'] = (
            data['Risk_RoofMatl'] * 0.30 +
            data['Risk_Exterior1st'] * 0.30 +
            data['Risk_Exterior2nd'] * 0.10 +
            data['Risk_MasVnrType'] * 0.10 +
            data['Risk_WoodDeckSF'] * 0.2
        )
        data['Risk_Level'] = data['Risk_Avg'].round()

        return data

    def SF_calculator(self, data):
        # 유효한 데이터 필터링
        dataset = data[(data['LotArea'] > 0) & 
                        (data['TotalBsmtSF'] + data['1stFlrSF'] + data['2ndFlrSF'] > 0)]

        # # 총 건물 면적 계산
        # # TotalSF = TotalBsmtSF + 1stFlrSF + 2ndFlrSF
        # # → 지하 + 1층 + 2층을 합친 총 연면적 (평단가 계산 기준)
        dataset['TotalSF'] = dataset['TotalBsmtSF'] + dataset['1stFlrSF'] + dataset['2ndFlrSF']

        # # 가중 평균 기반 LandValue, BuildingValue 계산
        alpha = 0.6  # 땅과 건물의 상대적 중요도

        # # denominator = α × LotArea + (1 - α) × TotalSF
        # # → 전체 면적 중 땅과 건물이 차지하는 가중합 (가격 분배 기준)
        denominator = alpha * dataset['LotArea'] + (1 - alpha) * dataset['TotalSF']

        # # LandValue = (α × LotArea) / (denominator) × SalePrice
        # # → 전체 주택 가격 중 땅 면적이 차지하는 비중만큼을 땅값으로 분배
        dataset['LandValue'] = (alpha * dataset['LotArea']) / denominator * dataset['SalePrice']

        # # BuildingValue = SalePrice - LandValue
        # # → 전체 집값에서 땅값을 빼고 남은 것이 건물값 (즉, 피해 대상)
        dataset['BuildingValue'] = dataset['SalePrice'] - dataset['LandValue']

        # 건물 평단가 계산
        # BuildingPricePerTotalSF = BuildingValue / TotalSF
        # → 건물 1평당 단가 = 실제 화재 피해 추정 단가    
        dataset['BuildingPricePerTotalSF'] = (dataset['BuildingValue'] / dataset['LotArea'])* 35.5832  # (1평 = 3.305 m², 1m² = 35.5832$)

        return dataset
    
    def load_data(self):
        self.data['Date'] = pd.to_datetime(
            self.data['YrSold'].astype(str) + '-' + self.data['MoSold'].astype(str),
            format='%Y-%m'
        )
        
        self.data = self.make_risk_point(self.data)
        self.data = self.SF_calculator(self.data)
        
        return self.data
    