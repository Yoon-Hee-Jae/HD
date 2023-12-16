# HD 선박 도착시간 예측 공모전

# 목차
  1. 공모전 개요
  2. EDA
  3. 데이터 전처리
  4. 모델 구성
  5. 결과

# 1. 공모전 개요

코로나19 이후 물류 정체로 인해 다수의 항만에서 선박 대기 시간이 길어지고, 이로 인한 물류 지연이 화두가 되고 있는 상황입니다. 

특히 전 세계 물동량의 85%를 차지하는 해운 물류 분야에서 항만 정체는 큰 문제로 인식되고 있는 상황입니다. 

본 대회에서는 접안(배를 육지에 대는 것;Berthing) 전에 선박이 해상에 정박(해상에 닻을 바다 밑바닥에 내려놓고 운항을 멈추는 것;Anchorage)하는 시간을 대기시간으로 정의하고, 선박의 제원 및 운항 정보를 활용하여 산출된 항차(voyage; 선박의 여정) 데이터를 활용하여 항만 內 선박의 대기 시간을 예측하는 AI 알고리즘을 개발을 제안합니다.

이를 통해 선박의 접안 시간 예측이 가능해지고, 선박의 대기시간을 줄임으로써 연료 절감 및 온실가스 감축효과를 기대할 수 있습니다.

# 2. EDA 및 데이터 전처리

## 2-1. 결측치 확인

```python

# 데이터프레임 전체 결측치 파악
total_missing = train.isna().sum()
print(total_missing)

```

<출력 결과>

![image](https://github.com/Yoon-Hee-Jae/HD/assets/140389762/b7ecc175-08e9-4aba-a4df-af8f61572671)


결측치가 절반 이상인 칼럼이 존재하는 것을 확인하였고 모두 기상관련 데이터임을 확인하였습니다. 
이 경우 다른 값으로 대체를 해주는 것보다 삭제를 하는 것이 좋겠다고 판단하였습니다.
결측치가 1개인 행의 경우 또한 발견하였습니다.



## 2-2 독립변수들간의 상관관계 파악

```python

# 데이터프레임에서 특정 열과 다른 열들을 선택
selected_columns = ['ARI_CO', 'ARI_PO', 'SHIP_TYPE_CATEGORY', 'DIST', 'ATA', 'ID',
       'BREADTH', 'BUILT', 'DEADWEIGHT', 'DEPTH', 'DRAUGHT', 'GT', 'LENGTH',
       'SHIPMANAGER', 'FLAG', 'ATA_LT', 'PORT_SIZE','CI_HOUR']  # 특정 열과 비교할 다른 열들

# 선택한 열들로 새로운 데이터프레임 생성
selected_data = filtered_train[selected_columns]

# 상관 행렬 계산
correlation_matrix = selected_data.corr()

# 상관 히트맵 그리기
plt.figure(figsize=(10, 8))  # 그래프 크기 설정
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)  # 상관 히트맵 그리기
plt.title(f'상관 히트맵 - {selected_column}과 다른 열들 간 상관성')  # 그래프 제목 설정
plt.show()

```

<출력 결과>

![image](https://github.com/Yoon-Hee-Jae/HD/assets/140389762/a1c3811b-3db8-4f16-8f82-c711d3464316)

특정독립변수들 사이에 높은 상관성이 있는 것을 확인할 수 있습니다.
독립변수들간의 상관성이 높을 경우 다중공선성을 의심해볼 수 있습니다.

## 2-3 다중공선성 확인을 위해 VIF값 계산

```python


from statsmodels.stats.outliers_influence import variance_inflation_factor

# 데이터프레임에서 독립 변수들을 선택
X = filtered_train.drop(columns = ['CI_HOUR','ARI_CO','ARI_PO','SHIP_TYPE_CATEGORY','ATA','ID','SHIPMANAGER','FLAG'])

# VIF 계산
vif_data = pd.DataFrame()
vif_data["Variable"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

# 결과 출력
print(vif_data)

```

<결과 출력>

![image](https://github.com/Yoon-Hee-Jae/HD/assets/140389762/c076a298-3e52-4e5b-8bae-65c9b6c0e374)

VIF값을 확인해본 결과 BREADTH/DEADWEIGHT/DEPTH/DRAUGHT/GT/LENGTH 독립변수들이 서로 다중공선성이 있음을 확인할 수 있었습니다.
(VIF값이 10을 넘을 경우 다중공선성이 있다고 판단)

## 2-4 계절성 유뮤 판단

ATA의 열의 경우 입항시간을 의미합니다. 따라서 ATA을 기준으로 다른 열을 살펴본다면 계절성의 유무를 판단할 수 있습니다.

```python

train['ATA'] = pd.to_datetime(train['ATA'])
df_ATAdate = train.copy()

df_ATAdate.drop(columns = 'U_WIND', inplace = True)
df_ATAdate.drop(columns = 'V_WIND', inplace = True)
df_ATAdate.drop(columns = 'AIR_TEMPERATURE', inplace = True)
df_ATAdate.drop(columns = 'BN', inplace = True)
df_ATAdate['ATA'] = df_ATAdate['ATA'].dt.floor('D')
#df_ATAdate['ATA'].head()

# 계절성 시각화
plt.figure(figsize=(10, 6))

for column in df_ATAdate.columns:
    sns.lineplot(data = df_ATAdate.sort_values(by = 'ATA'), x = 'ATA', y = column)
    plt.show()
     

```

<출력 예시>

![image](https://github.com/Yoon-Hee-Jae/HD/assets/140389762/0eae9162-46d9-4630-90ba-3b57a36721ef)

대부분의 열이 위와 같은 그래프를 가진 것을 확인할 수 있었습니다.
이를 통해서 계절성은 없다고 판단을 하였습니다.

## 2-5 DIST 칼럼 분석
DIST 칼럼의 경우 정박지와 접안지 사이의 거리를 의미합니다.
따라서 DIST의 값이 0인 경우 CI_HOUR는 0일 것이라는 가정을 하였습니다.
실제로 DIST가 0인 값을 확인해보니 15만개의 데이터 중 28개만이 0이 아닌 값을 가졌습니다.
따라서 이와 같은 값은 이상치로 판단하여 제거하도록 하였습니다.

```python

# DIST가 0인 행만 필터링합니다.
dist_0_data = train[train['DIST'] == 0]

# CI_HOUR 컬럼의 값과 그 구성을 확인합니다.
counts = dist_0_data['CI_HOUR'].value_counts()

print(counts)

```

<출력 결과>

![image](https://github.com/Yoon-Hee-Jae/HD/assets/140389762/30375974-3928-4f80-94bd-cd6b3a71f73f)


## 2-6 이상치 확인
각 열의 분포도를 찍은 그래프를 확인한 결과 DIST와 종속변수인 CI_HOUR에 이상치가 존재함을 확인할 수 있었습니다.

```python

columns = ['DIST', 'BREADTH', 'BUILT', 'DEADWEIGHT', 'DEPTH', 'DRAUGHT', 'GT', 'LENGTH', 'U_WIND', 'V_WIND', 'AIR_TEMPERATURE', 'BN', 'ATA_LT', 'PORT_SIZE', 'CI_HOUR']

plt.figure(figsize=(10, 6), dpi=100)  # 크기와 해상도 설정

# 밀도 그림 그리기
for column in columns:
    sns.kdeplot(train[column], shade=True)  # 밀도 그림 그리기
    plt.title('distplot')  # 그래프 제목 설정
    plt.xlabel(column)  # x축 레이블 설정
    plt.ylabel('frequency')  # y축 레이블 설정
    plt.show()  # 그래프 표시
     

```

<출력 예시>

- DIST

![image](https://github.com/Yoon-Hee-Jae/HD/assets/140389762/cb75207c-ece6-4fc9-9b09-249c750313a6)


- CI_HOUR

![image](https://github.com/Yoon-Hee-Jae/HD/assets/140389762/a7c49b04-770c-487d-8480-1df92f07bf59)


# 3. 데이터 전처리
EDA를 한 결과를 기준으로 데이터 전처리를 진행하였습니다
데이터전처리 내용은 다음과 같습니다.

- ATA 열을 year, month, day, hour, minute, weekday 으로 나누어서 새로운 열들로 대체해준 뒤 ATA 열은 제거하였습니다.

```python

# datetime 컬럼 처리
train['ATA'] = pd.to_datetime(train['ATA'])
test['ATA'] = pd.to_datetime(test['ATA'])

# datetime을 여러 파생 변수로 변환
for df in [train, test]:
    df['year'] = df['ATA'].dt.year
    df['month'] = df['ATA'].dt.month
    df['day'] = df['ATA'].dt.day
    df['hour'] = df['ATA'].dt.hour
    df['minute'] = df['ATA'].dt.minute
    df['weekday'] = df['ATA'].dt.weekday


# 2. datetime 컬럼 제거
train.drop(columns='ATA', inplace=True)
test.drop(columns='ATA', inplace=True)

```python

- Categorical 열인 'ARI_CO', 'ARI_PO', 'SHIP_TYPE_CATEGORY', 'ID', 'SHIPMANAGER', 'FLAG'의 경우 LabelEncoder()를 사용하여 숫자로 변환해주었습니다.

```

categorical_features = ['ARI_CO', 'ARI_PO', 'SHIP_TYPE_CATEGORY', 'ID', 'SHIPMANAGER', 'FLAG']
encoders = {}

for feature in tqdm(categorical_features, desc="Encoding features"):
    le = LabelEncoder()
    train[feature] = le.fit_transform(train[feature].astype(str))
    le_classes_set = set(le.classes_)
    test[feature] = test[feature].map(lambda s: '-1' if s not in le_classes_set else s)
    le_classes = le.classes_.tolist()
    bisect.insort_left(le_classes, '-1')
    le.classes_ = np.array(le_classes)
    test[feature] = le.transform(test[feature].astype(str))
    encoders[feature] = le

'''

- DIST가 0일 경우 CI_HOUR 모두 0으로 변경하였습니다.

```python

# 4. DIST가 0이면, CI_HOUR 모두 0으로 변경
train.loc[train['DIST'] == 0, 'CI_HOUR'] = 0


```

- 기상 관련 결측치 많은 열의 경우 제거하였습니다.
  knn를 활용하여 결측치의 값들을 대체해보았지만 실제로 모델에 입력하여 성능을 테스트해본 결과 유의미한 상승은 없었기 때문에 제외하도록 하였습니다.

```python
# 열 삭제
train.drop(columns = 'U_WIND', inplace = True)
train.drop(columns = 'V_WIND', inplace = True)
train.drop(columns = 'AIR_TEMPERATURE', inplace = True)
train.drop(columns = 'BN', inplace = True)

test.drop(columns = 'U_WIND', inplace = True)
test.drop(columns = 'V_WIND', inplace = True)
test.drop(columns = 'AIR_TEMPERATURE', inplace = True)
test.drop(columns = 'BN', inplace = True)

# knn 대체

knn_imputer = KNNImputer(n_neighbors=10)

# K-NN 대체를 적용할 열을 지정
columns_to_impute = ['U_WIND', 'V_WIND', 'AIR_TEMPERATURE', 'BN']

# train 데이터프레임에서 결측치를 K-NN으로 대체
filtered_train[columns_to_impute] = knn_imputer.fit_transform(filtered_train[columns_to_impute])

filtered_train = pd.DataFrame(filtered_train, columns=filtered_train.columns)
#filtered_train.to_csv('train_imputed.csv', index = False)


```

- 다중공선성 문제 해결을 위해 Length열을 제외한 다른 열들은 제거하였습니다. (BREADTH, DEADWEIGHT, DEPTH, DRAUGHT, GT)

```python

for df in [train, test]:
    df.drop(columns = 'BREADTH', inplace = True)
    df.drop(columns = 'DEADWEIGHT', inplace = True)
    df.drop(columns = 'DEPTH', inplace = True)
    df.drop(columns = 'DRAUGHT', inplace = True)
    df.drop(columns = 'GT', inplace = True)

```

- CI_HOUR과 DIST 열에서 표준편차 + 2 * 표준편차 이상이 되는 부분 이상치라고 판단 후 제거하였습니다.

```python

# 'CI_HOUR' 열의 평균과 표준 편차 계산 (DIST도 동일한 코드 사용)
mean = train['CI_HOUR'].mean()
std = train['CI_HOUR'].std()

# 이상치 경계 설정 (예: 평균에서 2배 표준 편차를 벗어나는 값)
lower_bound = mean - 2 * std
upper_bound = mean + 2 * std

# 이상치를 제거하고 정상 범위의 데이터만 남김
#filtered_train = train[(train['CI_HOUR'] >= lower_bound) & (train['CI_HOUR'] <= upper_bound)]
train = train[(train['CI_HOUR'] >= lower_bound) & (train['CI_HOUR'] <= upper_bound)].reset_index(drop=True)

```

- 시간(hour)는 cyclical encoding하여 변수를 추가하였습니다.(sin time & cos time)
  해당 작업을 추가한 이유는 시간의 경우 순환성을 가지기 때문입니다. 예를 들어 23시와 다음날 01시는 실제로는 두시간 차이밖에 나지않지만 22만큼 차이가 난다고 판단을 할 수 있기 때문입니다.
  따라서 cyclical encoding을 통하여 해당 문제를 완화시키도록 하였습니다.

```python

for df in [train, test]:
  # 시간
  df['sin_time_hour'] = np.sin(2*np.pi*train.hour/24)
  df['cos_time_hour'] = np.cos(2*np.pi*train.hour/24)
  # 월
  df['sin_time_month'] = np.sin(2*np.pi*train.month/12)
  df['cos_time_month'] = np.cos(2*np.pi*train.month/12)
  # weekday
  df['sin_time_weekday'] = np.sin(2*np.pi*train.weekday/7)
  df['cos_time_weekday'] = np.cos(2*np.pi*train.weekday/7)
  # minute
  df['sin_time_minute'] = np.sin(2*np.pi*train.minute/60)
  df['cos_time_minute'] = np.cos(2*np.pi*train.minute/60)

```

- 주말인 경우와 아닌 경우를 알 수 있는 weekday열을 새로 추가하였습니다.

```python

train['holiday'] = train.apply(lambda x : 0 if x['day']<5 else 1, axis = 1)
test['holiday'] = test.apply(lambda x : 0 if x['day']<5 else 1, axis = 1)

```

# 4. 모델 구성

## 4-1 DNN 모델 사용

- 라이브러리 import

```python

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

import lightgbm as lgb
import bisect
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold

```

- 정규화 진행

```python

# 정규화 진행
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Min-Max 스케일러 초기화 및 스케일링 적용
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

```

- 모델 구성

```python


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 모델
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(19,)),  # 입력 레이어
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)  # 출력 레이어 (회귀 모델이므로 하나의 출력 뉴런)
])

# 모델 컴파일
model.compile(optimizer='adam', loss='mean_absolute_error')

# 모델 학습
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping])
# model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.2)

# 모델 평가
loss = model.evaluate(X_train_scaled, y_train)

# 예측
y_pred = model.predict(X_test_scaled)

```


## 4-2 Catboost 모델 사용

```python

from catboost import CatBoostRegressor

def train_and_evaluate(model, model_name, X_train, y_train, cat_features= ['ARI_CO', 'ARI_PO', 'SHIP_TYPE_CATEGORY', 'ID', 'SHIPMANAGER', 'FLAG']):
    print(f'Model Tune for {model_name}.')
    model.fit(X_train, y_train, cat_features=cat_features)

    feature_importances = model.feature_importances_
    sorted_idx = feature_importances.argsort()

    plt.figure(figsize=(10, len(X_train.columns)))
    plt.title(f"Feature Importances ({model_name})")
    plt.barh(range(X_train.shape[1]), feature_importances[sorted_idx], align='center')
    plt.yticks(range(X_train.shape[1]), X_train.columns[sorted_idx])
    plt.xlabel('Importance')
    plt.show()

    return model, feature_importances

X_train = train.drop(columns='CI_HOUR')
y_train = train['CI_HOUR']

catboost_model = CatBoostRegressor()
catboost_model, catboost_feature_importances = train_and_evaluate(catboost_model, 'CatBoost', X_train, y_train)

```

<feature importance 확인>

![image](https://github.com/Yoon-Hee-Jae/HD/assets/140389762/75823961-7797-49e2-9bd2-f2c2c9e3e933)

![image](https://github.com/Yoon-Hee-Jae/HD/assets/140389762/d3131854-db4e-4f30-a25b-20895dee0029)


feature importance 확인 결과 중요도가 낮은 열의 경우 삭제를 하기로 하였습니다.
최적의 삭제기준을 찾기 위해서 다양한 threshold 값들을 적용해보았습니다.
이를 위해 k-fold 교차 검증을 실시하였고 각 폴드의 모델로부터 얻은 예측값을 평균하여 최종 앙상블 예측을 생성하였습니다.
이때 평가기준은 실제값과 예측값간의 MAE를 기준으로 하였습니다.

## Grid search 
최적의 threshold를 기준으로 그리드서치를 진행하였습니다.
그리드서치를 사용하여 catboost모델에서 최적의 파라미터를 찾을 수 있도록 하였습니다.

```python

# 그리드 서치

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_absolute_error

# 가장 성능 좋았던 Threshold 정의
threshold = 2.5

best_mae = float('inf')
best_threshold = None

low_importance_features = X_train.columns[catboost_feature_importances < threshold]
X_train_reduced = X_train.drop(columns=low_importance_features)
X_test_reduced = test.drop(columns=low_importance_features)




# GridSearchCV에 사용할 하이퍼파라미터 그리드 정의
param_grid = {
    'learning_rate': [0.1,0.2,0.3,0.4],
    'depth': [10,11,12,13,14,15],
    'iterations': [100,200,300,400,500]
}

# CatBoost 모델 초기화
new_features = ['ARI_CO', 'ARI_PO', 'SHIP_TYPE_CATEGORY']
catboost_model = CatBoostRegressor(cat_features=new_features)

# GridSearchCV를 사용하여 하이퍼파라미터 튜닝
grid_search = GridSearchCV(catboost_model, param_grid, scoring='neg_mean_absolute_error', cv=5, n_jobs=-1)
grid_search.fit(X_train_reduced, y_train)

# 최적의 하이퍼파라미터와 MAE 출력
best_params = grid_search.best_params_
best_mae = -grid_search.best_score_
print('--- 최적 하이퍼 파라미터 ---')
print("Best Hyperparameters:", best_params)
print("Best Validation MAE:", best_mae)

# 최적의 하이퍼파라미터로 모델 초기화
best_catboost_model = CatBoostRegressor(**best_params)

# 5-Fold 설정
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 각 fold의 모델로부터의 예측을 저장할 리스트와 MAE 점수 리스트
ensemble_predictions = []
scores = []

for train_idx, val_idx in kf.split(X_train_reduced):
    X_t, X_val = X_train_reduced.iloc[train_idx], X_train_reduced.iloc[val_idx]
    y_t, y_val = y_train[train_idx], y_train[val_idx]

    # CatBoost 모델 학습
    best_catboost_model.fit(X_t, y_t,cat_features=new_features)

    # 각 모델로부터 Validation set에 대한 예측을 평균내어 앙상블 예측 생성
    val_pred = best_catboost_model.predict(X_val)

    # Validation set에 대한 대회 평가 산식 계산 후 저장
    scores.append(mean_absolute_error(y_val, val_pred))

    # test 데이터셋에 대한 예측 수행 후 저장
    catboost_pred = best_catboost_model.predict(X_test_reduced)
    catboost_pred = np.where(catboost_pred < 0, 0, catboost_pred)

    ensemble_predictions.append(catboost_pred)

# K-fold 모든 예측의 평균을 계산하여 fold별 모델들의 앙상블 예측 생성
final_predictions = np.mean(ensemble_predictions, axis=0)

# 각 fold에서의 Validation Metric Score와 전체 평균 Validation Metric Score 출력
print("Validation MAE scores for each fold:", scores)
print("Validation MAE:", np.mean(scores))

```



















