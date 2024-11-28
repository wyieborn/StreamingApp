from sklearn.preprocessing import StandardScaler
from sklearn.metrics import*
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import joblib


def train(numeric_df, data):
    
    ds = numeric_df.copy()
    nullRowsIndex=ds[ds['Danceability'].isna()].index
    ds=ds.drop(nullRowsIndex)
    ds['Stream']=data['Stream']

    ds2 = ds.copy()
    ds2=ds2.drop(index=ds2[ds2['Stream'].isna()].index)
    scaler=StandardScaler()
    stdDs2=pd.DataFrame(scaler.fit_transform(ds2),columns=ds2.columns)
    stdDs2X=stdDs2.drop(columns=['Stream'], axis = 1)
    stdDsStreamsY=stdDs2['Stream']#target column

    rnd_reg = RandomForestRegressor(n_estimators=40, n_jobs=-1, random_state=40)
    rnd_reg.fit(stdDs2X, stdDsStreamsY)
    # importances = rnd_reg.feature_importances_
    # index = np.argsort(importances)
    value_predictions=rnd_reg.predict(stdDs2X)
    r2=r2_score(stdDsStreamsY, value_predictions)
    n=len(numeric_df.index)
    p=numeric_df.shape[1]
    adj_r2=1-(1-r2)*(n-1)/(n-p-1)
    print('R2',r2,'.Adj R2:',adj_r2)
    
    joblib.dump(rnd_reg,"random_forest_model.pkl")
    
    return rnd_reg