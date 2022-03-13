# -*- coding: utf-8 -*-
"""
Created on Fri May 21 12:48:35 2021

@author: ESThomasSa
"""

import numpy as np
import pandas as pd
from fbprophet import Prophet
import matplotlib.pyplot as plt

##=======================PARAMS============================
path = 'C:\\Users\\ESThomasSa\\Desktop\\My_Files'
variables = ['num_fte_rolling', 'in_pipeline_rolling']


##=======================FUNCTIONS============================

## load and merge the data
def LoadSOs(path):
    dt_1 = pd.read_excel(path + '\Demand_forecast\\confidence_levels.xlsx')
    dt_2 = pd.read_excel(path + '\Demand_forecast\\ASL CRM Extraction.xlsx')
    dt_3 = dt_1[['SOID', 'ConfidenceLevelForOppProsLeadToSign', 'Planned SO Won Date', 'ConfidenceLevelToDeliverOnTime']].merge(dt_2, how='left', on='SOID')
    
    dt_3['high_confidence'] = np.where(dt_3['ConfidenceLevelForOppProsLeadToSign'] == 'H', 1, 0)
    dt_3['mid_high_confidence'] = np.where(dt_3['ConfidenceLevelForOppProsLeadToSign'] == 'M-H', 1, 0)
    dt_3['low_mid_confidence'] = np.where(dt_3['ConfidenceLevelForOppProsLeadToSign'] == 'L-M', 1, 0)
    dt_3['low_confidence'] = np.where(dt_3['ConfidenceLevelForOppProsLeadToSign'] == 'L', 1, 0)
    return dt_3

## create measures for service order won date
def CreateTable(dt):
    df_list = []
    for date in dt['Service Order Won Date'].unique():
        temp = dt.loc[dt['Service Order Won Date'] == date].copy()
        revenue = temp['Service Order Amount'].sum()
        high_conf = temp['high_confidence'].sum()
        mid_high_conf = temp['mid_high_confidence'].sum()
        low_mid_conf = temp['low_mid_confidence'].sum()
        low_conf = temp['low_confidence'].sum()
        pipeline = temp['Planned Start Date'].count()
        num_fte = temp['Service Delivery Manager'].nunique()
        DF = pd.DataFrame({'date':date,
                          'revenue':[revenue],
                          'num_fte':[num_fte],
                          'high_conf_to_deliver':[high_conf],
                          'mid_high_conf_to_deliver':[mid_high_conf],
                          'low_mid_conf_to_deliver':[low_mid_conf],
                          'low_conf_to_deliver':[low_conf],
                          'in_pipeline':[pipeline]
                          })
        df_list.append(DF)
    return pd.concat(df_list)

## create rolling averages
def feature_engineering(masterDT):
    calendar = pd.DataFrame({'date': pd.date_range(min(masterDT['date']), max(masterDT['date']))}) 
    new_masterDT = calendar.merge(masterDT, how='left', on='date')
    iter_cols = [x for x in list(masterDT.columns) if x not in ('date')]
    for col in new_masterDT[iter_cols]:
        new_masterDT[col].fillna(0, inplace=True)
        new_masterDT[col + '_rolling'] = new_masterDT[col].rolling(30).mean()
        new_masterDT[col + '_rolling'] = new_masterDT[col + '_rolling'].interpolate(method='bfill')
    return new_masterDT

## create future values for the regressor variables
def future_values(new_masterDT, variables):
    df_list = []
    for var in variables:
        temp = new_masterDT.copy()
        temp.rename(columns={'date':'ds', str(var):'y'}, inplace=True)
#        train_data = temp.loc[temp['ds'].dt.year <= 2020]
        model = Prophet(yearly_seasonality=True, daily_seasonality=False)
        model.fit(temp)
        future = model.make_future_dataframe(365)
        forecast = model.predict(future)
        preds = forecast['yhat']
        df =  pd.DataFrame({str(var) + '_pred':preds})
        df_list.append(df)
    features_df = pd.concat(df_list, axis=1)
    featuresDS = pd.concat([future, features_df], axis=1)
    final_result = temp[['ds', 'revenue_rolling']].merge(featuresDS, how='right', on='ds')
    final_result['month'] = final_result['ds'].dt.month
    return final_result

## predict revenue against predicted regressor values
def predict(finalDT):
    df = finalDT.copy()
    df.rename(columns={'date':'ds', 'revenue_rolling':'y'}, inplace=True)

    train_data = df.loc[df['y'].notnull(), :]
    
    model = Prophet()
    model.add_regressor('num_fte_rolling_pred')
    model.add_regressor('in_pipeline_rolling_pred')
    model.add_regressor('month')
    model.fit(train_data)
    forecast = model.predict(df)
    
    plt.plot(forecast['ds'], df['y'])
    plt.plot(forecast['ds'], forecast['yhat'])
    plt.legend(['y', 'yhat'])
    plt.xticks(rotation=45)
    return forecast

## aggregate by month for monthly revenue
def aggregate(forecast):
    forecast['date'] = forecast['ds'].dt.year.astype(str) + '-' + forecast['ds'].dt.month.astype(str) + '-01'
    forecast['date'] = pd.to_datetime(forecast['date']).dt.date
    revenueDF = forecast.groupby('date').sum().round(2)
    revenueDF['yearMonth'] = pd.to_datetime(revenueDF['yearMonth'])
#    revenueDF.to_excel(path + '\\Demand_forecast\\prophet_predictions_.xlsx', index=False)
    return revenueDF


##=======================MAIN============================
dt = LoadSOs(path)
masterDT = CreateTable(dt)
new_masterDT = feature_engineering(masterDT)
finalDT = future_values(new_masterDT, variables)

forecast = predict(finalDT)
revenues = aggregate(forecast)
