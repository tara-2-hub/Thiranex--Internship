"""Thiranex Internship Task 3: Predictive Analytics Using Historical Data
Dependencies: pandas, numpy, scikit-learn
Run: python predictive_analytics.py
The included historical dataset is synthetic demonstration data.
"""
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
df=pd.read_csv("predictive_analytics_historical_data.csv",parse_dates=["Date"])
df["Month Number"]=df.Date.dt.month
df["Month Index"]=np.arange(len(df))
df["Month Sin"]=np.sin(2*np.pi*df["Month Number"]/12)
df["Month Cos"]=np.cos(2*np.pi*df["Month Number"]/12)
features=["Month Index","Month Number","Month Sin","Month Cos","Marketing Spend (₹)","Units Sold"]
cut=int(len(df)*.8); train,test=df.iloc[:cut],df.iloc[cut:]
model=LinearRegression().fit(train[features],train["Revenue (₹)"]); p=model.predict(test[features])
print("MAE:",round(mean_absolute_error(test["Revenue (₹)"],p),2))
print("RMSE:",round(mean_squared_error(test["Revenue (₹)"],p)**.5,2))
print("R2:",round(r2_score(test["Revenue (₹)"],p),3))
pd.DataFrame({"Date":test.Date,"Actual Revenue (₹)":test["Revenue (₹)"],"Predicted Revenue (₹)":p}).to_csv("predictive_model_test_predictions.csv",index=False)
model.fit(df[features],df["Revenue (₹)"])
dates=pd.date_range(df.Date.max()+pd.offsets.MonthBegin(1),periods=12,freq="MS"); idx=np.arange(len(df),len(df)+12)
f=pd.DataFrame({"Date":dates,"Month Number":dates.month,"Month Index":idx})
f["Month Sin"]=np.sin(2*np.pi*f["Month Number"]/12); f["Month Cos"]=np.cos(2*np.pi*f["Month Number"]/12)
f["Marketing Spend (₹)"]=900+22*idx; f["Units Sold"]=np.maximum(20,np.polyfit(np.arange(len(df)),df["Units Sold"],1)[0]*idx+np.polyfit(np.arange(len(df)),df["Units Sold"],1)[1]).round()
f["Forecast Revenue (₹)"]=np.maximum(0,model.predict(f[features])).round(2)
f[["Date","Marketing Spend (₹)","Units Sold","Forecast Revenue (₹)"]].to_csv("predictive_analytics_12_month_forecast.csv",index=False)
