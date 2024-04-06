# Ex.No: 07  AUTO REGRESSIVE MODEL
### AIM:
To Implementat an Auto Regressive Model using Python
### ALGORITHM:
1. Import necessary libraries
2. Read the CSV file into a DataFrame
3. Perform Augmented Dickey-Fuller test
4. Split the data into training and testing sets.Fit an AutoRegressive (AR) model with 13 lags
5. Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF)
6. Make predictions using the AR model.Compare the predictions with the test data
7. Calculate Mean Squared Error (MSE).Plot the test data and predictions.
### PROGRAM:

```
DEVELOPED BY: LOKESH R
REG NO: 212222240055
```


Import necessary libraries

```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error
```


Read the CSV file into a DataFrame

```
data = pd.read_csv("/content/Temperature.csv")  
data['date'] = pd.to_datetime(data['date'])
data.set_index('date', inplace=True)
```

Perform Augmented Dickey-Fuller test

```
result = adfuller(data['temp']) 
print('ADF Statistic:', result[0])
print('p-value:', result[1])
```

Split the data into training and testing sets

```
train_data = data.iloc[:int(0.8*len(data))]
test_data = data.iloc[int(0.8*len(data)):]

```

Fit an AutoRegressive (AR) model with 13 lags

```
lag_order = 13
model = AutoReg(train_data['temp'], lags=lag_order)
model_fit = model.fit()
```


Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF)

```
plot_acf(data['temp'])
plt.title('Autocorrelation Function (ACF)')
plt.show()
plot_pacf(data['temp'])
plt.title('Partial Autocorrelation Function (PACF)')
plt.show()
```

Make predictions using the AR model

```
predictions = model_fit.predict(start=len(train_data), end=len(train_data)+len(test_data)-1)

```

Compare the predictions with the test data

```
mse = mean_squared_error(test_data['temp'], predictions)
print('Mean Squared Error (MSE):', mse)
```


Plot the test data and predictions

```
plt.plot(test_data.index, test_data['temp'], label='Test Data')
plt.plot(test_data.index, predictions, label='Predictions')
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('AR Model Predictions vs Test Data')
plt.legend()
plt.show()
```


### OUTPUT:

GIVEN DATA

![image](https://github.com/LokeshRajamani/TSA_EXP7/assets/120544804/d5ca8495-647c-48eb-8a58-94a3b3534edb)



Augmented Dickey_Fuller Test

![image](https://github.com/LokeshRajamani/TSA_EXP7/assets/120544804/4308cd56-d16d-4948-95d7-e1de7f700095)



PACF - ACF

![image](https://github.com/LokeshRajamani/TSA_EXP7/assets/120544804/7e889c4d-c7d8-4a83-8d5b-832166bba336)


![image](https://github.com/LokeshRajamani/TSA_EXP7/assets/120544804/110d709a-433a-4821-878b-732121431eec)



Mean Squared Error

![image](https://github.com/LokeshRajamani/TSA_EXP7/assets/120544804/e18ba250-7b2f-4722-8450-1fd9ed27183b)




PREDICTION

![image](https://github.com/LokeshRajamani/TSA_EXP7/assets/120544804/89d9b04b-3f76-4d3b-b4c1-11e1d59db4e3)




### RESULT:
Thus we have successfully implemented the auto regression function using python.
