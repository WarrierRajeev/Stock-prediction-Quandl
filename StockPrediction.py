import pandas as pd
import quandl, math, datetime
import numpy as np
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

style.use('ggplot')

df= quandl.get('WIKI/GOOGL')
''' this is to get input from the quandl website on certain details of stocks on the market
    adj open is the price at which the market opens and adj close is the price at which the market closes
    We need to use linear regression to use some of these details as features and find a label.

'''

# This selects the columns that we need
df= df[['Adj. Open' , 'Adj. High',   'Adj. Low' , 'Adj. Close', 'Adj. Volume']] 

# We are creating features here namely percentage change and high-low difference %
df['HL_PCT']=(df['Adj. High']-df['Adj. Low'])/ df['Adj. Low'] *100.0            
df['PCT_change']=(df['Adj. Close']-df['Adj. Open'])/ df['Adj. Open'] *100.0

 #This is the final list of features we will work with
df=df[['Adj. Close' , 'HL_PCT',   'PCT_change' , 'Adj. Volume']]               

#forecast_column is what we are predicting using the algorithm, so we are going to predict adj. close value
forecast_col= 'Adj. Close'          

#There are a lot of columns with NA values, we are assigning a -99999 value to it
df.fillna(-99999, inplace=True)     

# forecast out is the number of days in integer to which we are predicting the value of our label(i.e. forecast_col)
# so if the dataframe contains 1000 days of entry, we are choosing 0.01 or 1% i.e. 10 days 
forecast_out=int(math.ceil(0.01*len(df)))

# Now the question of why are we doing that? To check the final table of data which we are going to use to train our algorithm
# In the below command, we shifted the forecast column by 1% to compare what the adj. close is today and what it was 10 days ago in the same line (row)
# Our aim is to train a model using the data of 10 days ago to predict a value of today and check if we are coming close.
# We are going iterate using that over the entire dataset to achieve a model
df['label']=df[forecast_col].shift(-forecast_out)

X = np.array(df.drop(['label'], 1)) #Creating the input array, dropping the label column
X = preprocessing.scale(X) #preprocessing X i.e. scaling it
X_lately = X[-forecast_out :] #Values to which we want to predict 
X = X[ : -forecast_out] #The new X which contains up until the point we start forecasting

df.dropna(inplace=True)
y=np.array(df['label']) #This is the output array, this only contains the label

#We use model selection from scikit-learn divide the data into train and test
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

clf= LinearRegression() # setting classifier as linear regression
clf.fit(X_train,y_train) #performing linear regression

#Creating a pickle to save the classifier
with open('linearregression.pickle', 'wb') as f:
	pickle.dump(clf, f)

accuracy = clf.score(X_test,y_test) #Calculating the accuracy of the result

forecast_set = clf.predict(X_lately) #Predicting over the future values that we have set

print(forecast_set, accuracy, forecast_out) #Printing the results

''' Here Stars everything to get a graph where we can relate forecasts with dates (i.e. date on the x axis)'''

df['Forecast'] = np.nan #A new forecast column which is going to forecast stock prices for the next day

last_date = df.iloc[-1].name
last_unix = last_date.timestamp() # finding the timestamp of the last date of known stock values
one_day = 86400 # defining a day in seconds
next_unix = last_unix + one_day # Adding a day to last date to get next unix

for i in forecast_set:
	next_date = datetime.datetime.fromtimestamp(next_unix)
	next_unix += one_day

	#setting all the future X values to nan and attaching all forecast values DATE-WISE
	df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i] 

print(df.tail()) # To reduce confusion on what the for loop does

#Plotting the values in a graph
df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc = 4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
