import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#import pandas_datareader as pdr
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score
from sklearn.metrics import confusion_matrix
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Function for confusion martix
def cm_analysis(y_test, y_pred, labels=[0, 1], ymap=None, figsize=(10,10)):
    if ymap is not None:
        y_pred = [ymap[yi] for yi in y_pred]
        y_test = [ymap[yi] for yi in y_test]
        labels = [ymap[yi] for yi in labels]
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=annot, fmt='', ax=ax)
    #plt.savefig(filename)
    plt.show()

#Load data
#data = pdr.get_data_yahoo("TLSA", "2015-07-10", "2020-07-10")
data = pd.read_csv("C:/Users/Vusal/OneDrive/Desktop/1/Thesis/ProgramToPredict/TSLA (2).csv" , index_col='Date')

#print(data.head(5))
#print(data.tail(5))

print(data.describe())

#Exploratory data analysis
plt.figure(figsize=(15, 5));
plt.plot(data.Close.values, color='violet', label='Close')
plt.title('Stock Price Changes Over Time' , size=20)
plt.xlabel('Time [days]', size=15)
plt.ylabel('Price',size=15)
plt.legend(loc='best')


fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2,figsize=(16,8))
fig.suptitle('Changes in Other instances of Stock Over Time')
ax1.plot(data.High.values, color='green', label='High Price')
ax1.legend(shadow=False, fancybox=True)
ax2.plot(data.Low.values, color='red', label='Low Price')
ax2.legend(shadow=False, fancybox=True)
ax3.plot(data.Open.values, color='darkgrey', label='Open Price')
ax3.legend(shadow=False, fancybox=True)
ax4.plot(data.Volume.values, color='orange', label='Volume')
ax4.legend(shadow=False, fancybox=True)

plt.show()

#Data pre-processing
data["DifCl"] = data.Close.diff()
data["SMA2"] = data.Close.rolling(2).mean()
data["ForceIndex"] = data["Close"] * data["Volume"]
data["y"] = data["DifCl"].apply(lambda x: 1 if x > 0 else 0).shift(-1)

print(data)



data = data.drop(
   ["Open", "High", "Low", "Close", "Volume", "DifCl", "Adj Close"],
   axis=1,
).dropna()

print(data)

X = data.drop(["y"], axis=1).values
y = data["y"].values

 
X_train, X_test, y_train, y_test = train_test_split(
   X,
   y,
   test_size=0.2,
   shuffle=False,
)

#MLP Model
mclf = make_pipeline(StandardScaler(), MLPClassifier(random_state=0, shuffle=False))
mclf.fit(
   X_train,
   y_train,
)
y_pred = mclf.predict(X_test)


#SVM Model 
sclf = make_pipeline(StandardScaler(), SVC(gamma="auto"))
sclf.fit(
   X_train,
   y_train,
)
z_pred = sclf.predict(X_test)

#LSTM Model
model = Sequential()
model.add(LSTM(2, input_shape=(X_train.shape[1], 1)))
model.add(Dense(1, activation="sigmoid"))
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])
model.fit(X_train[:, :, np.newaxis], y_train, epochs=100)
w_pred = model.predict(X_test[:, :, np.newaxis])

#Measures for prediction with MLPClassifier
print("MLP Accuracy Score:" , accuracy_score(y_test, y_pred))
print("MLP Mean Squared Error:" , mean_squared_error(y_test, y_pred))
print("MLP Mean Absolute Error:" , mean_absolute_error(y_test, y_pred))
print("MLP Variance Score:" , explained_variance_score(y_test, y_pred))
print("MLP R Squared:" , r2_score(y_test, y_pred))

cm_analysis(y_test, y_pred)

#Measures for prediction with SVM
print("SVM Accuracy Score:" , accuracy_score(y_test, z_pred))
print("SVM Mean Squared Error:" , mean_squared_error(y_test, z_pred))
print("SVM Mean Absolute Error:" , mean_absolute_error(y_test, z_pred))
print("SVM Variance Score:" , explained_variance_score(y_test, z_pred))
print("SVM R Squared:" , r2_score(y_test, z_pred))

cm_analysis(y_test, z_pred)

#Measures for prediction with LSTM
print("LSTM Accuracy Score:" , accuracy_score(y_test, w_pred > 0.5))
print("LSTM Mean Squared Error:" , mean_squared_error(y_test, w_pred))
print("LSTM Mean Absolute Error:" , mean_absolute_error(y_test, w_pred))
print("LSTM Variance Score:" , explained_variance_score(y_test, w_pred))
print("LSTM R Squared:" , r2_score(y_test, w_pred))

cm_analysis(y_test, w_pred > 0.5)
