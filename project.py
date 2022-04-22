import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from Pre_processing import *
from sklearn.metrics import r2_score
import time

data = pd.read_csv('player-value-prediction.csv')
data.fillna(value=0, inplace=True)
# print(data)
data.drop('national_team', axis=1, inplace=True)
data.drop('national_rating', axis=1, inplace=True)
data.drop('national_team_position', axis=1, inplace=True)
data.drop('national_jersey_number', axis=1, inplace=True)
data.drop('tags', axis=1, inplace=True)
X = data.iloc[:, 0:86]  # Features
Y = data['value']  # Label

z = X['work_rate'].str.split('/', expand=True)
f = X['positions'].str.split(',', expand=True)
X['work_rate'] = z[0]
X['work_rate_2'] = z[1]
X['positions'] = f[0]
X['positions_2'] = f[1]
X['positions_3'] = f[2]
X['positions_4'] = f[3]

label_encoder = preprocessing.LabelEncoder()
X['work_rate'] = label_encoder.fit_transform(X['work_rate'])
X['work_rate_2'] = label_encoder.fit_transform(X['work_rate_2'])
X['positions'] = label_encoder.fit_transform(X['positions'])
X['positions_2'] = label_encoder.fit_transform(X['positions_2'])
X['positions_3'] = label_encoder.fit_transform(X['positions_3'])
X['positions_4'] = label_encoder.fit_transform(X['positions_4'])


cols = (
    'name', 'full_name', 'birth_date', 'age', 'nationality', 'preferred_foot', 'body_type',
    'club_team', 'club_position', 'club_join_date', 'contract_end_year', 'traits',)
X = Feature_Encoder(X, cols)

# print(X.iloc[:, 60:86])

for i in range(60, 86):
    X.iloc[:, i] = X.iloc[:, i].str.split('+', expand=True)[0]
    X.iloc[:, i].fillna(value=0, inplace=True)
    X.iloc[:, i] = X.iloc[:, i].astype(int)
# print(X.iloc[:, i])


corr = data.corr()
top_feature = corr.index[abs(corr['value']) > 0.3]
plt.subplots(figsize=(12, 8))
top_corr = data[top_feature].corr()
sns.heatmap(top_corr, annot=True)
# plt.show()
# print(corr)
top_feature = top_feature.delete(-1)
# print(X)
X = X[top_feature]
X.drop('ball_control', axis=1, inplace=True)
X.drop('long_passing', axis=1, inplace=True)
# print(X.iloc[:,6:9])
X = featureScaling(X, 0, 1)
# print(top_feature)
# Split the data to training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, shuffle=True, random_state=10)

cls = linear_model.LinearRegression()
start = time.time()
cls.fit(X_train, y_train)
end = time.time()
prediction1 = cls.predict(X_test)
prediction2 = cls.predict(X_train)

print('Mean Square Error train', metrics.mean_squared_error(y_train, prediction2))
print('Mean Square Error test', metrics.mean_squared_error(y_test, prediction1))
print('accuracy = ', r2_score(y_test, prediction1))
print('train time = ', end - start, ' s')
true_player_value = np.asarray(y_test)[9]
predicted_player_value = prediction1[9]
print('True value for player in the test set is : ' + str(true_player_value))
print('Predicted value player in the test set is : ' + str(predicted_player_value))

# polynomial model
print('--------------------------------------------------------------------------------------------')
poly_features = PolynomialFeatures(degree=3)
X_train_poly = poly_features.fit_transform(X_train)
poly_model = linear_model.LinearRegression()
start2 = time.time()
poly_model.fit(X_train_poly, y_train)
end2 = time.time()
prediction_1 = poly_model.predict(poly_features.fit_transform(X_test))
prediction_2 = poly_model.predict(poly_features.fit_transform(X_train))
print('Mean Square Error train', metrics.mean_squared_error(y_train, prediction_2))
print('Mean Square Error test', metrics.mean_squared_error(y_test, prediction_1))
print('accuracy = ', r2_score(y_test, prediction_1))
print('train time = ', end2 - start2, ' s')
true_player_value = np.asarray(y_test)[9]
predicted_player_value = prediction_1[9]
print('True value for player in the test set is : ' + str(true_player_value))
print('Predicted value for player in the test set is : ' + str(predicted_player_value))
