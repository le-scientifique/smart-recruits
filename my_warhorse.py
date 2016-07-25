# This script shows you how to make a submission using a few
# useful Python libraries.
# It gets a public leaderboard score of 0.76077.
# Maybe you can tweak it and do better...?
global pd,np,xgb
import pandas as pd
pd.set_option('display.max_rows',None)
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib
from pandas import Series, DataFrame, Panel
from sklearn import cross_validation, metrics   #Additional sklearn functions
from sklearn.grid_search import GridSearchCV   #Performing grid search
from sklearn.neural_network import MLPClassifier
import time
# Load the data
train_df = pd.read_csv('Train_raw.csv', header=0)
test_df = pd.read_csv('Test_raw.csv', header=0)

# We'll impute missing values using the median for numeric columns and the most
# common value for string columns.
# This is based on some nice code by 'sveitser' at http://stackoverflow.com/a/25562948
from sklearn.base import TransformerMixin
class DataFrameImputer(TransformerMixin):
    def fit(self, X, y=None):
        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].median() for c in X],
            index=X.columns)
        return self
    def transform(self, X, y=None):
        return X.fillna(self.fill)

# postal_codes = pd.read_csv('all_india_pin_code.csv',header=0)
# postal_codes = postal_codes.set_index('pincode')
# unique_pincodes = postal_codes.index.unique()
# districts = []
# states = []

# # for pin in unique_pincodes:
# # 	districts.append(postal_codes.get_value(pin,'Districtname')[0])
# # 	states.append(postal_codes.get_value(pin,'statename')[0])
# # pin_df = pd.DataFrame({"pincode":unique_pincodes,"district_name":districts,"state_name":states},columns=["pincode","district_name","state_name"])

# # pin_df.to_csv("pin_codes_unique.csv", index=False)
# # print pin_df.head()

# Office_PIN
# Applicant_City_PIN

def merge_postal_code_data(pin_df,df):
	df_with_districts = df.merge(pin_df[['pincode','district_name','state_name']], how='left',left_on=['Office_PIN'], right_on=['pincode'])
	df_with_districts.drop('pincode',1)
	df_with_districts.columns = df_with_districts.columns.str.replace('district_name','Office_PIN_District')
	df_with_districts.columns = df_with_districts.columns.str.replace('state_name','Office_PIN_State')

	df_with_d_states = df_with_districts.merge(pin_df[['pincode','district_name','state_name']], how='left',left_on=['Applicant_City_PIN'], right_on=['pincode'])

	df_with_d_states.drop('pincode_x',1)
	df_with_d_states.drop('pincode_y',1)
	df_with_d_states.columns = df_with_d_states.columns.str.replace('district_name','Applicant_City_PIN_District')
	df_with_d_states.columns = df_with_d_states.columns.str.replace('state_name','Applicant_City_PIN_State')

	return df_with_d_states


pin_df = pd.read_csv('pin_codes_unique.csv', header=0)

train_df = merge_postal_code_data(pin_df,train_df)
test_df = merge_postal_code_data(pin_df,test_df)

dropped = [u'Manager_Num_Products',u'Manager_Num_Products2']

feature_columns_to_use = [u'Office_PIN_State',u'Applicant_City_PIN_State',u'Applicant_Gender',u'Applicant_Marital_Status',u'Applicant_Occupation',u'Applicant_Qualification',u'Manager_Joining_Designation',u'Manager_Current_Designation',u'Manager_Grade',u'Manager_Status',u'Manager_Gender',u'Manager_Num_Application',u'Manager_Num_Coded',u'Application_Receipt_Date',u"Applicant_BirthDate",u"Manager_DOJ",u"Manager_DoB","Manager_Business","Manager_Business2","Manager_Num_Products","Manager_Num_Products2"]
date_cols = ["Application_Receipt_Date","Applicant_BirthDate","Manager_DOJ","Manager_DoB"]
categorical_variables = [u'Office_PIN_State',u'Applicant_City_PIN_State',u'Applicant_Gender',u'Applicant_Marital_Status',u'Applicant_Occupation',u'Manager_Status',u'Manager_Gender',"Application_Receipt_Date_Month","Application_Receipt_Date_Year"]


# Join the features from train and test together before imputing missing values,
# in case their distribution is slightly different
big_X = train_df[feature_columns_to_use].append(test_df[feature_columns_to_use])
	
def handle_designations(x):
	if "Level" in str(x):
		return float(str(x).replace('Level','').strip())
	else:
		return float("0")

def profit_or_loss(x):
	if not np.isnan(x):
		if x > 0:
			return -1
		else:
			return 0
	else:
		return 0

def Applicant_Qualification_Encoding(x):
	if x == 'Others':
		return 0
	elif x == 'Class XII':
		return 1
	elif x == 'Graduate':
		return 2
	else:
		return 3

big_X_imputed = DataFrameImputer().fit_transform(big_X)

big_X_imputed['Manager_Joining_Designation'] = big_X_imputed['Manager_Joining_Designation'].apply(handle_designations)
big_X_imputed['Manager_Current_Designation'] = big_X_imputed['Manager_Current_Designation'].apply(handle_designations)

big_X_imputed['Manager_Business'] = big_X_imputed['Manager_Business'].apply(profit_or_loss)
big_X_imputed['Manager_Business2'] = big_X_imputed['Manager_Business2'].apply(profit_or_loss)

big_X_imputed['Applicant_Qualification'] = big_X_imputed['Applicant_Qualification'].apply(Applicant_Qualification_Encoding)

# u'Manager_Grade'

# big_X_imputed['Manager_Changed_Designation'] = big_X_imputed['Manager_Current_Designation'] - big_X_imputed['Manager_Joining_Designation']
#Handling date fields
big_X_imputed["Application_Receipt_Date"] = pd.DatetimeIndex(big_X_imputed["Application_Receipt_Date"])
# print big_X_imputed["Application_Receipt_Date"]

for feature in date_cols:
	if feature in ["Applicant_BirthDate","Manager_DOJ","Manager_DoB"]:
		# print feature
		big_X_imputed["Application_Receipt_Date"] = pd.DatetimeIndex(big_X_imputed["Application_Receipt_Date"])
		big_X_imputed[feature] = pd.DatetimeIndex(big_X_imputed[feature])
		# print big_X_imputed[feature]
		big_X_imputed[feature] = big_X_imputed["Application_Receipt_Date"].dt.year - big_X_imputed[feature].dt.year
	elif feature == "Application_Receipt_Date":
		big_X_imputed["Application_Receipt_Date_Year"] = big_X_imputed["Application_Receipt_Date"].dt.year
		big_X_imputed["Application_Receipt_Date_Month"] = big_X_imputed["Application_Receipt_Date"].dt.month

big_X_imputed.drop("Application_Receipt_Date", axis=1, inplace=True)
# XGBoost doesn't (yet) handle categorical features automatically, so we need to change
# them to columns of integer values.
# See http://scikit-learn.org/stable/modules/preprocessing.html#preprocessing for more
# details and options

#measure of a good manager 

# big_X_imputed['good_manager'] = big_X_imputed['Manager_Joining_Designation'].apply(lambda x: )

le = LabelEncoder()
for feature in categorical_variables:
    big_X_imputed[feature] = le.fit_transform(big_X_imputed[feature])

# #one-hot encoding
# big_X_imputed = pd.get_dummies(big_X_imputed, columns=categorical_variables)

print big_X_imputed.columns
# Prepare the inputs for the model
train_X = big_X_imputed[0:train_df.shape[0]].as_matrix()
test_X = big_X_imputed[train_df.shape[0]::].as_matrix()
train_y = train_df['Business_Sourced']

# You can experiment with many other options here, using the same .fit() and .predict()
# methods; see http://scikit-learn.org
# This example uses the current build of XGBoost, from https://github.com/dmlc/xgboost
alg = xgb.XGBClassifier(
	 learning_rate =0.1,
	 n_estimators=1000,
	 max_depth=10,
	 min_child_weight=1,
	 gamma=0,
	 subsample=0.8,
	 colsample_bytree=0.8,
	 objective= 'binary:logistic',
	 nthread=4,
	 scale_pos_weight=1,
	 seed=27).fit(train_X, train_y)

# alg = MLPClassifier(
# 	verbose=0,
# 	random_state=100,
# 	max_iter=400,
# 	algorithm='adam').fit(train_X, train_y)


# print "AUC Score (Train): %f" % metrics.roc_auc_score(train_y, train_predprob)


# cv_params = {'max_depth': [3,5,7], 'min_child_weight': [1,3,5]}
# ind_params = {'learning_rate': 0.1, 'n_estimators': 1000, 'seed':0, 'subsample': 0.8, 'colsample_bytree': 0.8, 
#              'objective': 'binary:logistic'}
# optimized_GBM = GridSearchCV(xgb.XGBClassifier(**ind_params), 
#                             cv_params, 
#                              scoring = 'roc_auc', cv = 5, n_jobs = -1) 
# # Optimize for accuracy since that is the metric used in the Adult Data Set notation

# optimized_GBM.fit(train_X, train_y)

# print optimized_GBM.grid_scores_

# best_parameters, score, _ = max(optimized_GBM.grid_scores_, key=lambda x: x[1])
# print(score)
# for param_name in sorted(best_parameters.keys()):
# 	print("%s: %r" % (param_name, best_parameters[param_name]))

# test_predictions = optimized_GBM.predict(test_X)
            
    

# Predict training set:
train_predictions = alg.predict(train_X)
train_predprob = alg.predict_proba(train_X)[:,1]

test_predictions = alg.predict(test_X)


# Kaggle needs the submission to have a certain format;
# see https://www.kaggle.com/c/titanic-gettingStarted/download/gendermodel.csv
# for an example of what it's supposed to look like.
sub_columns = ['ID','Business_Sourced']
submission = pd.DataFrame({ 'Business_Sourced': test_predictions,'ID': test_df['ID']},columns=sub_columns)
submission.to_csv("submission_" + str(time.time()) + ".csv", index=False)