import pandas as pd
from pandas import Series, DataFrame
import time
from sklearn.tree import DecisionTreeClassifier

def make_labels(row):
    if row == 'TRANSFER':
        return 1
    elif row == 'CASH_IN':
        return 2
    elif row == 'DEBIT':
        return 3
    elif row == 'PAYMENT':
        return 4
    else:
        return 0

def underSampling(df):
   return  pd.concat([
       df.loc[df['type_id']==0][0:600]
        ,df.loc[df['type_id']==1][0:600],
       df.loc[df['type_id']==2][0:460],
       df.loc[df['type_id']==3][0:460],
        df.loc[df['type_id']==4][0:460]

   ])

training_data = pd.read_csv('train.csv')

training_data.columns = ["type", "montant", "compteSource","sourceAvant", "sourceApres","compteDist","distAvant","distApres","isFraud"]
training_data['OrigC']=training_data['compteSource'].apply(lambda x: 1 if str(x).find('C')==0 else 0)
training_data['DestC']=training_data['compteDist'].apply(lambda x: 1 if str(x).find('C')==0 else 0)
training_data['type_id']=training_data['type'].apply(make_labels)

train_data_X = training_data[['type_id', 'montant', 'OrigC', 'sourceAvant', 'sourceApres', 'DestC', 'distApres']].copy()
train_data_y = training_data[['isFraud']].copy()

test_data = pd.read_csv('test.csv')
test_data.columns = ["type", "montant", "compteSource","sourceAvant", "sourceApres","compteDist","distAvant","distApres","isFraud"]

test_data['OrigC']=test_data['compteSource'].apply(lambda x: 1 if str(x).find('C')==0 else 0)
test_data['DestC']=test_data['compteDist'].apply(lambda x: 1 if str(x).find('C')==0 else 0)
test_data['type_id']=test_data['type'].apply(make_labels)
test_data_X = test_data[['type_id', 'montant', 'OrigC', 'sourceAvant', 'sourceApres', 'DestC', 'distApres']].copy()
test_data_y = test_data[['isFraud']].copy()

predict_data = pd.read_csv('predict.csv')
predict_data.columns = ["type", "montant", "compteSource","sourceAvant", "sourceApres","compteDist","distAvant","distApres"]
predict_data['OrigC']=predict_data['compteSource'].apply(lambda x: 1 if str(x).find('C')==0 else 0)
predict_data['DestC']=predict_data['compteDist'].apply(lambda x: 1 if str(x).find('C')==0 else 0)
predict_data['type_id']=predict_data['type'].apply(make_labels)
predict_data_X = predict_data[['type_id', 'montant', 'OrigC', 'sourceAvant', 'sourceApres', 'DestC', 'distApres']].copy()

train_sampled = underSampling(training_data)
train_sampled_X = training_data[['type_id', 'montant', 'OrigC', 'sourceAvant', 'sourceApres', 'DestC', 'distApres']].copy()
train_sampled_y = training_data[['isFraud']].copy()


tree = DecisionTreeClassifier(random_state=0)
t = time.process_time()
tree.fit(train_sampled_X, train_sampled_y)
elapsed_time = time.process_time() - t
training_accuracy = tree.score(train_sampled_X, train_sampled_y)
test_accuracy = tree.score(test_data_X, test_data_y)
print("Training accuracy:", training_accuracy)
print("Test accuracy:", test_accuracy)
print(elapsed_time)

tree_results = tree.predict(predict_data_X)

final_predict_data = pd.read_csv('predict.csv')
final_predict_data.columns = ["type", "montant", "compteSource","sourceAvant", "sourceApres","compteDist","distAvant","distApres"]
final_predict_data['isFraud']=tree_results
final_predict_data.to_csv("outScript.csv",header=False,index=False)