import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold


# function used for fast switching between models
def create_model(model_name: str, num_iter: int, class_weight: dict):
    d = {
        'logistic': LogisticRegression(penalty='l2',
                                       random_state=RS,
                                       max_iter=num_iter,
                                       solver='liblinear',
                                       class_weight=class_weight),
        'svm': svm.SVC(max_iter=num_iter,
                       random_state=RS,
                       gamma='scale',
                       class_weight=class_weight),
        'random-forest': RandomForestClassifier(random_state=RS,
                                                class_weight=class_weight,
                                                n_estimators=100)
    }
    return d[model_name]


data_file_name = "./fertility_Diagnosis.csv"

data_columns = [
    'Season',
    'Age',
    'Childish diseases',
    'Accidents or serious trauma',
    'Surgical intervention',
    'High fevers in the last year',
    'Frequency of alcohol consumption',
    'Smoking habit',
    'Number of hours spent sitting per day',
    'Output'
]

# columns used for prediction
columns_x = [
    # 'Season',
    'Age',
    'Childish diseases',
    'Accidents or serious trauma',
    'Surgical intervention',
    'High fevers in the last year',
    'Frequency of alcohol consumption',
    'Smoking habit',
    'Number of hours spent sitting per day'
]

# columns that are predicted
columns_y = [
    'Output'
]

# random-state, used as seed value for other libraries, so that we always get the same result
RS = 1
# number of splits during cross-validation
NUM_SPLITS = 5

# class weights to fix imbalanced classes
class_weights = {
    'N': 1,
    'O': 8
}

# read data from csv file
df = pd.read_csv(data_file_name)

# get numpy representation of training data
X = df[columns_x].values
Y = df[columns_y].values

# create cross-validation
kf = KFold(n_splits=NUM_SPLITS, shuffle=True, random_state=RS)

# iterate over cross-validation sets
# train_index - indices of records used for training at current iteration over cross-validation sets
# test_index - indices of records used for testing at current iteration over cross-validation sets
for train_index, test_index in kf.split(X, Y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]
    # create and train model
    model = create_model(
        'random-forest',
        300,
        class_weights
    ).fit(X_train, np.reshape(y_train, (y_train.shape[0])))

    # get model score on test set
    score = model.score(X_test, y_test)
    # get model score on whole data-set
    score_ds = model.score(X, Y)

    # test model on members of class 'O'
    df_test = df.loc[df.Output == 'O']
    predictions = model.predict(df_test[columns_x].values)

    # print results
    print('Test score: ', score)
    print('Whole data-set score: ', score_ds)
    print(predictions)




