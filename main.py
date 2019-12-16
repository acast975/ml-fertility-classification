import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold


def create_model(model_name: str, num_iter: int, class_weights: dict):
    d = {
        'logistic': LogisticRegression(penalty='l2',
                                       random_state=RS,
                                       max_iter=num_iter,
                                       solver='liblinear',
                                       class_weight=class_weights),
        'svm': svm.SVC(max_iter=num_iter,
                       random_state=RS,
                       gamma='scale',
                       class_weight=class_weights),
        'random-forest': RandomForestClassifier(random_state=RS,
                                                class_weight=class_weights,
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

columns_y = [
    'Output'
]

RS = 1
NUM_SPLITS = 5

# class weight to fix bad model predictions
class_weights = {
    'N': 1,
    'O': 8
}

df = pd.read_csv(data_file_name)

X = df[columns_x].values
Y = df[columns_y].values

kf = KFold(n_splits=NUM_SPLITS, shuffle=True, random_state=RS)

for train_index, test_index in kf.split(X, Y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]
    model = create_model(
        'random-forest',
        300,
        class_weights
    ).fit(X_train, np.reshape(y_train, (y_train.shape[0])))
    score = model.score(X_test, y_test)
    score_ds = model.score(X, Y)

    df_test = df.loc[df.Output == 'O']
    predictions = model.predict(df_test[columns_x].values)

    print('Test score: ', score)
    print('Whole dataset score: ', score_ds)
    print(predictions)




