
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree

# function used for fast switching between models
def create_model(model_name: str, num_iter: int, class_weight: dict, random_state: int):
    d = {
        'logistic': LogisticRegression(penalty='l2',
                                       random_state=random_state,
                                       max_iter=num_iter,
                                       solver='liblinear',
                                       class_weight=class_weight),
        'svm': svm.SVC(max_iter=num_iter,
                       random_state=random_state,
                       gamma='scale',
                       class_weight=class_weight),
        'random-forest': RandomForestClassifier(random_state=random_state,
                                                class_weight=class_weight,
                                                n_estimators=10),
        'decision-tree': tree.DecisionTreeClassifier(random_state=random_state)
    }
    return d[model_name]