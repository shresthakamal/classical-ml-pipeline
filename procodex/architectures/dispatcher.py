from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

MODELS = {
    "logisticregression": LogisticRegression,
    "svm": svm.SVC,
    "decisiontree": DecisionTreeClassifier,
    "knn": KNeighborsClassifier,
    "naivebayes": GaussianNB,
    "randomforest": RandomForestClassifier,
    "gradientboost": GradientBoostingClassifier,
}


if __name__ == "__main__":
    pass
