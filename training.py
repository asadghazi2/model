import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import dill as pickle
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier


def build_and_train():
    data = pd.read_csv('zones_set.csv')
    data.head(20)
    data.describe()
    data['Dimension'].value_counts().plot(kind='bar')
    plt.title('price ratio')
    plt.xlabel('price per marla')
    plt.ylabel('zone')
    sns.despine()
    reg = LinearRegression()
    label = data['date']
    conv_dates = [1 if values == 2017 else 0 for values in data.date]
    data['date'] = conv_dates
    train1 = data.drop(['id', 'price', 'Name'], axis=1)
    x_train, x_test, y_train, y_test = train_test_split(train1, label, test_size=0.1, random_state=2)
    reg.fit(x_train, y_train)
    reg.score(x_test, y_test)




class PreProcessing(BaseEstimator, TransformerMixin):
    """Custom Pre-Processing estimator for our use-case
    """


if __name__ == '__main__':
    model = build_and_train()

    filename = 'model_v1.pk'
    with open('../flask_api/models/'+filename, 'wb') as file:
        pickle.dump(model, file)