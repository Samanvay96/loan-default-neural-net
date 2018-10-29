import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix

def read_dataset(fpath, y_feature):
    wine = pd.read_csv(fpath).fillna(0)
    wine = convert_categoricals(wine)

    # Create X and Y datasets
    X = wine.drop(y_feature,axis=1)
    y = wine[y_feature]
    return X, y

# Split dataset into test and train
def split_dataset(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    return X_train, X_test, y_train, y_test

def scale_features(X_train, X_test):
    # Scale Features
    scaler = StandardScaler()

    # Configure
    scaler.fit(X_train)
    StandardScaler(copy=True, with_mean=True, with_std=True)

    return scaler.transform(X_train), scaler.transform(X_test)

# Initialise model
def run_model(X_train, y_train):
    mlp = MLPClassifier(hidden_layer_sizes=(3),max_iter=100)
    mlp.fit(X_train,y_train)
    return mlp

def convert_categoricals(df):
    return pd.get_dummies(df)

def run(fpath, y_feature):
    X, y = read_dataset(fpath, y_feature)
    X_train, X_test, y_train, y_test = split_dataset(X, y)
    X_train, X_test = scale_features(X_train, X_test)
    mlp = run_model(X_train, y_train)
    predictions = mlp.predict(X_test)
    print(classification_report(y_test,predictions))

if __name__ == '__main__':
    run('data/LoanStats3a.csv', 'loan_status_Charged Off')
