import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os

#Read in file
df = pd.read_csv('src/data/Account_Sales.csv')
df = df.drop(['Days Since Most Recent Deal Close','Employees','Billing Country','Industry',
                'Owner ID','Annual Revenue','Account Type','Average Age','Account Source',
                'Top Product Family','Days Since First Deal Close','Account ID', 'Account Name',
                'Created Date','Billing State/Province','Customer Priority','First Deal Date',
                'Number of Locations','Top Product Name'
                ],axis=1)

# Save the data in train and test directories
def save_train_test_data(X_train, y_train,X_test, y_test):
    if not os.path.isdir("src/data/train_model_data"):
        os.mkdir("src/data/train_model_data")
    np.savetxt("src/data/train_model_data/train_features.csv", X_train)
    np.savetxt("src/data/train_model_data/train_labels.csv", y_train)
    # pd.DataFrame(X_train).to_csv("src/data/train_model_data/train_features.csv")
    # pd.DataFrame(y_train).to_csv("src/data/train_model_data/train_labels.csv")
    
    if not os.path.isdir("src/data/test_model_data"):
        os.mkdir("src/data/test_model_data")
    np.savetxt("src/data/test_model_data/test_features.csv", X_test)
    np.savetxt("src/data/test_model_data/test_labels.csv", y_test)
    # pd.DataFrame(X_test).to_csv("src/data/test_model_data/test_features.csv")
    # pd.DataFrame(y_test).to_csv("src/data/test_model_data/test_labels.csv")
    return

# Save the dataframe
def save_processed_data(df: pd.DataFrame):
    output_path = "src/data/Processed_Account_Sales.csv"
    df.to_csv(output_path, index=False)
    return

# Split the data
def split_train_test_data():
    X = df.drop(columns='Total Revenue', axis=1)
    y = df.pop('Total Revenue')
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=2)
    save_train_test_data(X_train, y_train, X_test, y_test)
    save_processed_data(df)
    return X_train, y_train, X_test, y_test
    
split_train_test_data()