# train.py ---------------------------


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import pandas as pd
import pickle

def main():

    # read in data
    data = pd.read_csv("data/creditcard.csv")

    _RANDOM_STATE_ = 0
    X_train, X_test, y_train, y_test = train_test_split(data.drop('Class', axis=1),
                                                        data['Class'],
                                                        test_size=0.2,
                                                        random_state=_RANDOM_STATE_
                                                        )

    # build out SMOTE Data set
    sm = SMOTE(random_state=_RANDOM_STATE_)
    X_smote, y_smote = sm.fit_resample(X_train, y_train)

    # initialize and fit the the model
    rand_forest_smote = RandomForestClassifier(random_state=_RANDOM_STATE_)

    rand_forest_smote.fit(X_smote, y_smote)

    pickle.dump(rand_forest_smote, open('pkl_objects/model.pkl', 'wb'), protocol=4)
    print("Done")

    return


if __name__ == "__main__":
    main()
