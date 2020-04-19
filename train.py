# train.py ---------------------------


from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from imblearn.over_sampling import ADASYN
import pandas as pd
import pickle

def main():

    # read in data
    data = pd.read_csv("data/creditcard.csv")

    # split out training data
    X_train, X_test, y_train, y_test = train_test_split(data.drop('Class', axis=1),
                                                        data['Class'],
                                                        test_size=0.2,
                                                        random_state=123
                                                        )

    # run adasyn on train set
    adasyn = ADASYN(random_state=123)
    X_adasyn, y_adasyn = adasyn.fit_resample(X_train, y_train)
    print(X_adasyn.shape, y_adasyn.shape)

    # initialize and fit the the model
    forest_adasyn = ExtraTreesClassifier(
        n_estimators=500,
        max_depth=6,
        criterion='gini',
        random_state=123
    )

    forest_adasyn.fit(X_adasyn, y_adasyn)

    pickle.dump(forest_adasyn, open('pkl_objects/adasyn.pkl', 'wb'), protocol=4)
    print("Done")

    return


if __name__ == "__main__":
    main()
