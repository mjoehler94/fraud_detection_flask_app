# modelapi.pyp -------------------------------------

# libraries
import pickle
import pandas as pd
import numpy as np
# from sklearn.ensemble import ExtraTreesClassifier


# helper functions  -------
def load_model():
    fraud_model = pickle.load(open('pkl_objects/model.pkl', 'rb'))
    return fraud_model


def make_prediction(model, df_string):
    # if True:
    #     print("Skipping model for now")
    #     return 1, 1
    # )
    try:
        entry_data_file = "data/entry_data.json"
        with open(entry_data_file, 'w') as f:
            f.write(df_string)
        df = pd.read_json(entry_data_file)
        result = model.predict(df)[0]
        proba = np.round(model.predict_proba(df).max(), 4)
    except ValueError:
        result = -1
        proba = "Error"
    label = {0: 'Not Fraud :)', 1: 'Fraud!', -1: "Error"}
    return label[result], proba


def main():

    good_df = pd.read_json("data/good_example.json")
    good_df['Time'] = good_df['Time'].astype('float')
    print(good_df)

    fraud_df = pd.read_json("data/fraud_example.json")
    print(fraud_df)

    # mod = load_model()
    print("Inside main function of modelapi.py")

    print(make_prediction(good_df))
    print(make_prediction(fraud_df))
    return


if __name__ == '__main__':
    main()
