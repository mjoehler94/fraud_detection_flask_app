# Credit Card Fraud Detection

[Dataset](https://drive.google.com/file/d/1CTAlmlREFRaEN3NoHHitewpqAtWS5cVQ/view)

[ADASYN Paper](http://scholar.google.com/scholar_url?url=https://sci2s.ugr.es/keel/pdf/algorithm/congreso/2008-He-ieee.pdf&hl=en&sa=X&scisig=AAGBfm1uJ1FIWcOyYTHBq1effUALxWAmHg&nossl=1&oi=scholarr)

[SMOTE Paper](https://arxiv.org/pdf/1106.1813)

___

This project was my capstone project for the Udacity Machine Learning Engineering Nanodegree

**A Quick Guide for navigating this repository**

- The `deliverables` directory contains:
    - jupyter notebook with my data exploration, experimentation, and modeling process
    - `proposal.pdf`: the project proposal submitted before the completion of this project
    - `report.pdf`: The final report that describes this project in detail.
- The other files in this repository are part of the flask app that is used to deploy and interact with the model
    - `train.py`: trains a model and saves it to a pickled binary file that the app can load
    - `modelapi.py`: this script has the functions to load and get predictions from the model
    - `templates`: this directory has all of the html templates used by the web app 
    - `app.py`: **Run this script to load the app and interact with the model**
    -`data`: this directory has the json files needed to use the app. The actual fraud dataset will need to be downloaded
    with the link provided at the top of this `README`
    
 NOTES: 
 - If loading this repository for yourself, you may first need to download the data and run `train.py` to create and save the model.
 - If you don't already have them, you will need `pandas`, `flask`. `wtforms`,`sklearn`, and `imblearn`
 (all of these can be installed with `pip`)
 
