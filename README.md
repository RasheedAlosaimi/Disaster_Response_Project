# Disaster Response Pipeline Project

## Overview:
The project is a part of the requirements of Data Scientist Nanodegree Program at Udacity. This particular project's materials that contain pre-labelled tweets and text messages were provided by Figure Eight, it is gathered from real-life disasters. The objective of the project is to build a supervised learning model that can accurately classify messages in order to help for future disaster reponses. The project is divided into three stages that starts with preparing the data using ETL pipeline, then using a machine learning pipeline for building the model, and finally deploying the model on Web App using Flask. 



### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
https://SPACEID-3001.SPACEDOMAIN
