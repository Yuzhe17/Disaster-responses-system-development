# Disaster Response Classification System

![homepage](web_app_screenshots/homepage.JPG)

## Introduction

In this project, we aims to design a web-based disaster response classification system which labels
specific message with different categories. The projects can be divided into the following steps:

1. Establishing an ETL pipeline to gather the raw data, wrangle the data and store the data in a database.

2. Creating a machine learning pipeline to correctly classify the disaster response

3. Developing a web app to classify disaster response messages in real time.

## Getting started

### Dependencies

In order to use the project, you need the following python packages:
1. numpy, pandas
2. sklearn, nltk
3. SQLAlchemy
4. flask, plotly

### Installing

Clone this repository:
'git clone https://github.com/Yuzhe17/Disaster-responses-system-development.git'

### Executing the program
To execute the ETL pipeline, run 'process_data.py 'data/disaster_messages.csv' 'data/disaster_categories.csv' 'disaster_response.db''

To execute the machine learning pipeline, run 'train_classifier.py 'disaster_response.db' 'model/classifier.pkl''

To deploy the web app, go the DisasterResponse_app and run the run.py, and go to 
[http://localhost:3001/index](http://localhost:3001/index)

## Files
DisasterResponse_app/templates/go.html: a html file which displays the classification result
DisasterResponse_app/templates/master.html: a html file which displays the visualizations of the data
DisasterResponse_app/custom_transformers.py: a python scripts containing custom sklearn transformers
DisasterResponse/run.py: a python script containing web app initiation and routes
data/: a folder containing two csv files of meassages and categories of all disaster responses
model/classifier.pkl: pickle file of a sklearn classifier
process_data.py: a python script implementing an ETL pipeline
train_classifier.py: a python script implementing a machine learning pipeline
disaster_response.db: a sqlite database storing all the disaster response messages and categories

## Screenshots
![homepage](web_app_screenshots\homepage.JPG)
![classification](web_app_screenshots\homepage.JPG)

## Acknowledgements
Thanks to [figure eight](https://appen.com/) for providing the dataset













