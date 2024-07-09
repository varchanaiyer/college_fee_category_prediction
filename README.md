# 🏫College Fee Category Prediction

-Anushka Rathour

## 📚Overview

* This project analyzes a dataset of engineering colleges in India to determine the fee category based on various features and provides lists of colleges according to the cattegory.
* The project utilizes Random Forest machine learning techniques to classify colleges into different fee categories that is High ,Low,and Medium. 
* Offers a Flask API to serve predictions, along with an HTML frontend for user interaction.

## 🚀 Instructions for Running the Code

1️⃣ Clone the Repository
            
    git clone https://https://github.com/rathour-anushka/college_fee_category_prediction
     cd college_fee_category_predictor

2️⃣ Set Up a Virtual Environment
 
 It's recommended to use a virtual environment to manage dependencies.

    python -m venv venv
    source venv/Scripts/activate     # Work On Windows 

3️⃣ Install Dependencies

Install the required packages using pip:

     pip install -r requirements.txt

4️⃣ Run the Flask Application

Start the Flask server by running:

    python app.py

The application will be available at http://127.0.0.1:5000/.

5️⃣ Access the Frontend

Open your web browser and navigate to http://127.0.0.1:5000/ to access the HTML frontend and interact with the application.

![web front](https://github.com/rathour-anushka/college_fee_category_prediction/blob/main/frontend/testing/img_1.png)

## 📦 Dependencies

The project requires the following dependencies:

Flask 🌐

Pandas 🐼

NumPy 🔢

Scikit-learn 📊

Matplotlib 📉

Seaborn 🖌️

Jupyter 📓

These are listed in the requirements.txt file and can be installed using pip install -r requirements.txt.

## 🔧 Additional Scripts

- **Data Preprocessing** (intel project.ipynb)
  
  This script handles the cleaning and preprocessing of the dataset.

- **Model Training** (intel project.ipynb)

    This script trains the machine learning model and saves it for later use.

- **Model Evaluation** (intel project.ipynb)

   This script evaluates the performance of the trained model and prints a classification report.

- **Flask Application** (app.py)

   This script is the main Flask application that serves the model and provides an API for predictions.

## 🛠️Usage

- **API Endpoints**

  GET /

  Returns the HTML frontend where users can interact with the application.

  POST /predict

  Accepts input data in JSON format and returns the fee category prediction based on the trained 
  machine learning model.

- **Data Analysis**

  The intel project.ipynb notebook contains detailed steps for data analysis, including:

  1. Data Preparation and Cleaning 🧹
  2. Exploratory Data Analysis (EDA) 🔍
  3.  Classification using Decision Tree,SVM and Random Forest 🌳
  4. Model Evaluation and Validation ✅
  
  Users can open this notebook in Jupyter to explore and understand the analysis process.

- **Classification and Prediction**
  1. Input Features: Total Student Enrollments, Total Faculty, Established Year, and other relevant features.
  2. Output: Fee Category (Low, Medium, High)
  3. The trained model classifies colleges into fee categories based on the input features.
  
- **Visualization**

  Visualizations are generated using Matplotlib and Seaborn and can be found in the project.ipynb 
  notebook. These visualizations help in understanding the distribution of features and the 
  performance of the classification model.

 ##  🔍Conclusion

The College Fee Category Predictor project is a powerful tool designed to simplify the process of evaluating and categorizing engineering colleges in India based on their fees. By leveraging advanced machine learning techniques and providing a user-friendly interface, this project aims to assist students, parents, and educational consultants in making informed decisions.
 
