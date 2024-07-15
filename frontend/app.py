from flask import Flask, request, jsonify, render_template
import pickle  # or joblib, depending on your model
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load the saved model
with open('admission_model.pkl', 'rb') as admission:
    saved_model = pickle.load(admission)
    model = saved_model['model']
    le = saved_model['le']
    scaler = saved_model['scaler']
    categorical_cols = saved_model['categorical_cols']
    numerical_cols = saved_model['numerical_cols']

# homepage
@app.route('/')
def home():
    return render_template('intel.html')

#prediction
@app.route('/result', methods=['POST'])
def predict():
    state_name = request.form['state']
    college_type = request.form['college_type']
    avg_fee = request.form["average_fees"]
    est_year = request.form['established_year']
    gen_accepted = request.form['genders_accepted']

    input_data = pd.DataFrame({
    'Genders Accepted': [gen_accepted],
    'Total Student Enrollments': [4500],
    'Total Faculty': [142],
    'Established Year': [est_year],
    'University': ['NaN'],
    'Courses': ['Course1'],
    'Facilities': ['Facility1'],
    'City': ['New City'],
    'State': [state_name],
    'Country': ['India'],
    'College Type': [college_type],
    'Average Fees': [avg_fee],
    'StudentFacultyRatio': [15.0]
})
    input_data.fillna({'University': 'Unknown'}, inplace=True)

    # Label encode categorical variables
    le = LabelEncoder()
    for col in categorical_cols:
        input_data[col] = le.fit_transform(input_data[col]) 

    # Scale numerical variables
    input_data_numerical_cols = input_data[numerical_cols]
    scaled_input_numerical_cols = scaler.transform(input_data_numerical_cols)

    # Combine categorical and numerical variables
    input_data_combined = np.concatenate((input_data[categorical_cols], scaled_input_numerical_cols), axis=1)

    # Make predictions
    predictions = model.predict(input_data_combined)

    # Loading the dataframe
    data = pd.read_csv("colleges_data.csv")
    data.head()

    print(predictions)
    data = data[data["Fee Category"]==predictions[0]]
    data = data[data["Genders Accepted"] == gen_accepted] 
    data = data[data["State"]==state_name] 
    data = data[data["College Type"]==college_type]
    #data = data[data["Average Fees"]<= int(avg_fee)+100000]
    #data = data[data["Average Fees"]>= int(avg_fee)-100000]
    data = data[["College Name", "Average Fees","State","College Type"]]
    data_html = data.to_html()
    print(data)
    return render_template("result.html", value=predictions[0],data_var=data_html)

if __name__ == '__main__':
    app.run(debug=True)