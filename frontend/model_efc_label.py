import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import numpy as np

# Load the dataset
df = pd.read_csv('colleges_data.csv')
df.head()

# Drop the "College Name" column
df = df.drop('College Name', axis=1)

# Separate categorical and numerical columns
categorical_cols = ['Genders Accepted', 'University', 'Courses', 'Facilities', 'City', 'State', 'Country', 'College Type']
numerical_cols = ['Total Student Enrollments', 'Total Faculty', 'Established Year', 'Average Fees', 'StudentFacultyRatio']

# Label encode categorical variables
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# Scale numerical variables
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Split the data into training and testing sets
X = df.drop('Fee Category', axis=1)
y = df['Fee Category']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a random forest classifier
rfc = RandomForestClassifier(n_estimators=100, random_state=42)
rfc.fit(X_train, y_train)

# Save the model to a pickle file
with open('admission_model.pkl', 'wb') as f:
    pickle.dump({'model': rfc, 'le': le, 'scaler': scaler, 'categorical_cols': categorical_cols, 'numerical_cols': numerical_cols}, f)

# Evaluate the model on the testing set
y_pred = rfc.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Example only ------------------------------------------------------->
input_data = pd.DataFrame({
    'Genders Accepted': ['Co-Ed'],
    'Total Student Enrollments': [5000],
    'Total Faculty': [200],
    'Established Year': [2010],
    'University': ['NaN'],
    'Courses': ['Course1'],
    'Facilities': ['Facility1'],
    'City': ['New City'],
    'State': ['New State'],
    'Country': ['India'],
    'College Type': ['Private'],
    'Average Fees': [400000],
    'StudentFacultyRatio': [15.0]
})

# Fill NaN values
input_data.fillna({'University': 'Unknown'}, inplace=True)

# Label encode categorical variables
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# Scale numerical variables
input_data_numerical_cols = input_data[numerical_cols]
input_data_numerical_cols = scaler.transform(input_data_numerical_cols)

# Combine categorical and numerical variables
input_data_combined = np.concatenate((input_data[categorical_cols], input_data_numerical_cols), axis=1)

# Make prediction
def make_predictions(input_data):
    # Label encode categorical variables
    le_input = LabelEncoder()
    for col in categorical_cols:
        input_data[col] = le_input.fit_transform(input_data[col])

    # Scale numerical variables
    input_data_numerical_cols = input_data[numerical_cols]
    scaled_input_numerical_cols = scaler.transform(input_data_numerical_cols)

    # Combine categorical and numerical variables
    input_data_combined = np.concatenate((input_data[categorical_cols], scaled_input_numerical_cols), axis=1)

    # Make predictions
    predictions = rfc.predict(input_data_combined)

    return predictions

print(make_predictions(input_data))