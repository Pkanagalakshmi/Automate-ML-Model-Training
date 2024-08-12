

# Automate ML Model Training

## Project Overview

The `Automate-ML-Model-Training` project is designed to provide an interactive web interface for automating the training of machine learning models. Using Streamlit, users can easily select datasets, preprocess data, choose from various models, train them, and evaluate their performance—all without writing any code.

## Features

- **Dataset Selection**: Choose datasets from a predefined directory.
- **Data Preprocessing**: Handle missing values, scaling, and encoding of features.
- **Model Training**: Train models using Logistic Regression, Support Vector Classifier, Random Forest Classifier, or XGBoost Classifier.
- **Model Evaluation**: Evaluate model performance and view accuracy metrics.

## Setup and Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/Pkanagalakshmi/Automate-ML-Model-Training.git
   cd Automate-ML-Model-Training
   ```

2. **Install Dependencies**

   Install the required Python packages by running:

   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare Data**

   Place your dataset files (CSV or Excel) in the `data` directory.

4. **Run the Application**

   Start the Streamlit app using:

   ```bash
   streamlit run app.py
   ```

## Code Explanation

### Streamlit App (`app.py`)

This script creates an interactive web interface using Streamlit. It allows users to:
- Select datasets and view them.
- Choose preprocessing options, target columns, and machine learning models.
- Train and evaluate the selected model.

#### Key Components
- **Imports**: Various libraries for creating the UI and handling machine learning models.
- **Streamlit Configuration**: Sets up the page title, icon, and layout.
- **UI Elements**: Dropdowns for selecting datasets, target columns, scalers, and models, and a button for initiating training.
- **Model Training and Evaluation**: Calls utility functions to preprocess data, train the model, and display accuracy.

### Utility Functions (`ml_utility.py`)

This module contains helper functions used by the Streamlit app:

#### **Read Data**
```python
def read_data(file_name):
    file_path = f"{parent_dir}/data/{file_name}"
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
        return df
    elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        df = pd.read_excel(file_path)
        return df
```
- **`read_data`**: Reads data from CSV or Excel files into a Pandas DataFrame.

#### **Preprocess Data**
```python
def preprocess_data(df, target_column, scaler_type):
    X = df.drop(columns=[target_column])
    y = df[target_column]

    numerical_cols = X.select_dtypes(include=['number']).columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns

    if len(numerical_cols) > 0:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        num_imputer = SimpleImputer(strategy='mean')
        X_train[numerical_cols] = num_imputer.fit_transform(X_train[numerical_cols])
        X_test[numerical_cols] = num_imputer.transform(X_test[numerical_cols])

        if scaler_type == 'standard':
            scaler = StandardScaler()
        elif scaler_type == 'minmax':
            scaler = MinMaxScaler()

        X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
        X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

    if len(categorical_cols) > 0:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        X_train[categorical_cols] = cat_imputer.fit_transform(X_train[categorical_cols])
        X_test[categorical_cols] = cat_imputer.transform(X_test[categorical_cols])

        encoder = OneHotEncoder()
        X_train_encoded = encoder.fit_transform(X_train[categorical_cols])
        X_test_encoded = encoder.transform(X_test[categorical_cols])
        X_train_encoded = pd.DataFrame(X_train_encoded.toarray(), columns=encoder.get_feature_names(categorical_cols))
        X_test_encoded = pd.DataFrame(X_test_encoded.toarray(), columns=encoder.get_feature_names(categorical_cols))
        X_train = pd.concat([X_train.drop(columns=categorical_cols), X_train_encoded], axis=1)
        X_test = pd.concat([X_test.drop(columns=categorical_cols), X_test_encoded], axis=1)

    return X_train, X_test, y_train, y_test
```
- **`preprocess_data`**: Handles missing values, scales numerical features, and one-hot encodes categorical features.

#### **Train Model**
```python
def train_model(X_train, y_train, model, model_name):
    model.fit(X_train, y_train)
    with open(f"{parent_dir}/trained_model/{model_name}.pkl", 'wb') as file:
        pickle.dump(model, file)
    return model
```
- **`train_model`**: Trains the selected model and saves it to a file.

#### **Evaluate Model**
```python
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy = round(accuracy, 2)
    return accuracy
```
- **`evaluate_model`**: Evaluates the model’s accuracy on the test set.

