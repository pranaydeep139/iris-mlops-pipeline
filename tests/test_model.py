import pytest
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

# Test 1: Data Validation
def test_data_columns():
    """
    Tests if the input data contains the expected columns.
    """
    # Load the test data
    test_df = pd.read_csv('tests/test_data.csv')
    
    expected_columns = [
        'sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'
    ]
    
    # Standardize column names just like in the training script
    test_df.columns = [col.lower().replace('(cm)', '').strip().replace(' ', '_') for col in test_df.columns]
    
    # Assert that all expected columns are present
    assert all(col in test_df.columns for col in expected_columns), "Data validation failed: Missing columns."

# Test 2: Model Evaluation
def test_model_accuracy():
    """
    Tests if the model's accuracy is above a minimum threshold.
    """
    # Load the pre-trained model and test data
    try:
        model = joblib.load('artifacts/model.pkl')
        test_df = pd.read_csv('tests/test_data.csv')
    except FileNotFoundError:
        pytest.fail("Model or test data not found. Ensure DVC has pulled the files.")

    # Prepare the data
    test_df.columns = [col.lower().replace('(cm)', '').strip().replace(' ', '_') for col in test_df.columns]
    X_test = test_df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    y_test = test_df['species']

    # Make predictions
    predictions = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, predictions)
    
    # Assert that the accuracy is at least 90%
    assert accuracy >= 0.9, f"Model evaluation failed: Accuracy {accuracy} is below the threshold of 0.9."
