import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import warnings

def predict_show_no_show(customer_name, phone_number, email):
    df = pd.read_csv('collective_dataset_with_duplicates.csv')

    used_data = df[(df['Customer Name'] == customer_name) &
                   (df['Phone Number'] == phone_number) &
                   (df['Email'] == email)]

    if used_data.empty:
        return "No historical data found for the specified customer."
    else:
        # Majority voting using a Random Forest of 100 trees
        num_trees = len(used_data)
        forest_predictions = []

        for tree_num in range(num_trees):
            # Randomly sample data with replacement (bootstrapping)
            sampled_data = used_data.sample(frac=1, replace=True)

            # Train a tree on the sampled data
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tree_model = RandomForestClassifier(n_estimators=1)
                tree_model.fit(sampled_data[['Customer ID']], sampled_data['Show/NoShow'])

            # Make a prediction for the user input
            tree_prediction = tree_model.predict([[sampled_data['Customer ID'].iloc[0]]])

            # Convert prediction to int or use 1 for 'Show' and 0 for 'No Show'
            tree_prediction_numeric = 1 if tree_prediction[0] == 'Show' else 0
            forest_predictions.append(tree_prediction_numeric)

        # Determine the final prediction based on majority voting
        final_prediction = 'Show' if sum(forest_predictions) > num_trees / 2 else 'No Show'

        return f"The Random Forest predicts that the customer '{customer_name}' with Phone Number '{phone_number}' and Email '{email}' will likely '{final_prediction}' for the appointment."
