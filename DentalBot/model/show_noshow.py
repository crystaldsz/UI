import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import warnings

df = pd.read_csv('/collective_dataset.csv')

df['Appointment_Date'] = pd.to_datetime(df['Appointment_Date'], format='%d-%m-%Y')

df['Day'] = df['Appointment_Date'].dt.dayofweek
df['Hour'] = df['Appointment_Time'].astype(str).str.split(':').str[0].astype(int)

day_input = input("Enter the day (e.g., Sunday, Monday, etc.): ")
time_input = int(input("Enter the hour (in 24-hour format): "))

day_mapping = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
day = day_mapping.get(day_input.capitalize(), -1)

if day == -1:
    print("Invalid day input.")
else:
    used_data = df[(df['Day'] == day) & (df['Hour'] == time_input)]

    if used_data.empty:
        print("Not enough historical data.")
    else:
        # Majority voting using a Random Forest of 100 trees
        num_trees = 100
        forest_predictions = []

        for tree_num in range(num_trees):
            # Randomly sample data with replacement (bootstrapping)
            sampled_data = used_data.sample(frac=1, replace=True)

            # Train a tree on the sampled data
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tree_model = RandomForestClassifier(n_estimators=1)
                tree_model.fit(sampled_data[['Day', 'Hour']], sampled_data['Show_NoShow'])


            # Make a prediction for the user input
            tree_prediction = tree_model.predict([[day, time_input]])

            # Convert prediction to int or use 1 for 'Show' and 0 for 'No Show'
            tree_prediction_numeric = 1 if tree_prediction[0] == 'Show' else 0
            forest_predictions.append(tree_prediction_numeric)

        # Determine the final prediction based on majority voting
        final_prediction = 'Show' if sum(forest_predictions) > num_trees / 2 else 'No Show'

        print(f"\nThe Random Forest predicts that the customer will likely '{final_prediction}' for the appointment on {day_input} at {time_input}:00.")
