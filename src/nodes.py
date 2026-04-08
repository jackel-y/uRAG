# BASE
# ------------------------------------------------------
import numpy as np
import pandas as pd

# SKLEARN
# ------------------------------------------------------
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer

# XGBOOST
# ------------------------------------------------------
import xgboost as xgb

def feature_engineering(
        df: pd.DataFrame,
        is_training=True, 
        prev_predictions=None,
        od_columns_from_training=None
        ):
    """Feature engineering for the flight sales prediction model.
    Args:
        df (pd.DataFrame): Input DataFrame containing flight data.
        is_training (bool): Flag indicating if the data is for training or prediction.
        prev_predictions (dict): Previous predictions for lag features, if available.
        od_columns_from_training (list): List of origin-destination columns from training data.
    Returns:
        pd.DataFrame: Processed DataFrame with engineered features.
    """
    df_processed = df.copy()

    # Time-based Features
    df_processed['day_of_week'] = df_processed['departure_Date'].dt.dayofweek
    df_processed['day_of_year'] = df_processed['departure_Date'].dt.dayofyear
    df_processed['day'] = df_processed['departure_Date'].dt.day
    df_processed['month'] = df_processed['departure_Date'].dt.month
    df_processed['year'] = df_processed['departure_Date'].dt.year
    df_processed['quarter'] = df_processed['departure_Date'].dt.quarter
    df_processed['is_weekend'] = (df_processed['day_of_week'] >= 5).astype(int)
    df_processed['week_of_year'] = df_processed['departure_Date'].dt.isocalendar().week.astype(int)
    df_processed['days_x_month'] = df_processed['days_before_departure'] * df_processed['month']

    if 'OD_DXB-XXX' in df.columns:
        df_processed['OD_DXB-XXX'] = df_processed['OD_DXB-XXX'].astype(int)
    if 'OD_XXX_DXB' in df_processed.columns:
        df_processed['OD_XXX_DXB'] = df_processed['OD_XXX-DXB'].astype(int)



    df_processed['lag_1_corrected_seats_sold'] = 0.0
    df_processed['lag_2_corrected_seats_sold'] = 0.0
    df_processed['lag_7_corrected_seats_sold'] = 0.0

    if not is_training and prev_predictions is not None:
        current_day_idx = df_processed['days_before_departure'].iloc[0]

        # Use shift(1) logic for "lag" meaning previous point in time (larger days_before_departure)
        # If your training used shift(-1) for lags, adjust this accordingly
        if (current_day_idx + 1) in prev_predictions: # Prev day in sequence (e.g. 341 for 340)
            df_processed['lag_1_corrected_seats_sold'] = prev_predictions[current_day_idx + 1]
        if (current_day_idx + 2) in prev_predictions:
            df_processed['lag_2_corrected_seats_sold'] = prev_predictions[current_day_idx + 2]
        if (current_day_idx + 7) in prev_predictions:
            df_processed['lag_7_corrected_seats_sold'] = prev_predictions[current_day_idx + 7]

    return df_processed

def predictions(
    df: pd.DataFrame,
    xgb_model: xgb.XGBRegressor,
) -> pd.DataFrame:
    """Generate predictions for flight sales using the provided XGBoost model.
    Args:
        df (pd.DataFrame): Input DataFrame containing flight data.
        xgb_model (xgb.XGBRegressor): Trained XGBoost model for predictions.
    Returns:
        pd.DataFrame: DataFrame containing wide and long format predictions.
    """
    
    df['departure_Date'] = pd.to_datetime(df['departure_Date'])

    all_predictions_wide = []
    all_predictions_long = [] 
    features = ['Target','days_before_departure','OD_DXB-XXX', 'OD_XXX-DXB', 'day_of_week', 'day_of_year', 'day', 'month', 'year',
                'quarter', 'is_weekend', 'week_of_year', 'days_x_month',
                'lag_1_corrected_seats_sold', 'lag_2_corrected_seats_sold', 'lag_7_corrected_seats_sold']

    print(f"Starting predictions for {len(df)} flights.")

    # Iterate through each flight 
    for index, row in df.iterrows():
        flight_departure_date = row['departure_Date']
        flight_origin_destination_str = row['Origin_Destination']
        flight_target = row['Target']

        days_to_forecast = range(342)

        sequential_predictions_for_flight = {}
        temp_long_data = [] 

        for day_idx in reversed(days_to_forecast): # Loop from 341 down to 0
            col_name = f'{day_idx}_SeatsSold_sum'
            actual_val = row.get(col_name) # Check if an actual value exists for this specific day_idx

            current_day_seats_sold = None
            value_type = None

            if pd.notna(actual_val):
                current_day_seats_sold = actual_val
                value_type = 'actual'
            else:
                # 1. Current Day df
                current_day_input_df = pd.DataFrame({
                    'departure_Date': [flight_departure_date],
                    'Origin_Destination': [flight_origin_destination_str],
                    'Target': [flight_target],
                    'days_before_departure': [day_idx]
                })

                # 2. Apply feature engineering
                current_day_features_df = feature_engineering(
                    current_day_input_df,
                    is_training=False, 
                    prev_predictions=sequential_predictions_for_flight, 
                    od_columns_from_training=['OD_DXB-XXX','OD_XXX-DXB'] 
                )

                # 3. Ensure the feature columns are in the exact order and type as the model expects.
                processed_features_for_prediction = pd.DataFrame(columns=features)
                for col in features:
                    if col in current_day_features_df.columns:
                        processed_features_for_prediction[col] = current_day_features_df[col]
                    else:
                        processed_features_for_prediction[col] = 0.0

                # 4. Make the prediction for this single day.
                predicted_val = xgb_model.predict(processed_features_for_prediction)[0]
                current_day_seats_sold = max(0, int(np.round(predicted_val))) # Ensure non-negative and integer
                value_type = 'predicted'

            # Store the current day's calculated seats_sold (actual or predicted) in the memory dictionary.
            sequential_predictions_for_flight[day_idx] = current_day_seats_sold

            # Append data to the long format list for the current day
            temp_long_data.append({
                'departure_Date': flight_departure_date,
                'Origin_Destination': flight_origin_destination_str,
                'Target': flight_target,
                'days_before_departure': day_idx,
                'seats_sold': current_day_seats_sold,
                'value_type': value_type
            })



        # After forecasting all days for the current flight, sort the long data
        forecasted_sales_wide = {}
        for day_idx in days_to_forecast:
            # Retrieve the final, determined value (actual or predicted) for each day_idx
            forecasted_sales_wide[f'{day_idx}_SeatsSold_sum'] = sequential_predictions_for_flight.get(day_idx)
            
        temp_long_data_sorted = sorted(temp_long_data, key=lambda x: x['days_before_departure'])
        all_predictions_long.extend(temp_long_data_sorted)

        # Append the wide format results for the current flight
        all_predictions_wide.append({
            'departure_Date': flight_departure_date,
            'Origin_Destination': flight_origin_destination_str,
            'Target': flight_target,
            **forecasted_sales_wide
        })


    # Convert lists of dictionaries to DataFrames
    forecasted_output_wide = pd.DataFrame(all_predictions_wide)
    forecasted_output_long = pd.DataFrame(all_predictions_long)

    print("\n--- Prediction Process Complete ---")
    print(f"Generated {len(forecasted_output_wide)} wide-format flight predictions.")
    print(f"Generated {len(forecasted_output_long)} long-format daily predictions.")

    
    return forecasted_output_wide, forecasted_output_long