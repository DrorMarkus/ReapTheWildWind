# In this script, we run the first section of the "In Period" Experimental Setup.
# Essentially, we begin with a seed list of storms that we wish to use to calibrate the anomaly detection on the media dispersion signals.
# Note - in the first round, we utilized the initial seed list adopted from Boydstun et al., (2014). However, as the rounds continue, more storms are added to the initial seed following human validation.
# We then run a random search to ascertain the optimal parameters for the anomaly detection, using precision and recall scores.
# In our experiments, we prioritized recall.
# The output of this script is a csv with the random search results for researchers to examine and select hyperparameter configuration for the anomaly detection.DeprecationWarning

## Variables to designate: dispersion signals csv, seed storm list, output csv



## Imports
import pandas as pd
import numpy as np
from tqdm import tqdm
from prophet import Prophet
from sklearn.model_selection import ParameterSampler
from datetime import timedelta

np.random.seed(17) ## set the seed for reproducibility

# --------------------------------------------------------------
# LOADING THE DISPERSION SIGNALS AND SEED STORMS
# --------------------------------------------------------------

DIR = r"/Users/dromar/Downloads/ReapWildWind_ReplicationMaterials" # user replace with their directory
DISPERSION_CSV = os.path.join(DIR, 'MediaDispersionSignals.csv') 
STORMS_CSV = os.path.join(DIR, 'SeedStorms.csv') 

## Loading the dispersion signals
df = pd.read_csv(DISPERSION_CSV,
                 parse_dates=[0], index_col=[0]) # replace with user's data file

## create rolling means of the signals:
ROLLING = df.copy()
ROLLING = ROLLING.rolling(7).mean()
ROLLING['year'] = ROLLING.index.year

## Load the seed list of storms - this csv needs the list of storms and their start and end dates
STORMS = pd.read_csv(STORMS_CSV) # replace with user's seed list
STORMS['start_date'] = pd.to_datetime(STORMS['start_date'])
STORMS['end_date'] = pd.to_datetime(STORMS['end_date'])
SEED_STORM_LIST = list(STORMS['storm_label'])

# print(STORMS['storm_label'].value_counts()) # can examine to ensure no duplicates, etc.

## note - in ensuing iterations, can continue to add storm lists to the seed list:
# STORMS = pd.concat([STORMS, NEW_STORMS]) # replace with user's new storms list

# --------------------------------------------------------------
# ANOMALY DETECTION FUNCTIONS
# --------------------------------------------------------------


# Prepare DataFrame for Prophet model input requirements
def create_prophet_series(prophetdf, model, date_col='publication_date'): # set date column
    # Select required columns and rename for Prophet model
    return prophetdf[[date_col, model]].rename(columns={date_col:'ds', model:'y'})

# Fit Prophet model to data and generate forecast
def fit_predict_model(dataframe, interval_width, seasonality_mode, changepoint_range, changepoint_prior):
    # Define and fit Prophet model
    model = Prophet(daily_seasonality=False, yearly_seasonality=True, weekly_seasonality=False,
                    seasonality_mode=seasonality_mode, changepoint_prior_scale=changepoint_prior,
                    interval_width=interval_width, changepoint_range=changepoint_range).fit(dataframe)
    # Generate forecast
    forecast = model.predict(dataframe)
    # Add actual data to forecast DataFrame
    forecast['fact'] = dataframe['y'].reset_index(drop=True)
    return forecast

# Identify anomalies in the forecasted data
def detect_anomalies(forecast):
    # Select required columns
    forecasted = forecast[['ds','trend', 'yhat', 'yhat_lower', 'yhat_upper', 'fact']].copy()
    # Initialize 'anomaly' and 'importance' columns
    forecasted['anomaly'] = 0
    forecasted['importance'] = 0
    # Define anomaly conditions for both upper and lower bounds
    forecasted.loc[forecasted['fact'] > forecasted['yhat_upper'], 'anomaly'] = 1
    forecasted.loc[forecasted['fact'] < forecasted['yhat_lower'], 'anomaly'] = -1
    # Define importance conditions for both upper and lower anomalies
    forecasted.loc[forecasted['anomaly'] == 1, 'importance'] = (forecasted['fact'] - forecasted['yhat_upper'])/forecast['fact']
    forecasted.loc[forecasted['anomaly'] == -1, 'importance'] = (forecasted['yhat_lower'] - forecasted['fact'])/forecast['fact']

    return forecasted

# Filter anomalies from the forecasted data
def return_anomalies(anomaly_df, signal):
    # Extract only the lower anomalies
    temp = anomaly_df[anomaly_df['anomaly'] == -1]
    # Select and rename the required columns
    temp = temp[['ds', 'fact', 'importance']].set_index('ds')
    temp.columns = [f'{signal}_fact', f'{signal}_importance']

    return temp

# Convert DataFrame values to binary form
def convert_to_binary(anomaly_df):
    # Convert non-null values to 1 and null values to 0
    return anomaly_df.notnull().astype(int).fillna(0)

# Simplify the anomaly DataFrame
def simplify_df(anomaly_df):
    new_df = pd.DataFrame(index=anomaly_df.index)
    # Extract only the anomalies
    for col in anomaly_df.columns: # note, the columns here correspond to the signals
        if '_fact' in col:
            new_df[f'{col.split("_")[1]}_Anomalies'] = anomaly_df[col]

    return new_df

# Run entire anomaly detection process
def run_anomaly_detection(raw_signals, interval_width, seasonality_mode, changepoint_range, changepoint_prior):
    # Make a copy of raw_signals DataFrame and add a 'publication_date' column
    prophetdf = raw_signals.copy()
    prophetdf['publication_date'] = ROLLING.index

    # Initialize signaldf DataFrame
    signaldf = pd.DataFrame()
    signaldf['ds'] = raw_signals.index

    # Run the entire process for each signal
    for signal in raw_signals.drop(columns=['year']):
        prophet_series = create_prophet_series(prophetdf, signal)
        pred = fit_predict_model(prophet_series, interval_width, seasonality_mode, changepoint_range, changepoint_prior)
        anoms = detect_anomalies(pred)
        anoms = return_anomalies(anoms, signal)

        # Merge the returned anomalies with signaldf
        signaldf = signaldf.merge(anoms, on='ds', how='outer')

    # Set 'ds' as index
    signaldf.set_index('ds', inplace=True)

    # Convert signaldf to binary
    signaldf = convert_to_binary(signaldf)

    # Simplify signaldf
    signaldf = simplify_df(signaldf)

    return signaldf ### output = df with only the binary anomalies - date index and signal columns
   


# --------------------------------------------------------------
# EVALUATING THE RESULTS OF THE ANOMALY DETECTION
# --------------------------------------------------------------



def find_majority_vote(accuracy_scores):
    """
    Add a new column 'majority' to the accuracy scores dataframe.
    The new column contains the mode of anomalies of the given signals

    Parameters:
    accuracy_scores (DataFrame): A dataframe that contains accuracy scores information
    
    Returns:
    DataFrame: A dataframe with the 'majority' column added
    """
    signals = ['entitiesEntropy_Anomalies', 'NEAT_Anomalies', '100topics_Anomalies']
    accuracy_scores['majority'] = accuracy_scores[signals].mode(axis=1)[0]
    
    return accuracy_scores

def create_clusterdf(storms, anomaly_binary_df):
    """
    Create a dataframe for clustering anomalies and consensus. 
    Instead of identifying and counting single anomaly days, we want to identify media storms - clusters of anomalies.
    
    Parameters:
    storms (List): A list of storms
    anomaly_binary_df (DataFrame): A dataframe with binary daily anomaly information
    
    Returns:
    DataFrame: A dataframe with cumulative counts of continuous anomalies and consensus
    """
    clusterings = anomaly_binary_df.copy()

    # Add columns for majority vote and consensus
    clusterings = find_majority_vote(clusterings)

    # Count cumulative anomaly periods for each signal - 
    # we do this for all signals and not just majorities in case researcher wants to examine further the divergence
    for signal in clusterings.columns:
        # Running count of continuous anomalies
        y = clusterings[signal]
        signal_prefix = signal.split("_")[0]
        clusterings[f'{signal_prefix}_periods'] = y * (y.groupby((y != y.shift()).cumsum()).cumcount() + 1)

        # Add the next day's continuous count value
        # This will be used to discover where each cluster begins and ends
        clusterings[f'{signal_prefix}_d+1'] = clusterings[f'{signal_prefix}_periods'].shift(-1)

    return clusterings

def find_signal_cluster_dates(signal, cluster_df):
    """
    Find start and end dates of the anomaly clusters for a given signal
    
    Parameters:
    signal (str): The name of the signal
    cluster_df (DataFrame): A dataframe with cluster information for all signals
    
    Returns:
    DataFrame: A dataframe with the start and end dates of the anomaly clusters for the given signal
    """
    # Filter to only include anomalies
    anomalies = cluster_df[cluster_df[signal] > 0].fillna(0)

    # Initialize lists for cluster starts, ends and lengths
    starts, ends, lengths = [], [], []

    signal_prefix = signal.split('_')[0]

    # Find start and end dates for each cluster
    for index, row in anomalies.iterrows():
        if row[f'{signal_prefix}_periods'] == 1:
            starts.append(index)
        if row[f'{signal_prefix}_d+1'] == 0:
            ends.append(index)
            lengths.append(row[f'{signal_prefix}_periods'])

    # Create a dataframe with start and end dates of the anomaly clusters
    clusters = pd.DataFrame({'START': starts, 'END': ends, 'LENGTH': lengths})
    
    return clusters

# --------------------------------------------------------------
# EVALUATING THE PRECISION
# --------------------------------------------------------------
## We want to run over all the anomaly clusters and see which ones match the seed storms.
def match_anomaly_clusters_to_storms(storm_df, cluster_dates_df):
    """
    Check for matches between anomaly clusters and storms.
    
    Parameters:
    storm_df (DataFrame): A dataframe containing storm data
    cluster_dates_df (DataFrame): A dataframe with start and end dates of anomaly clusters

    """
    # Keep clusters longer than 1 day
    cluster_dates_df = cluster_dates_df[cluster_dates_df['LENGTH'] > 1]
    
    # Initialize counter for matched clusters
    matched_clusters = 0 

    # For each cluster, check if any of the storm dates fall within the cluster duration
    for _, cluster in cluster_dates_df.iterrows(): # iterate over the anomaly clusters
        for _, storm in storm_df.iterrows(): # iterate over the seed storms
            if cluster['START'] <= storm['start_date'] <= cluster['END']:
                matched_clusters += cluster['LENGTH'] # try to weight clusters by length, focus on longer anomalies
                break
            elif (cluster['START'] <= storm['start_date'] + timedelta(days=1) <= cluster['END'] or 
                  cluster['START'] <= storm['start_date'] - timedelta(days=1) <= cluster['END']):
                matched_clusters += cluster['LENGTH']
                break
                
    return matched_clusters

def find_precision_metrics(storm_df, anomaly_binary_df):
    """
    Calculate precision metrics for anomaly detection.
    """
    # Create clusters of consecutive anomalies
    clusterings = create_clusterdf(storm_df, anomaly_binary_df)

    # Initialize the relevant signal for majority
    signals = ['majority']

    matches_dict = {}

    # For each signal, find the start and end dates, and see if the timespan coincides with a storm
    for signal in signals:
        cluster_dates = find_signal_cluster_dates(signal, clusterings)
        total_anomalies = clusterings[signal].sum()

        matches = match_anomaly_clusters_to_storms(storm_df, cluster_dates)
        matches_dict['majority'] = matches / total_anomalies
    
    # Add the total number of clusters and mean number of anomalies to the dictionary
    matches_dict['total clusters'] = clusterings['majority'].sum()
    matches_dict['mean anomalies'] = anomaly_binary_df.sum().mean()

    return matches_dict

# --------------------------------------------------------------
# EVALUATING THE RECALL
# --------------------------------------------------------------
## Run over all the seed storms and see which ones match an anomaly cluster.

# Specify the number of days to check after each storm to see if there are anomalies
WINDOW = 3

def create_recall_df(storm_df, anomalies_df, window=WINDOW):
    """
    Creates dataframe to compare the seed storms to the anomaly clusters
    """
    # Create clusters of consecutive anomalies
    clusterings = create_clusterdf(storm_df, anomalies_df)

    # Initialize list of signal columns - the anomaly columns and majority vote column.
    signals = ['entitiesEntropy_Anomalies', 'all-mpnet-base-v2_Anomalies', 'NEAT_Anomalies', '100topics_Anomalies', 'majority']
    # Create a new dataframe based on storm_df
    recall_df = pd.DataFrame()
    recall_df['storm_label'] = storm_df['storm_label']
    recall_df.set_index('storm_label', inplace=True)
    
    # For each signal, find the start, end dates, and check for storm connection
    for signal in signals:
        # Find the start, end dates and lengths
        cluster_dates = find_signal_cluster_dates(signal, clusterings)

        # For each cluster row, run over the storms
        for index, cluster in cluster_dates.iterrows(): 
            for _, storm in storm_df.iterrows(): 
                # Create a list of dates based on the window
                dates = [(storm.date + timedelta(days=n)) for n in range(window)] 

                # Check if each date is found within the start and end of the current cluster
                if any(cluster['START'] < date < cluster['END'] for date in dates):
                    # If found, add cluster number and storm topic to the dataframe
                    recall_df.loc[storm.topic, f'{signal}_cluster'] = index

    return recall_df

def find_recall_metrics(storm_df, anomalies_df, window=WINDOW):
    """
    Calculate recall metrics for anomaly detection
    
    Parameters:
    storm_df (DataFrame): A dataframe with storm data
    anomalies_df (DataFrame): A dataframe with binary anomaly information

    """
    # Create the accuracy df
    accuracy_df = create_recall_df(storm_df, anomalies_df, window)
    
    # Columns to count storms
    cols = ['majority_cluster']
    
    # Find the total number of storms
    total_storms = storm_df.shape[0]
    
    # Count the number of detected storms for the majority signal
    detected_storm_counts = accuracy_df[cols].count()

    # Calculate recall percentage for the majority signal
    recall_majority = (detected_storm_counts / total_storms) * 100
    
    return recall_majority

# --------------------------------------------------------------
# RUNNING THE RANDOM HYPERPARAMETER SEARCH
# --------------------------------------------------------------
## Combining all functions into a single iteration. Use random search to optimize the anomaly detection hyperparameters.

def run_iteration(storm_df, seed_storm_list, dispersion_signals, interval_width, 
                  seasonality_mode, changepoint_range, changepoint_prior):
    """
    This function runs an iteration of anomaly detection and calculates the precision and recall for a list of storms and signals.
    
    Parameters:
    - storm_df: DataFrame containing storm data.
    - seed_storm_list: List of seed storms.
    - dispersion_signals: Raw signals data (dispersion signals).
    - interval_width: Interval width for anomaly detection model.
    - seasonality_mode: Seasonality mode for anomaly detection model.
    - changepoint_range: Changepoint range for anomaly detection model.
    - changepoint_prior: Changepoint prior for anomaly detection model.
    
    Returns:
    - precision_majority: Precision score for majority signal.
    - recall_majority: Recall score for majority signal.
    - total_clusters: Total number of anomaly clusters detected.
    - mean_anomalies: Mean number of anomalies across signals.
    """
    
    # Run anomaly detection
    signal_df = run_anomaly_detection(dispersion_signals, interval_width, seasonality_mode,
                                      changepoint_range, changepoint_prior)
    
    # Filter storm_df to only include storms that are in the relevant_storm_list
    subset_storm_df = storm_df[storm_df['storm_label'].isin(seed_storm_list)]
    
    # Calculate recall metrics (now only returning recall_majority)
    recall_majority = find_recall_metrics(subset_storm_df, signal_df, window=WINDOW)
    
    # Calculate precision metrics (only the relevant precision metric and additional metrics)
    precision_dict = find_precision_metrics(subset_storm_df, signal_df)
    precision_majority = precision_dict['majority'] * 100
    total_clusters = precision_dict['total clusters']
    mean_anomalies = precision_dict['mean anomalies']
    
    # Return only the relevant metrics
    return precision_majority, recall_majority, total_clusters, mean_anomalies


### RUNNING THE RANDOM SEARCH

## define the ranges for the random search space
param_grid = {
    'interval_width': np.around(np.linspace(0.3, .9, 40), 2),
    'changepoint_prior': np.around(np.linspace(0.01, 0.9, 70), 2),
    'changepoint_range': np.around(np.linspace(0.2, 0.9, 25), 2)
}

## set the number of iterations:
n_iter= 4000

# Initialize  a dictionary to hold the hyperparameter values and results
simulation_dict = {
    'interval_width': [],
    'changepoint_prior': [],
    'changepoint_range': [],
    'Precision_Majorities': [],
    'Recall_Majorities': [],
    'Num_Clusters': [],
    'Num_Anomalies': []
}

random_grid = ParameterSampler(param_grid, n_iter=n_iter, random_state=17)


# Perform the random search
for i, params in enumerate(tqdm(random_grid, desc='Random Search Iterations')):
    interval_width = params['interval_width']
    changepoint_prior = params['changepoint_prior']
    changepoint_range = params['changepoint_range']
    rel_storms = SEED_STORM_LIST
    
    # Run the simulation
    results = run_iteration(STORMS, rel_storms, ROLLING, interval_width, 'multiplicative', changepoint_range, changepoint_prior)
    
    # Unpack the relevant results
    precision_majority, recall_majority, num_clusters, num_anomalies = results
    
    # Append the results to the dictionary
    simulation_dict['interval_width'].append(interval_width)
    simulation_dict['changepoint_prior'].append(changepoint_prior)
    simulation_dict['changepoint_range'].append(changepoint_range)
    
    simulation_dict['Precision_Majorities'].append(precision_majority)
    simulation_dict['Recall_Majorities'].append(recall_majority)
    simulation_dict['Num_Clusters'].append(num_clusters)
    simulation_dict['Num_Anomalies'].append(num_anomalies)
    
    # Save results every 100 iterations
    if (i + 1) % 100 == 0:
        simulation_df = pd.DataFrame(simulation_dict)
        simulation_df.to_csv(f'CalibrationResults_RandomSearch_{i+1}.csv', index=False)
        print(f'Saved results after {i+1} iterations.')
        
# Save the final DataFrame after the loop
simulation_df = pd.DataFrame(simulation_dict)
simulation_df.to_csv('CalibrationResults_RandomSearch_Final.csv', index=False)