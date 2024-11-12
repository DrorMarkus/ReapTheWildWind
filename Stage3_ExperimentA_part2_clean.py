"""
In this script, we apply the best configuration found from the random search in the first part of Experiment A to extract a list of anomaly clusters.
These are potential new media storms that need to be passed off to expert validation.
and generate a list of potential new storms for expert validation. 

In order to aid the human validation, we will extract the dates of the anomalies and an interactive tsne visualization
to allow the researcher to explore the articles published during the anomaly period and determine if a media storm is present.
"""

import numpy as np
from datetime import timedelta
import os
import pandas as pd
import pickle
from tqdm import tqdm_notebook
import plotly.express as px
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from prophet import Prophet


np.random.seed(17) ## set the seed for reproducibility

# --------------------------------------------------------------
# DETERMINE AT THE OUTSET THE BEST HYPERPARAMETER CONFIGURATION 
# IDENTIFIED IN THE FIRST PART OF THE EXPERIMENT
# --------------------------------------------------------------

# Researchers can input their best hyperparameter configuration here:
INTERVAL_WIDTH = 0.3
CHANGEPOINT_RANGE = 0.48
CHANGEPOINT_PRIOR = 0.25

# --------------------------------------------------------------
# LOADING THE DISPERSION SIGNALS AND SEED STORMS
# --------------------------------------------------------------

DISPERSION_CSV = 'dispersion_signals.csv' # replace with user's data file
STORMS_CSV = 'seed_storms.csv' # replace with user's seed list

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
# RUN THE ANOMALY DETECTION USING THE BEST CONFIGURATION
# --------------------------------------------------------------

def create_prophet_series(prophetdf, model, date_col='publication_date'):
    ## create copy with only the rel columns
    prophet_series = prophetdf[[date_col, model]].copy()
    prophet_series.rename(columns={date_col:'ds', model:'y'}, inplace=True)
    prophet_series = prophet_series[['ds','y']]
    
    return prophet_series


def fit_predict_model(dataframe, interval_width, SEASONALITY_MODE, CHANGEPOINT_RANGE, CHANGEPOINT_PRIOR):
    m = Prophet(daily_seasonality = False, yearly_seasonality = True, weekly_seasonality = True,
                seasonality_mode = SEASONALITY_MODE, changepoint_prior_scale=CHANGEPOINT_PRIOR,
                interval_width = interval_width, changepoint_range=CHANGEPOINT_RANGE) 
    m = m.fit(dataframe)
    np.random.seed(17) ## set the seed
    forecast = m.predict(dataframe)
    forecast['fact'] = dataframe['y'].reset_index(drop = True)
    return forecast

def detect_anomalies(forecast):
    forecasted = forecast[['ds','trend', 'yhat', 'yhat_lower', 'yhat_upper', 'fact']].copy()

    forecasted['anomaly'] = 0
    forecasted.loc[forecasted['fact'] > forecasted['yhat_upper'], 'anomaly'] = 1
    forecasted.loc[forecasted['fact'] < forecasted['yhat_lower'], 'anomaly'] = -1

    #anomaly importances
    forecasted['importance'] = 0
    forecasted.loc[forecasted['anomaly'] ==1, 'importance'] = \
        (forecasted['fact'] - forecasted['yhat_upper'])/forecast['fact']
    forecasted.loc[forecasted['anomaly'] ==-1, 'importance'] = \
        (forecasted['yhat_lower'] - forecasted['fact'])/forecast['fact']
    
    return forecasted


def return_anomalies(anomaly_df, signal):
    if signal == 'CoefficientOfVariance':
        print('Running on Coefficient of Variance, so looking for peak...')
    else:
        temp = anomaly_df[anomaly_df['anomaly'] == -1] # for traces looking for valleys
    
    temp[f'{signal}_fact'] = temp['fact']
    temp[f'{signal}_importance'] = temp['importance']
    temp = temp[['ds',f'{signal}_fact',f'{signal}_importance']]
    temp = temp.set_index(['ds'])
    
    return temp

## convert all nulls to ones
def convert_2_binary(anomaly_df):
    ## then convert all non-zeros to 1
    anomaly_df = anomaly_df.notnull().astype(int)
     ## first convert nans to zeros
    anomaly_df = anomaly_df.fillna(0)
    return anomaly_df

### simplify df
def convert_to_easyform(anomaly_df):
    new_df = pd.DataFrame()
    new_df.index = anomaly_df.index
    for col in list(anomaly_df):
        if '_fact' in col:
            new_df[f'{col.split("_")[1]}_Anomalies'] = anomaly_df[col]      
    return new_df

### Create single function for the anomaly detection:

def run_anomaly_detection(raw_signals, INTERVAL_WIDTH, SEASONALITY_MODE,
                  CHANGEPOINT_RANGE, CHANGEPOINT_PRIOR):
    
    prophetdf = raw_signals.copy()
    prophetdf['publication_date'] = ROLLING.index
    
    ## create signal df:
    signaldf = pd.DataFrame()
    signaldf['ds'] = raw_signals.index
    
    for signal in list(ROLLING):
        if signal not in ['year', 'publication_date']:
            ## convert columns for prophet:
            prophet_series = create_prophet_series(prophetdf, signal)
            ## fit the prophet model
            pred = fit_predict_model(prophet_series, INTERVAL_WIDTH, SEASONALITY_MODE, CHANGEPOINT_RANGE, CHANGEPOINT_PRIOR)
            ## detect anomalies on the signal:
            anoms = detect_anomalies(pred)
            ## filter the previous df to only include the anomalies
            anoms = return_anomalies(anoms, signal)
            signaldf = pd.merge(signaldf, anoms, on='ds', how='outer')

    signaldf = signaldf.set_index('ds')
    signaldf = convert_2_binary(signaldf)
    signaldf = convert_to_easyform(signaldf)
    
    return signaldf

## run the anomaly detection and return df with anomalies per signal
signaldf = run_anomaly_detection(ROLLING, INTERVAL_WIDTH, 'multiplicative',
                  CHANGEPOINT_RANGE,CHANGEPOINT_PRIOR)


# --------------------------------------------------------------
# IDENTIFY THE ANOMALY CLUSTERS/MEDIA STORM CANDIDATES
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

### Run the functions and find the anomaly clusters
clusterings = create_clusterdf(STORMS, signaldf)
majority_clusters = find_signal_cluster_dates('majority_periods', clusterings)

# --------------------------------------------------------------
# LINK ANOMALY CLUSTERS TO SEED STORMS
# --------------------------------------------------------------
## We want human validation to focus on new, unidentified media storms
## Thus, we link the clusters to the seed storms where relevant
## We added a flexibility in the date identification due to the nature both of media coverage
## and the fact that we are utilizing a rolling mean for the dispersion signals


## Initializing the empty lists
CORRESPONDING_STORMS = []
CORRESPONDING_STARTS = []
CORRESPONDING_ENDS = []

for i, cluster in majority_clusters.iterrows(): # Run over all media storm candidates
    ## Run through the seed storms
    for ii, storm in tqdm_notebook(STORMS.iterrows()):
        STORM_DATE = storm['start_date']
        
        # Check if the storm date falls within the cluster
        if cluster['START'] <= STORM_DATE <= cluster['END']:
            CORRESPONDING_STORMS.append(storm.storm_label)
            CORRESPONDING_STARTS.append(cluster.START)
            CORRESPONDING_ENDS.append(cluster.END)
            break
        
        # Check if the storm date + 1 day falls within the cluster (1 day tolerance forward)
        elif cluster['START'] <= STORM_DATE + timedelta(days=1) <= cluster['END']:
            CORRESPONDING_STORMS.append(storm.storm_label)
            CORRESPONDING_STARTS.append(cluster.START)
            CORRESPONDING_ENDS.append(cluster.END)
            break
        
        # Check if the storm date + 2 days falls within the cluster (2 days tolerance forward)
        elif cluster['START'] <= STORM_DATE + timedelta(days=2) <= cluster['END']:
            CORRESPONDING_STORMS.append(storm.storm_label)
            CORRESPONDING_STARTS.append(cluster.START)
            CORRESPONDING_ENDS.append(cluster.END)
            break
        
        # Check if the storm date + 3 days falls within the cluster (3 days tolerance forward)
        elif cluster['START'] <= STORM_DATE + timedelta(days=3) <= cluster['END']:
            CORRESPONDING_STORMS.append(storm.storm_label)
            CORRESPONDING_STARTS.append(cluster.START)
            CORRESPONDING_ENDS.append(cluster.END)
            break
        
        # Check if the storm date - 1 day falls within the cluster (1 day tolerance backward)
        elif cluster['START'] <= STORM_DATE - timedelta(days=1) <= cluster['END']:
            CORRESPONDING_STORMS.append(storm.storm_label)
            CORRESPONDING_STARTS.append(cluster.START)
            CORRESPONDING_ENDS.append(cluster.END)
            break


#### Save as new dataframe 
temp = pd.DataFrame({'SEED STORM':CORRESPONDING_STORMS,
                'START':CORRESPONDING_STARTS,
            'END':CORRESPONDING_ENDS})


#### merge with the anomaly clusters df

majority_clusters = pd.merge(majority_clusters, temp, on=['START','END'], how='left')
majority_clusters = majority_clusters[majority_clusters['LENGTH']>1] # keep only cluster with more than 1 day of anomaly

majority_clusters.to_csv('majority_clusters.csv', index=False) # save the clusters for human validation

print('Found {} potential new media storms for human validation.'.format(len(majority_clusters)))

# --------------------------------------------------------------
# CREATE THE INTERACTIVE TSNE VISUALIZATIONS FOR EACH ANOMALY CLUSTER
# --------------------------------------------------------------
## We will based the TSNE visualization on the LLM-generated embeddings of the articles

### Load the LLM embeddings
### For our meta data, need the unique identifier, the publication date and title of each article
def load_embeddings(metadata_filename, embeddings_pickle_filename, directory):
    """
    Load embeddings and associated dataframe from user-specified directory and files.
    """

    # Load the CSV file with the unique identifiers and dates
    csv_path = os.path.join(directory, metadata_filename)
    df = pd.read_csv(csv_path)

    # Load the Pickle file with embeddings
    pickle_path = os.path.join(directory, embeddings_pickle_filename)
    with open(pickle_path, "rb") as input_file:
        embeddings = pickle.load(input_file)

    # Attach the embeddings to the dataframe
    df['embedding'] = embeddings

    # Drop duplicates based on a unique identifier, like 'urn' (can be customized)
    df = df.drop_duplicates(subset='urn')

    df['publication_date'] = pd.to_datetime(df['publication_date'])
    df = df.sort_values(by="publication_date")  

    return df


EMBEDDING_DIR = # User specifies the directory
CSV_FILENAME = # User specifies the CSV file
EMBEDDINGS_FILENAME = # User specifies the Pickle file

# Load embeddings and dataframe
df = load_embeddings(CSV_FILENAME, EMBEDDINGS_FILENAME, EMBEDDING_DIR)

#### COVARIANCE FUNCTIONS

def create_subdf(date_, raw_df, window=7):
    ## set event, find 7 day period after the event
    start = date_ - timedelta(days=2)
    end_ = date_ + timedelta(days=window)
    ## create subset for the time period
    sub_df = raw_df[(raw_df['publication_date'] >= start) & (raw_df['publication_date'] <= end_)]
    return sub_df

def create_matrix(subset, col='embedding', normalized=False):
    matrix_ = np.matrix([np.array(x) for x in list(subset[col])])
    if normalized == False:
        return matrix_
    elif normalized == True:
        normalized_ = matrix_ - np.mean(matrix_, axis=0)
        return normalized_

def create_covariate_matrix(matrix):
    cov_ = np.cov(matrix, rowvar=False, bias=False)
    return cov_

# The code creates a new column 'Expert_Label' in the 'majority_clusters' DataFrame.
# If a row has a value in the 'SEED STORM' column, that value is used in 'Expert_Label'.
# If the 'SEED STORM' value is missing (NaN), it assigns a new candidate label 
# like '1_candidate', '2_candidate', etc., based on the row index.
# The fillna() method ensures that missing values are filled with these candidate labels.

majority_clusters['Expert_Label'] = majority_clusters['SEED STORM'].fillna(
    pd.Series([f'{i+1}_candidate' for i in majority_clusters.index], index=majority_clusters.index))

### create function to take date, length, and find subdf with covariance
def find_cov_per_anomaly(start_date, duration, df):
    TEMP = create_subdf(start_date, df, window=duration)
    print(len(TEMP))
    matrix_ = create_matrix(TEMP, col='embedding', normalized=True)
    return create_covariate_matrix(matrix_)

def find_embeddings_per_anomaly(start_date, duration, df):
    TEMP = create_subdf(start_date, df, window=duration)
    print(len(TEMP))
    matrix_ = create_matrix(TEMP, col='embedding', normalized=True)
    return matrix_, list(TEMP['title']), list(TEMP['publication_date'])

### Put it all together to run over the anomaly clusters and output tsnes

def tsne_storm(start_date, storm_name, duration, df, filename, outdir):
    ## create matrix and labels
    cov_mat, Labels, Dates = find_embeddings_per_anomaly(start_date, duration, df)
    
    ## format for tsne
    X = pd.DataFrame(cov_mat)
    XMETA = X.copy()
    XMETA['Label'] = Labels
    XMETA['Date'] = Dates
    
    # Standardising the values
    X_std = StandardScaler().fit_transform(X)
    
    # Invoking the t-SNE method
    tsne = TSNE()
    tsne_results = tsne.fit_transform(X_std) 
    
    DATA = pd.DataFrame(X_std)
    DATA['Label'] = Labels
    DATA['Date'] = [str(x).split('T')[0].split(' ')[0] for x in Dates]
    
    features = DATA.loc[:, DATA.columns != 'Label']

    tsne = TSNE(n_components=2, random_state=0, early_exaggeration=1, perplexity=50)
    projections = tsne.fit_transform(X_std)
    

    fig = px.scatter(
        projections, x=0, y=1,
        labels=DATA['Label'], hover_data={'Date':DATA.Date, 'Title':DATA.Label},
        title=f'Date {start_date} \n Label: {storm_name}', width=800, height=900)
    
    fig.update_layout(
    hoverlabel=dict(
        bgcolor="white",
        font_size=15,
        font_family="Rockwell"
    ))
    fig.show()
    
    os.chdir(outdir)
    fig.write_html(f'{filename}.html')
    
OUTDIR = # User specifies the output directory

for i,r in tqdm(majority_clusters.iterrows()):
    print(f'Starting on {r["Expert_Label"]}..')
    SS = r.START
    EE = r.END
    delta = EE - SS
    if '.' in str(r['Expert_Label']):
        filename = f"TSNE_{r['Expert_Label'].split('.')[0]}_{delta.days}days"
    else:
        filename = f"TSNE_{r['Expert_Label']}_{delta.days}days"
    tsne_storm(r.START, r['Expert_Label'], delta.days, df, filename, OUTDIR)