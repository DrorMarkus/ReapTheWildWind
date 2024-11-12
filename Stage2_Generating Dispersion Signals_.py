# This script consists of 2 sections representing two ways we calculated dispersion signals utilized in our experiments.

# First, we create a dispersion signal by calculating the trace in cases where we have a dense matrix (e.g., sentence embeddings) representing our article.
# This section comes after the researcher has already generated embeddings for their corpus that are saved in a pickle file (as per our research case).
# Additionally, the important metadata is a unique_id and the publication date for each article. In this way, we can generate the time series of the signal.
# The output here is a dataframe with the date and the corresponding trace value - the time series of the dispersion signal.

# Second, we added a section below which calculates entropy instead of trace. We used this specifically when working with the NER outputs as the matrices are exceedingly large (as well as sparse). T
# Thus, instead of finding the covariance and trace, we instead found the entropy of the entitiy frequencies as a single metric representing dispersion.

# User-defined variables to be set before running the script

# # Directory where the embeddings and metadata are stored
# DIR = "/path/to/your/data"  # Set the relevant directory
# # The filename for the pre-generated embeddings (without the .pickle extension)
# embedding_type = "sentence_embeddings"  # Replace with the correct filename
# # Size of the rolling window (in days)
# WINDOW = 7  # Set the desired window size
# # The column containing processed NER output (for Section 2)
# NER_PROCESSED_COLUMN = 'ner_processed'  # Replace if the column name is different
# # Output filename for the Trace results (Section 1)
# TRACE_OUTPUT_FILENAME = "trace_output.csv"  # Replace with the desired output filename
# # Output filename for the Entropy results (Section 2)
# ENTROPY_OUTPUT_FILENAME = "entropy_output.csv"  # Replace with the desired output filename
# # Path to the pre-generated TFIDF vocabulary model for Section 2 (in pickle format)
# TFIDF_MODEL_PATH = os.path.join(DIR, "2000_2017_tfidfmodel_19052023_smallest.pickle")  # Adjust as needed


########### SECTION 1 ###########
##### CALCULATING THE TRACE #####

##### Imports
import os
import pandas as pd
import pickle
import numpy as np
from datetime import datetime
from datetime import timedelta
from tqdm import tqdm_notebook

# load the embeddings and metadata
DIR = directory # set the relevant directory

embedding_type = # the filename for the pre-generated embeddings

DF_pickle = os.path.join(DIR, f"{embedding_type}.pickle")
with open(DF_pickle, "rb") as input_file:
    EMBEDS = pickle.load(input_file)

DF = pd.read_csv(os.path.join(DIR, f"{embedding_type}.csv")) # replace with user's meta data for embeddings
print(DF.shape, len(DF_pickle)) # ensure the two files correspond to each other/are of the same length
DF['embedding'] = EMBEDS # merge

### now, order by dates to create the sorted dtm
DF['publication_date'] = pd.to_datetime(DF['publication_date'])
DF = DF.sort_values(by="publication_date")

### Functions used to create running trace calculation over the time series

def create_subdf(date_, raw_df, window): 
    ## Create a subset df based on the rolling window
    ## note - rolling window of 0 = 1 day subset
    start = date_
    end_ = date_ + timedelta(days=window)
    
    ## Filter the main df based on the dates
    sub_df = raw_df[(raw_df['publication_date'] >= start) & (raw_df['publication_date'] <= end_)]
    
    return sub_df


def create_matrix(subset, col='embedding', normalized=False):
    ## Convert the subset df to a matrix, normalize if determined to be relevant
    matrix_ = np.matrix([np.array(x) for x in list(subset[col])])
    if normalized == False:
        return matrix_
    elif normalized == True:
        normalized_ = matrix_ - np.mean(matrix_, axis=0)
        return matrix_


def create_covariance_matrix(matrix):
    ## Find covariance matrix for the subset matrix
    cov_ = np.cov(matrix, rowvar=False, bias=False)
    return cov_


def find_trace(cov_matrix):
    ## Calculate the trace value for the covariance matrix
    trace = cov_matrix.trace()
    return trace

### Function to run together
def running_window(date, raw_df, window, normalized=True): 
    ### create the subdf
    sub = create_subdf(date, raw_df, window=window)
    ### create normalized matrix from vectors
    sub_ = create_covariance_matrix(create_matrix(sub, normalized=normalized))
    ### output the trace of the covariance matrix
    trace_ = find_trace(sub_)
    normalized_trace = (trace_ / len(sub)) # Can normalize if determined to be relevant
    return trace_, normalized_trace

### RUN THE TRACE FUNCTION OVER THE ARTICLES
START_DATE = DF.publication_date.min() # Set the start date for the analysis

DATES = []
TRACES = []
TRACES_NORMALIZED = []


for day in tqdm_notebook(range(len(DF))): # run over the full time-span
    DATE = START_DATE + timedelta(days=day)
    ## create subdfs, run functions
    TRACE, TRACE_NORMALIZED = running_window(DATE, DF, window=WINDOW, normalized=True)
    DATES.append(DATE)
    TRACES.append(TRACE)
    TRACES_NORMALIZED.append(TRACE_NORMALIZED)

### Create a df with the dates and trace values - the output

output = pd.DataFrame()
output.index = DATES
output['Trace'] = TRACES
output['Trace_Normalized'] = TRACES_NORMALIZED

#output.to_csv(OUTPUT_FILENAME)

########## SECTION 2 ##########
##### CALCULATING ENTROPY #####

### Additional imports
import random
import statistics
import matplotlib.pyplot as plt
from collections import Counter
import scipy
from scipy.sparse import csr_matrix
from scipy.stats import entropy
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import normalize
from math import e

### Import the data -
## The df should include the all the articles, with a list of entities for each and the publication date.
df = pd.read_csv('newspaper_entities.csv') # replace with user's data

### now, order the articles by their dates to create the sorted document-term matrix with the entities for each article
df['publication_date'] = pd.to_datetime(df['publication_date'])
df = df.sort_values(by="publication_date")

### prepare the vocab
### We utilized the tfidf scores to create a vocab dictionary in a separate script before running this
## The purpose being to focus on the most informative entities only
with open(os.path.join(DIR, "2000_2017_tfidfmodel_19052023_smallest.pickle"), "rb") as input_file: # replace with user's vocab file
    TFIDF = pickle.load(input_file)
vocab_dict = TFIDF.vocabulary_
countvectorizer = CountVectorizer(lowercase=False, min_df=0, max_df=1,
                                  tokenizer=str.split, 
                                 vocabulary= vocab_dict)

# Function to clean up document strings (entity lists) for vectorization
# This is based on our specific data after running the Spacy NER separately
def updated_clean_doc(text):
    if '[' in text:
        ## clean prefix, suffix
        text = text.split('[')[1]
        text = text.split(']')[0]
        splitted = text.split(',')
        splitted = [x.strip(' ') for x in splitted]
        splitted = [x.replace(' ','-') for x in splitted]
        text = ' '.join(splitted)
    else:
        text = text
    return text

# Function to calculate entropy over a subset of articles 
def running_entropy(date, raw_df, model=countvectorizer):
    # Step 1: Create a subset of articles based on the rolling window
    subset = create_subdf(date, raw_df, WINDOW)

    if len(subset) > 1:  # Proceed if there is more than 1 article
        # Step 2: Clean and process the articles
        docs_ = [updated_clean_doc(x) for x in subset['ner_processed']] # note - this is based on our specific data
        
        # Step 3: Apply CountVectorizer to the cleaned articles to generate a document-term matrix
        dtf = model.fit_transform(docs_)
        
        # Step 4: Calculate raw entropy
        dtfbow = pd.DataFrame(dtf.toarray(), columns=model.get_feature_names_out())
        vc_ = np.array(dtfbow.sum())  # Get term counts across all documents
        vc_probs = vc_ / sum(vc_)  # Compute probabilities for entropy calculation
        entropy_ = entropy(vc_probs, base=2)  # Calculate entropy with log base 2
        
        # Step 5: Normalize entropy by the number of articles in the subset
        normalized_entropy = entropy_ / len(subset)
    else:
        # If there's only one article, return zeros
        normalized_entropy = 0
    
    return normalized_entropy

DATES = []
ENTROPIES = []

## run over the full time-span
for day in tqdm_notebook(range(6210)): # 13 years
    DATE = START_DATE + timedelta(days=day)
    entropy_  = running_entropy(DATE, df)
    DATES.append(DATE)
    ENTROPIES.append(entropy_)


### create the df and save

output = pd.DataFrame()
output.index = DATES
output['Entropy'] = ENTROPIES

#output.to_csv(OUTPUT_FILENAME)