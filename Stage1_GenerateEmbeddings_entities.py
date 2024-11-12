### In this script, we generate embeddings for each of the newspaper articles in the dataset.
### Here, we use a SentenceTransformer model to find the representations of the articles.
### Note - due to the large size of the dataset, we process the data in chunks to avoid memory issues.

# Import necessary libraries
import pandas as pd  
import os 
import numpy as np  
from tqdm.notebook import tqdm_notebook 
import pickle  
from sentence_transformers import SentenceTransformer  

import spacy

### GPU setup for spacy
spacy.require_gpu()
### Load a large Spacy model for increased accuracy when identifying entities
nlp = spacy.load("en_core_web_trf", disable=['lemmatizer', 'attribute_ruler', 'parser']) 
print(nlp.pipe_names) # ensure that the unecessary components are disabled for time efficiency

# Load the pre-trained sentence transformer model (for text similarity/embeddings)
# We used the 'all-mpnet-base-v2' model
model = SentenceTransformer('all-mpnet-base-v2', device='cuda')  # Use CUDA (GPU) if available for faster processing

# Define columns to keep after filtering
columns_2_keep = ['urn', 'publication_date', 'unit_text', 'title', 'section']

# Path to the newspaper dataset
# in our case, we analyzed separate csvs with the articles for each newspaper analyzed
paper_path = r"E:\The Washington Post_1069201.csv" # example csv

# Function to filter the dataset by removing unwanted titles and date ranges
def filter_newspaper(df, outlet, start_date, end_date):
    """
    We manually identified several types of articles for each newspaper that we wanted to exclude from the analysis.
    These articles include corrections, obituaties and other pieces deemed not relevant for our study.
    We recommend adapting such filters to the specific needs of your project.
    
    Args:
    df (pd.DataFrame): The newspaper DataFrame.
    outlet (str): The outlet name which we gave each newspaper ('Wpost', 'NYT', 'LAT').
    start_date (str): Start date for filtering (format 'YYYY-MM-DD').
    end_date (str): End date for filtering (format 'YYYY-MM-DD').

    """
    # Remove rows where the 'title' is null
    df = df.dropna(subset=['title'])

    # Define titles to remove based on the outlet - these were manually identified for each newspaper
    if outlet == 'Wpost':
        titles_to_remove = ['The Talk Shows', 'CORRECTION', 'Correction']
    elif outlet == 'NYT':
        titles_to_remove = ['Paid Notice: Deaths', 'Corrections', 'Paid Notice: Memorials', 'Wedding']
    elif outlet == 'LAT':
        titles_to_remove = ['FOR THE RECORD', 'For the Record', 'For the record']
    else:
        print("Outlet not recognized")
        titles_to_remove = ['CORRECTIONS & CLARIFICATIONS']
    
    # Filter out unwanted titles
    df = df[~df['title'].str.contains('|'.join(titles_to_remove))]

    # Convert the 'publication_date' column to datetime
    df['publication_date'] = pd.to_datetime(df['publication_date'])
    
    # Filter rows based on date range
    # we used this to ensure the consistency of our time frame between the newspapers
    df = df[(df['publication_date'] >= start_date) & (df['publication_date'] < end_date)]
    
    # Combine 'title' and 'unit_text' to form a single text document
    df['doc'] = df['title'] + ' . ' + df['unit_text']

    # Remove rows where 'doc' is null (if both 'title' and 'unit_text' are empty)
    df = df.dropna(subset=['doc'])

    return df  # Return the filtered DataFrame


# Function to limit a document to a specific word count
def cut_doc(doc, wordlimit=200):
    cuts = doc.split()[:wordlimit]  # Split the document into words and cuts
    return ' '.join(cuts)  



# Function to extract named entities (NER) from a document using spaCy
def extract_ents(doc):

    doc = nlp(doc)  # Process the document with the spaCy model
    return list(doc.ents)  # Return the named entities


### Date range for filtering the newspaper dataset
FIRST = '1996-01-01'
LAST = '2017-01-01'

# Set the output directory for saving the emebbeding chunks
OUTDIR = r"E:\outputs"

# Newspaper identifier (used in file names)
newspaper = 'WPOST' # for example, here we used 'WPOST' for The Washington Post

# Columns to retain in the output DataFrame
rel_cols = ['urn', 'publication_date']

### Process the dataset in chunks, we used this since we were dealing with huge csv files
CHUNK_SIZE = 250  

# Change the working directory to the output directory
os.chdir(OUTDIR)

# Loop over the dataset in chunks
for i, chunk in tqdm_notebook(enumerate(pd.read_csv(paper_path, chunksize=CHUNK_SIZE))):
    # Skip processing if the chunk has already been saved
    if os.path.exists(rf'{OUTDIR}\{newspaper}_Chunk_{i}_NER.pickle'):
        print(f'Skipping chunk {i}...')  
        continue  
    
    # Filter the current chunk of the dataset - set the outlet name for the filtering of the articles
    chunk = filter_newspaper(chunk, "Wpost", FIRST, LAST)
    
    # List of document texts
    docs = list(chunk['doc'])

    # Trim the documents 
    docs = [cut_doc(doc, wordlimit=200) for doc in docs]
    
    # Extract named entities from each document
    processed = []
    for doc in tqdm_notebook(docs, total=len(docs)):
        processed.append(extract_ents(doc))

    chunk['processed'] = processed
    chunk['source'] = str(newspaper)

    chunk.to_csv(f'{newspaper}_Chunk_{i}_processed_NER.csv', index=False)
