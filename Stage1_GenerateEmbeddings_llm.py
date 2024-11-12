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
nlp = spacy.load('en_core_web_sm')  # Set the spacy model, we used en_core_web_sm

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


# Function to split a document into sentences using spaCy sentence tokenizer
def split_sentences(doc):
    return [str(sent) for sent in nlp(doc).sents]  


# Function to compute sentence embeddings and return the average embedding for the document
def find_embeddings(doc):
    sentences = split_sentences(doc)  
    embeds_ = model.encode(sentences)  
    return np.mean(embeds_, axis=0)  


### Date range for filtering the newspaper dataset
FIRST = '1996-01-01'
LAST = '2017-01-01'

# Set the output directory for saving the emebbeding chunks
OUTDIR = r"E:\outputs"

# Newspaper identifier (used in file names)
newspaper = 'WPOST'

# Columns to retain in the output DataFrame
rel_cols = ['urn', 'publication_date']

### Process the dataset in chunks, we used this since we were dealing with huge csv files
CHUNK_SIZE = 10000  

# Change the working directory to the output directory
os.chdir(OUTDIR)

# Loop over the dataset in chunks
for i, chunk in tqdm_notebook(enumerate(pd.read_csv(paper_path, chunksize=CHUNK_SIZE))):
    # Skip processing if the chunk has already been saved
    if os.path.exists(rf'{OUTDIR}\{newspaper}_Chunk_{i}_all-mpnet-base-v2_FULL.pickle'):
        print(f'Skipping chunk {i}...')  
        continue  
    
    # Filter the current chunk of the dataset - set the outlet name for the filtering of the articles
    chunk = filter_newspaper(chunk, "Wpost", FIRST, LAST)

    # Trim the documents 
    docs = [cut_doc(doc, wordlimit=200) for doc in docs]
    
    # Compute the embeddings
    embeds = [find_embeddings(doc) for doc in tqdm_notebook(docs)]
    
    # Save in a pickle file
    with open(f"{newspaper}_Chunk_{i}_all-mpnet-base-v2_FULL.pickle", 'wb') as handle:
        pickle.dump(embeds, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Save a csv with the list of article ids (urns) to link with the embeddings later
    outdf = chunk[rel_cols]
    outdf.to_csv(f"{newspaper}_Chunk_{i}__urnkey.csv", index=False)
