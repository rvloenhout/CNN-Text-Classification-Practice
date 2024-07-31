#Packages required for preprocess_text
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

#Packages required for the class TextDataset
import torch
from torch.utils.data import Dataset

def preprocess_text(text):
    """
    Preprocesses the input text by performing several cleaning steps.

    Parameters:
    text (str): The text to be preprocessed.

    Returns:
    str: The preprocessed text.

    The function performs the following steps:
    - Removes special characters and digits using a regular expression that retains only alphabets and spaces.
    - Reduces multiple whitespaces to a single space.
    - Trims leading and trailing whitespaces.
    - Converts the text to lowercase for uniformity.
    - Removes punctuation by translating them to an empty string.
    - Tokenizes the text into words, removes stop words, and applies stemming to each word.
    - Joins the filtered and stemmed words back into a single string separated by spaces.

    Note:
    - The function assumes the availability of a stemmer object and a list of stop words.
    - 're' and 'string' modules should be imported as they are used for regex operations and punctuation removal respectively.
    - 'word_tokenize' function should be imported from the 'nltk.tokenize' module for tokenization.
    """

    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove multiple whitespaces
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading and trailing whitespaces
    text = text.strip()
    
    # Convert text to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove stop words and perform stemming
    words = word_tokenize(text)
    filtered_words = [stemmer.stem(word) for word in words if word not in stop_words]
    
    return ' '.join(filtered_words)

class TextDataset(Dataset):
    """
    A custom dataset class that inherits from PyTorch's Dataset class.

    Parameters:
    encodings (dict): A dictionary containing the input encodings.
    labels (list): A list of labels corresponding to the input encodings.

    The class overrides two methods from the PyTorch Dataset class:
    - __getitem__: Retrieves an encoding and its corresponding label at a given index.
    - __len__: Returns the total number of items in the dataset.

    Methods:
    __init__(self, encodings, labels): Initializes the dataset with encodings and labels.
    __getitem__(self, idx): Returns the encoding and label at the specified index.
    __len__(self): Returns the size of the dataset.

    Note:
    - The 'torch' module should be imported as it is used for tensor operations.
    - The 'Dataset' class should be imported from 'torch.utils.data'.
    """
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)