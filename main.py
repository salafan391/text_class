from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import tensorflow as tf
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import random


class TextNLP:
  def __init__(self,main_path):
    self.main_path = main_path
  def preprocess_text(self,filename):
    """Returns a list of dictionaries of abstract line data.

    Takes in filename, reads its contents and sorts through each line,
    extracting things like the target label, the text of the sentence,
    how many sentences are in the current abstract and what sentence number
    the target line is.

    Args:
        filename: a string of the target text file to read and extract line data
        from.

    Returns:
        A list of dictionaries each containing a line from an abstract,
        the lines label, the lines position in the abstract and the total number
        of lines in the abstract where the line is from. For example:

        [{"target": 'CONCLUSION',
          "text": The study couldn't have gone better, turns out people are kinder than you think",
          "line_number": 8,
          "total_lines": 8}]
    """
    with open(self.main_path+filename) as file:
      input_lines=  file.readlines()
     # get all lines from filename
    abstract_lines = "" # create an empty abstract
    abstract_samples = [] # create an empty list of abstracts

    # Loop through each line in target file
    for line in input_lines:
      if line.startswith("###"): # check to see if line is an ID line
        abstract_id = line
        abstract_lines = "" # reset abstract string
      elif line.isspace(): # check to see if line is a new line
        abstract_line_split = abstract_lines.splitlines() # split abstract into separate lines

        # Iterate through each line in abstract and count them at the same time
        for abstract_line_number, abstract_line in enumerate(abstract_line_split):
          line_data = {} # create empty dict to store data from line
          target_text_split = abstract_line.split("\t") # split target label from text
          line_data["target"] = target_text_split[0] # get target label
          line_data["text"] = target_text_split[1].lower() # get target text and lower it
          line_data["line_number"] = abstract_line_number # what number line does the line appear in the abstract?
          line_data["total_lines"] = len(abstract_line_split) - 1 # how many total lines are in the abstract? (start from 0)
          abstract_samples.append(line_data) # add line data to abstract samples list

      else: # if the above conditions aren't fulfilled, the line contains a labelled sentence
        abstract_lines += line
    return abstract_samples
  def to_pandas(self,data_name):
    df = pd.DataFrame(data_name)
    return df
  def text_tolist(self,df):
    return df['text'].to_list()
  def label_encoding(self,df,split_on):
    encoder = LabelEncoder()
    target = df['target'].to_numpy()
    if split_on == 'train':
      target = encoder.fit_transform(target)
      return target
    elif split_on == 'validation' or split_on=='test':
      target = encoder.transform(target)
      return target
  def onehot_encoding(self,df,split_on):
    target = df.target.to_numpy().reshape(-1,1)
    onehot = OneHotEncoder(sparse_output=False)
    if split_on == 'train':
      target = onehot.fit_transform(target)
      return target
    elif split_on == 'validation' or split_on=='test':
      target = onehot.transform(target)
      return target
  def calculate(self,y_true,preds):
    scores = {}
    scores['accuracy'] = accuracy_score(y_true,preds)
    scores['recall']=recall_score(y_true,preds,average='weighted')
    scores['precision'] = precision_score(y_true,preds,average='weighted')
    scores['f1_score'] = f1_score(y_true,preds,average='weighted')
    return scores

  def create_baseline_model(self,X_train,y_train):
    pipe = Pipeline([('tfidf',TfidfVectorizer()),('model',MultinomialNB())])
    pipe.fit(X_train,y_train)
    return pipe
  def get_random_sentence(self,sentence,*processes):
    rand_sent = random.choice(sentence)
    print(rand_sent)
    for process in processes:
      rand_sent = process(rand_sent)
      print(rand_sent)

  def dataset_autotune(self,X,y):
    dataset = tf.data.Dataset.from_tensor_slices((X,y)).batch(32).prefetch(tf.data.AUTOTUNE)
    return dataset






