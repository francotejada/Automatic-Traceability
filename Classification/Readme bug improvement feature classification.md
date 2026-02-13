Repository: francotejada/automatic-traceability
File: 2_Classification_bug_improv_new_feat_2025_.ipynb
Lines: 703

Estimated tokens: 5.0k

Directory structure:
└── 2_Classification_bug_improv_new_feat_2025_.ipynb


"""
<a href="https://colab.research.google.com/github/francotejada/Automatic-Traceability/blob/main/Classification/2_Classification_bug_improv_new_feat_2025_.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
"""

# install the requirements
!pip install spacy
#!python -m spacy download es_core_news_md
#!python -m spacy download en_core_web_md
!python -m spacy download en_core_web_sm

!pip install torch
!pip install transformers
#!pip install contextualSpellCheck
#!pip install textblob
!pip install wordninja

import torch
import pandas as pd
from tqdm.notebook import tqdm

from transformers import AutoTokenizer #DistilBertTokenizer # BertTokenizer
from torch.utils.data import TensorDataset

from transformers import AutoModelForSequenceClassification # DistilBertForSequenceClassification #BertForSequenceClassification

device = torch.device("cuda")

# TEST
import re
import spacy

# Function checks if the string
# contains any special character
def check_token_accepted(string):

    special_characters = "!@#$%^&*()-+?_=,<>\/"
    s=string
    # Example: $tackoverflow

    if any(c in special_characters for c in s):
        return 0
    else:
        return 1

def clean_tokens_special_char(string):
    out = ''
    for word in string.split():
        if check_token_accepted(word) == 1:
           out = out + word + ' '
    return(out)

sen = 'Newline escape has the wrong order of \n\r CandidateStep value.replaceAll(""(\n)|(\r\n)"", System.getProperty(""line.separator"")); must be: value.replaceAll(""(\n)|(\n\r)"", System.getProperty(""line.separator""));'
sen1 = "that don't need to be in a stack"
print(clean_tokens_special_char(sen))

for t in sen1.split(" "):
  print(t)

# Output:
#   Newline escape has the wrong order of CandidateStep must be: 

#   that

#   don't

#   need

#   to

#   be

#   in

#   a

#   stack


import re
from typing import List
import wordninja
import pandas as pd

import spacy
from spacy.tokens import Doc
from tqdm import tqdm



class SpacyPreprocessor:
    def __init__(
        self,
        spacy_model=None,
        #remove_numbers=False,
        remove_numbers=True,
        remove_special=True,
        pos_to_remove=None,
        #remove_stopwords=False,
        remove_stopwords=True,
        lemmatize=False,
    ):
        """
        Preprocesses text using spaCy
        :param remove_numbers: Whether to remove numbers from text
        :param remove_stopwords: Whether to remove stopwords from text
        :param remove_special: Whether to remove special characters (including numbers)
        :param pos_to_remove: list of PoS tags to remove
        :param lemmatize:  Whether to apply lemmatization
        """

        self._remove_numbers = remove_numbers
        self._pos_to_remove = pos_to_remove
        self._remove_stopwords = remove_stopwords
        self._remove_special = remove_special
        self._lemmatize = lemmatize

        if not spacy_model:
            self.model = spacy.load("en_core_web_sm")
        else:
            self.model = spacy_model

    @staticmethod
    def download_spacy_model(model="en_core_web_sm"):
        print(f"Downloading spaCy model {model}")
        spacy.cli.download(model)
        print(f"Finished downloading model")

    @staticmethod
    def load_model(model="en_core_web_sm"):
        return spacy.load(model, disable=["ner", "parser"])

    def tokenize(self, text) -> List[str]:
        """
        Tokenize text using a spaCy pipeline
        :param text: Text to tokenize
        :return: list of str
        """
        doc = self.model(text)
        return [token.text for token in doc]

    def preprocess_text(self, text) -> str:
        """
        Runs a spaCy pipeline and removes unwanted parts from text
        :param text: text string to clean
        :return: str, clean text
        """
        doc = self.model(text)
        return self.__clean(doc)

    def preprocess_text2(self, text) -> str:
        """
        Runs a spaCy pipeline and removes unwanted parts from text
        :param text: text string to clean
        :return: str, clean text
        """
        doc = self.model(text)
        return self.__clean2(doc)

    def preprocess_text_list(self, texts=List[str]) -> List[str]:
        """
        Runs a spaCy pipeline and removes unwantes parts from a list of text.
        Leverages spaCy's `pipe` for faster batch processing.
        :param texts: List of texts to clean
        :return: List of clean texts
        """
        clean_texts = []
        for doc in tqdm(self.model.pipe(texts)):
            clean_texts.append(self.__clean(doc))

        return clean_texts

    def __clean(self, doc: Doc) -> str:

        tokens = []
        # POS Tags removal
        if self._pos_to_remove:
            for token in doc:
                if token.pos_ not in self._pos_to_remove:
                    tokens.append(token)
        else:
            tokens = doc

        # Remove Numbers
        if self._remove_numbers:
            tokens = [
                token for token in tokens if not (token.like_num or token.is_currency)
            ]

        # Remove Stopwords
        if self._remove_stopwords:
            tokens = [token for token in tokens if not token.is_stop]
        # remove unwanted tokens
        tokens = [
            token
            for token in tokens
            if not (
                token.is_punct or token.is_space or token.is_quote or token.is_bracket #or len(token) > 30
            )
        ]

        # Remove empty tokens
        tokens = [token for token in tokens if token.text.strip() != ""]

        # Lemmatize
        if self._lemmatize:
            text = " ".join([token.lemma_ for token in tokens])
        else:
            text = " ".join([token.text for token in tokens])

        if self._remove_special:
            # Remove non alphabetic characters
            text = re.sub(r"[^a-zA-Z\']", " ", text)
        # remove non-Unicode characters
        text = re.sub(r"[^\x00-\x7F]+", "", text)

        text = text.lower()

        return text

    def __clean2(self, doc: Doc) -> str:

        tokens = []

        tokens = doc

        tokens = [
                token for token in tokens if not (token.like_num or token.is_currency)
        ]

        # Remove empty tokens
        tokens = [token for token in tokens if token.text.strip() != "" or len(token) > 30]

        text = " ".join([token.text for token in tokens])

        # Remove non alphabetic characters
        text = re.sub(r"[^a-zA-Z\']", " ", text)

        # remove non-Unicode characters
        text = re.sub(r"[^\x00-\x7F]+", "", text)

        text = text.lower()

        return doc


import numpy as np
import csv
from spacy import displacy
import re
#from textblob import TextBlob
#import wordninja
#import contextualSpellCheck

if __name__ == "__main__":

    spacy_model = SpacyPreprocessor.load_model()
    preprocessor = SpacyPreprocessor(spacy_model=spacy_model, lemmatize=True, remove_numbers=True, remove_stopwords=True)

    #clean_text = preprocessor.preprocess_text("spaCy is awesome! 123")
    #print(clean_text)

    #df = pd.read_csv('jbehave_all.csv')
    df = pd.read_csv('axis2.sqlite3.csv')
    df.head()
    #print(df['summary'])

    texto = df.loc[:,"summary"]
    tipo = df.loc[:,"type"]

    cols = np.array(texto)
    cols2 = np.array(tipo)

    file ="jbehave_cleaned.csv"

    nlp = spacy.load("en_core_web_sm")
    #nlp = spacy.load("en_core_web_md")
    #en_core_web_sm

    #contextualSpellCheck.add_to_pipe(nlp)

    #doc = nlp("This is a sentence.")
    #displacy.serve(doc, style="dep")

    with open(file,"w", newline='', encoding='utf8') as rf:
        fieldnames=['summary','type']

        writer= csv.DictWriter(rf,fieldnames=fieldnames)
        writer.writerow({'summary':'summary','type':'type'})

        for i in range(0,len(cols)):
            #texto_col = cols[i].split(" ")
            # 06.09.2021 print(cols[i], ' ')
            #clean_text = preprocessor.preprocess_text(cols[i])
            clean_text = re.sub(r'{code}.*$', "", cols[i])

            clean_text = re.sub(r'{noformat}.*$', "", cols[i])

            # Remove URLs
            clean_text = re.sub("(?P<url>https?://[^\s]+)0123456789", '', clean_text, flags=re.MULTILINE)

            # Elimina tokens con caracteres especiales
            clean_text = clean_tokens_special_char(clean_text)

            clean_text = preprocessor.preprocess_text2(clean_text)
            print(clean_text)

            # FT 21.10.2021
            #clean_text = re.sub(' +', ' ',clean_text)

            #doc = nlp(clean_text)
            #print(doc._.outcome_spellCheck)
            #writer.writerow({'summary':doc._.outcome_spellCheck,'type':cols2[i]})

            # 06 09 2021 #
            #text = wordninja.split(clean_text)
            #text = TextBlob(str(text))

            # FT 21.10.2021
            #clean_text = clean_text.replace(',', '')

            #print(i, ' ')
            #print(i, ' ')
            writer.writerow({'summary':clean_text,'type':cols2[i] })  #cols[i]})

            #writer.writerow({'summary':str(text.correct()),'type':cols2[i]})



df = pd.read_csv('jbehave_cleaned.csv')
#df = pd.read_csv('jbehave_feat.csv')

df.head()
# Output:
#                                                summary         type

#   0  Adding CORS Support User agents commonly apply...  New Feature

#   1     JMX support Adding JMX support for AXIS2. See   New Feature

#   2  Provide new plugpoint in code This work will p...  New Feature

#   3  2.1: Add support for AddressingFeature and Sub...  New Feature

#   4  Add support for and Complete support for the n...  New Feature

df['type'].value_counts()
# Output:
#   Bug            230

#   Improvement    230

#   New Feature    230

#   Name: type, dtype: int64

possible_labels = df.type.unique()

label_dict = {}
for index, possible_label in enumerate(possible_labels):
    label_dict[possible_label] = index
label_dict
# Output:
#   {'Bug': 1, 'Improvement': 2, 'New Feature': 0}

df['label'] = df.type.replace(label_dict)

df.head()
# Output:
#                                                summary         type  label

#   0  Adding CORS Support User agents commonly apply...  New Feature      0

#   1     JMX support Adding JMX support for AXIS2. See   New Feature      0

#   2  Provide new plugpoint in code This work will p...  New Feature      0

#   3  2.1: Add support for AddressingFeature and Sub...  New Feature      0

#   4  Add support for and Complete support for the n...  New Feature      0

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(df.index.values,
                                                  df.label.values,
                                                  test_size=0.15,
                                                  random_state=42,
                                                  stratify=df.label.values)

df['data_type'] = ['not_set']*df.shape[0]

df.loc[X_train, 'data_type'] = 'train'
df.loc[X_val, 'data_type'] = 'val'

df.groupby(['type', 'label', 'data_type']).count()
# Output:
#                                summary

#   type        label data_type         

#   Bug         2     train         1402

#                     val            248

#   Improvement 0     train         1403

#                     val            247

#   New Feature 1     train         1402

#                     val            248

#tokenizer = BertTokenizer.from_pretrained('distilbert-base-uncased',#'allenai/scibert_scivocab_uncased') # 'bert-base-uncased',
#                                          do_lower_case=True)
tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-base',#'allenai/scibert_scivocab_uncased') # 'bert-base-uncased',
                                                do_lower_case=True)

# Output:
#   Downloading:   0%|          | 0.00/28.0 [00:00<?, ?B/s]
#   Downloading:   0%|          | 0.00/226k [00:00<?, ?B/s]
#   Downloading:   0%|          | 0.00/455k [00:00<?, ?B/s]
#   Downloading:   0%|          | 0.00/483 [00:00<?, ?B/s]

text_file = open("vocab.txt", "r")
new_tokens = text_file.readlines()
print(new_tokens)
print(len(new_tokens))
text_file.close()

encoded_data_train = tokenizer.batch_encode_plus(
    df[df.data_type=='train'].summary.values,
    add_special_tokens=True,
    return_attention_mask=True,
    pad_to_max_length=True,
    max_length=512, #256
    return_tensors='pt'
)

encoded_data_val = tokenizer.batch_encode_plus(
    df[df.data_type=='val'].summary.values,
    add_special_tokens=True,
    return_attention_mask=True,
    pad_to_max_length=True,
    max_length=512, #256
    return_tensors='pt'
)


input_ids_train = encoded_data_train['input_ids']
attention_masks_train = encoded_data_train['attention_mask']
labels_train = torch.tensor(df[df.data_type=='train'].label.values)

input_ids_val = encoded_data_val['input_ids']
attention_masks_val = encoded_data_val['attention_mask']
labels_val = torch.tensor(df[df.data_type=='val'].label.values)

dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)
dataset_val = TensorDataset(input_ids_val, attention_masks_val, labels_val)

len(dataset_train), len(dataset_val)
# Output:
#   (586, 104)

#model = BertForSequenceClassification.from_pretrained("distilbert-base-uncased", #"allenai/scibert_scivocab_uncased", #"bert-base-uncased",
#                                                      num_labels=len(label_dict),
#                                                      output_attentions=False,
#                                                      output_hidden_states=False)
model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-v3-base", #"allenai/scibert_scivocab_uncased", #"bert-base-uncased",
                                                            num_labels=len(label_dict),
                                                            output_attentions=False,
                                                            output_hidden_states=False)

print("[ BEFORE ] tokenizer vocab size:", len(tokenizer))
added_tokens = tokenizer.add_tokens(new_tokens)

print("[ AFTER ] tokenizer vocab size:", len(tokenizer))
print()
print('added_tokens:',added_tokens)
print()

# resize the embeddings matrix of the model
model.resize_token_embeddings(len(tokenizer))
# Output:
#   [ BEFORE ] tokenizer vocab size: 30522

#   [ AFTER ] tokenizer vocab size: 31948

#   

#   added_tokens: 1426

#   

#   Embedding(31948, 768)

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

batch_size = 3

dataloader_train = DataLoader(dataset_train,
                              sampler=RandomSampler(dataset_train),
                              batch_size=batch_size)

dataloader_validation = DataLoader(dataset_val,
                                   sampler=SequentialSampler(dataset_val),
                                   batch_size=batch_size)

from transformers import AdamW, get_linear_schedule_with_warmup

optimizer = AdamW(model.parameters(),
                  lr=1e-5,
                  eps=1e-8)

epochs = 5

scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0,
                                            num_training_steps=len(dataloader_train)*epochs)

from sklearn.metrics import f1_score

def f1_score_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat, average='weighted')

def accuracy_per_class(preds, labels):
    label_dict_inverse = {v: k for k, v in label_dict.items()}

    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    for label in np.unique(labels_flat):
        y_preds = preds_flat[labels_flat==label]
        y_true = labels_flat[labels_flat==label]
        print(f'Class: {label_dict_inverse[label]}')
        print(f'Accuracy: {len(y_preds[y_preds==label])}/{len(y_true)}\n')

import random
import numpy as np

seed_val = 17
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

print(device)
# Output:
#   cpu


def evaluate(dataloader_val):

    model.eval()

    loss_val_total = 0
    predictions, true_vals = [], []

    for batch in dataloader_val:

        batch = tuple(b.to(device) for b in batch)

        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2],
                 }

        with torch.no_grad():
            outputs = model(**inputs)

        loss = outputs[0]
        logits = outputs[1]
        loss_val_total += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)

    loss_val_avg = loss_val_total/len(dataloader_val)

    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)

    return loss_val_avg, predictions, true_vals

for epoch in tqdm(range(1, epochs+1)):

    model.train()

    loss_train_total = 0

    progress_bar = tqdm(dataloader_train, desc='Epoch {:1d}'.format(epoch), leave=False, disable=False)
    for batch in progress_bar:

        model.zero_grad()

        batch = tuple(b.to(device) for b in batch)

        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2],
                 }

        outputs = model(**inputs)

        loss = outputs[0]
        loss_train_total += loss.item()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

        progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item()/len(batch))})


    torch.save(model.state_dict(), f'finetuned_BERT_epoch_{epoch}.model')

    tqdm.write(f'\nEpoch {epoch}')

    loss_train_avg = loss_train_total/len(dataloader_train)
    tqdm.write(f'Training loss: {loss_train_avg}')

    val_loss, predictions, true_vals = evaluate(dataloader_validation)
    val_f1 = f1_score_func(predictions, true_vals)
    tqdm.write(f'Validation loss: {val_loss}')
    tqdm.write(f'F1 Score (Weighted): {val_f1}')

#model = BertForSequenceClassification.from_pretrained("distilbert-base-uncased",#"allenai/scibert_scivocab_uncased", #"bert-base-uncased",
#                                                      num_labels=len(label_dict),
#                                                      output_attentions=False,
#                                                      output_hidden_states=False)
model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-v3-base",#"allenai/scibert_scivocab_uncased", #"bert-base-uncased",
                                                            num_labels=len(label_dict),
                                                            output_attentions=False,
                                                            output_hidden_states=False)
model.to(device)

model.load_state_dict(torch.load('finetuned_BERT_epoch_5.model', map_location=torch.device('cuda')))

_, predictions, true_vals = evaluate(dataloader_validation)

accuracy_per_class(predictions, true_vals)

# Clasificacion de New Feature
a = df[['summary', 'type', 'data_type']]
#print(a)
filter1 = a["data_type"]=="val"
filter2 = a["type"]=="New Feature"
a.where(filter1 & filter2, inplace = True)
b = a.dropna()
print(b)
b.to_csv('new_feature_val.csv')

import spacy
from spacy import displacy

nlp = spacy.load("en_core_web_sm")

def verify_modal_verb(text, model=nlp):
    # Create doc object
    doc = model(text)
    modal = False

    # Generate list of POS tags
    for token in doc:
        if token.text in ('can', 'could', 'may', 'might', 'shall', 'should', 'will', 'would', 'must') :
           modal = True
           break

    pos = [token.pos_ for token in doc]
    #print(pos)

    # Return number of proper nouns
    if pos.count('VERB') > 0 and modal == True :
       return 1
    else :
       return 0


df1 = pd.read_csv('new_feature_val.csv')
cont_good_class_nf = 0
cont_bad_class_nf = 0

for index, row in df.iterrows():
    print(row['summary'])
    print(verify_modal_verb(row['summary']))
    row['summary'] = row['summary'].lower()

    if verify_modal_verb(row['summary']) == 1:
       cont_good_class_nf = cont_good_class_nf + 1
    else:
       cont_bad_class_nf = cont_bad_class_nf + 1

print('Nro de regs bien clasificados NF: ', cont_good_class_nf)
print('Nro de regs mal clasificados NF: ', cont_bad_class_nf)

