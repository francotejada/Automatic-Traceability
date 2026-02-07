Repository: francotejada/automatic-traceability
File: 4_Extraction_text_feature_2025.ipynb
Lines: 2,049

Estimated tokens: 17.0k

Directory structure:
└── 4_Extraction_text_feature_2025.ipynb


================================================
FILE: Extraction/4_Extraction_text_feature_2025.ipynb
================================================
# Jupyter notebook converted to Python script.

"""
<a href="https://colab.research.google.com/github/francotejada/Automatic-Traceability/blob/main/Extraction/4_Extraction_text_feature_2025.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
"""

"""
In this notebook, we will check out how to perform information extraction using NLP techniques in Python.

I recommend, going over [this article](https://www.analyticsvidhya.com/blog/2020/06/nlp-project-information-extraction/) to understand the concept in detail.
"""

"""
# What is Information Extraction?
"""

"""
Text data contains a lot of information but not all of it will be important to you. We might be looking for names of entities, others would want to extract specific relationships between those entities. Our intentions differ according to our requirements.

Imagine having to go through all the legal documents to find legal precedence to validate your current case. Or having to go through all the research papers to find relevant information to cure a disease. There are many more examples like resume harvesting, media analysis, email scanning, etc.

But just imagine having to manually go through all of the textual data and extracting the most relevant information. Clearly, it is an uphill battle and you might even end up skipping some important information.

For anyone trying to analyze textual data, the difficult task is not of finding the right documents, but of finding the right information from these documents. Understanding the relationship between entities, understanding how the events have unfolded, or just simply finding hidden gems of information, is clearly what anyone is looking for when they go through a piece of text.

Therefore, coming up with an automated way of extracting the information from textual data and presenting it in a structured manner will help us reap a lot of benefits and tremendously reduce the amount of time we have to spend time skimming through text documents. This is precisely what information extraction strives to achieve.


> `The task of Information Extraction (IE) involves extracting meaningful information from unstructured text data and presenting it in a structured format.`


Using information extraction, we can retrieve pre-defined information such as the name of a person, location of an organization, or identify a relation between entities, and save this information in a structured format such as a database.

This enables us to reap the benefits of powerful query tools like SQL for further analysis. Creating such structured data using information extraction will not only help us in analyzing the documents better but also help us in understanding the hidden relationships in the text.
"""

"""
# How does Information Extraction work?
"""

"""
Given the capricious nature of text data that changes depending on different authors and under different contexts, Information Extraction seems like a daunting task at hand. But it doesn't have to be so.

We all know that sentences are made up of words belonging to various different Parts of Speech (POS). Their are eight different POS in the english language: noun, pronoun, verb, adjective , adverb, preposition, conjuction, and intersection. The POS determines how a specific word functions in meaning in a given sentence. For example take the word "right". In the sentence, "The boy was awarded with a chocolate for giving the right answer", "right" is used as an adjective. Whereas, in the sentence, "You have the right to say whatever you want", "right" is treated as a noun. This goes to show that POS tag of a word carries a lot of significance when it comes to understanding the meaning of a sentence. No doubt we can leverage it to extract meaningful information from our text.
"""

# import spacy
import spacy

# load english language model
nlp = spacy.load('en_core_web_sm',disable=['ner','textcat'])

text = "allow annotation based configuration annotations based configuration can be preferable to some users investigate and implement annotation based alternatives to programmatic configuration which should always be possible"

# create spacy
doc = nlp(text)

for token in doc:
    print(token.text,'->',token.pos_)
# Output:
#   allow -> VERB

#   annotation -> NOUN

#   based -> VERB

#   configuration -> NOUN

#   annotations -> NOUN

#   based -> VERB

#   configuration -> NOUN

#   can -> VERB

#   be -> AUX

#   preferable -> ADJ

#   to -> ADP

#   some -> DET

#   users -> NOUN

#   investigate -> VERB

#   and -> CCONJ

#   implement -> VERB

#   annotation -> NOUN

#   based -> VERB

#   alternatives -> NOUN

#   to -> ADP

#   programmatic -> ADJ

#   configuration -> NOUN

#   which -> DET

#   should -> VERB

#   always -> ADV

#   be -> AUX

#   possible -> ADJ


"""
We were easily able to determine the POS tags of all the words in the sentence. But how does it help in Information Extraction?

Well, if we wanted to extract nouns from the sentences, we could take a look at POS tags of the words/tokens in the sentence, using the attribute **.pos_**, and extract them accordingly.
"""

for token in doc:
    # check token pos
    if token.pos_=='NOUN':
        # print token
        print(token.text)
# Output:
#   annotation

#   configuration

#   annotations

#   configuration

#   users

#   annotation

#   alternatives

#   configuration


"""
It was that easy to extract words based on their POS tags. But sometimes extracting information purely based on the POS tags is not enough. Have a look at the sentence below:
"""

text = "The children love cream biscuits"

# create spacy
doc = nlp(text)

for token in doc:
    print(token.text,'->',token.pos_)
# Output:
#   The -> DET

#   children -> NOUN

#   love -> VERB

#   cream -> NOUN

#   biscuits -> NOUN


"""
If I wanted to extract the subject and the object from a sentence, I can’t do that based on their POS tags. For that, I need to look at how these words are related to each other. These are called **Dependencies**.

We can make use of [spaCy’s displacy](https://explosion.ai/demos/displacy) visualizer that displays the word dependencies in a graphical manner:
"""

from spacy import displacy
displacy.render(doc, style='dep',jupyter=True)
# Output:
#   <IPython.core.display.HTML object>

"""
Pretty cool! This directed graph is known as a [dependency graph](https://www.analyticsvidhya.com/blog/2017/12/introduction-computational-linguistics-dependency-trees/). It represents the relations between different words of a sentence.

Each word is a **node** in the Dependency graph. The relationship between words is denoted by the edges. For example, “The” is a determiner here, “children” is the subject of the sentence, “biscuits” is the object of the sentence, and “cream” is a compound word that gives us more information about the object.

The arrows carry a lot of significance here:

- The **arrowhead** points to the words that are **dependent** on the word pointed by the **origin of the arrow**
- The former is referred to as the **child node** of the latter. For example, “children” is the child node of “love”
- The word which has no incoming arrow is called the **root node** of the sentence

Let’s see how we can extract the subject and the object from the sentence. Like we have an attribute for POS in SpaCy tokens, we similarly have an attribute for extracting the dependency of a token denoted by dep_:
"""

for token in doc:
    # extract subject
    if (token.dep_=='nsubj'):
        print(token.text)
    # extract object
    elif (token.dep_=='dobj'):
        print(token.text)
# Output:
#   children

#   biscuits


"""
Voila! We have the subject and object of our sentence.

Using POS tags and Dependency tags, we can look for relationship between different entities in a sentence. For example, in the sentence "The **cat** perches **on** the **window sill**", we have two noun entities,"cat" and "window sill", related by the preposition "on". We can look for such relationships and much more to extract meaningful information from our text data.

*I suggest going through [this amazing blog](https://www.analyticsvidhya.com/blog/2019/09/introduction-information-extraction-python-spacy/) which explains Information Extraction with tons of examples.*
"""

"""
# Where Do We Go from Here?
"""

"""
We have briefly spoken about the theory regarding Information Extraction which I believe is important to understand before jumping into the crux of this article.

`“An ounce of practice is generally worth more than a ton of theory.” –E.F. Schumacher`

In the following sections, I am going to explore a text dataset and apply the information extraction technique to retrieve some important information, understand the structure of the sentences, and the relationship between entities.

So, without further ado, let’s get cracking on the code!
"""

"""
# Getting Familiar with the Text Dataset
"""

"""
The dataset we are going to be working with is the [United Nations General Debate Corpus](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/0TJX8Y). It contains speeches made by representatives of all the member countries from the year 1970 to 2018 at the General Debate of the annual session of United Nations General Assembly.

But we will take a subset of this dataset and work with speeches made by India at these debates. This will allow us to stay on track and better understand the task at hand of understanding Information Extraction. This leaves us with 49 speeches made by India over the years, each speech ranging from anywhere between 2000 to 6000+ words.

Having said that, let’s have a look at our dataset:
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import re

# Folder path
folders = glob.glob('jbehave_256_cleaned.csv') #('./UNGD/UNGDC 1970-2018/Converted sessions/Session*')

# Dataframe
#df = pd.DataFrame(columns={'Country','Speech','Session','Year'})
df = pd.DataFrame(columns={'summary','type'})

# Read speeches by India
i = 0
for file in folders:

    speech = glob.glob(file)

    with open(speech[0],encoding='utf8') as f:
        # Speech
        #df.loc[i,'Speech'] = f.read()
        df.loc[i,'summary'] = f.read()
        # Year
        #df.loc[i,'Year'] = speech[0].split('_')[-1].split('.')[0]
        # Session
        #df.loc[i,'Session'] = speech[0].split('_')[-2]
        # Country
        #df.loc[i,'Country'] = speech[0].split('_')[0].split("\\")[-1]
        # Increment counter
        i += 1

df.head()

"""
I will print a snapshot of one of the speeches to give you a feel of what the data looks like.
"""

#df = pd.read_csv('jbehave_all.csv')
df = pd.read_csv('jbehave_256_cleaned.csv')
df['type'].value_counts()

# Output:
#   Improvement    116

#   Bug             55

#   New Feature     55

#   Name: type, dtype: int64

df.loc[0,'summary']
# Output:
#   'springstoryreporterbuilder does not expose all properties in storyreporterbuilder here is a patch to start to fix it i was unsure on other properties https github com jbehave jbehave core pull '

"""
Now let's start working with out dataset!
"""

"""
# Speech Text Pre-Processing
"""

"""
First, we need to clean our text data. When I went over a few speeches, I found each paragraph in the speech was numbered to distinctly identify it. There were obviously unwanted characters like newline character, a hyphen, salutations, and apostrophes, like in any other text dataset.

But another unique and unwanted information present were the references made in each speech to other documents. We obviously don’t want that either.

I have written a simple function to clean the speeches. **An important point here is that I haven’t used lemmatization or changed the words to lowercase as it has the potential to change the POS tag of the word.** We certainly don’t want to do that as you will see in the upcoming subsections.
"""

# function to preprocess speech
def clean(text):

    # removing paragraph numbers
    text = re.sub('[0-9]+.\t','',str(text))
    # removing new line characters
    text = re.sub('\n ','',str(text))
    text = re.sub('\n',' ',str(text))
    # removing apostrophes
    text = re.sub("'s",'',str(text))
    # removing hyphens
    text = re.sub("-",' ',str(text))
    text = re.sub("— ",'',str(text))
    # removing quotation marks
    text = re.sub('\"','',str(text))
    # removing salutations
    text = re.sub("Mr\.",'Mr',str(text))
    text = re.sub("Mrs\.",'Mrs',str(text))
    # removing any reference to outside text
    text = re.sub("[\(\[].*?[\)\]]", "", str(text))

    return text

# preprocessing speeches
df['Speech_clean'] = df['summary'].apply(clean)

"""
Right, now that we have our minimally cleaned speeches, we can split it up into separate sentences.
"""

"""
# Split the Speech into Different Sentences
"""

"""
Spliting our speeches into separate sentences will allow us to extract information from each sentence and later we can combine it to get cummulative information from any specific year.
"""

# split sentences
def sentences(text):
    # split sentences and questions
    text = re.split('[.?]', text)
    clean_sent = []
    for sent in text:
        clean_sent.append(sent)
    return clean_sent

# sentences
df['sent'] = df['Speech_clean'].apply(sentences)

df.head()
# Output:
#                                                summary  ...                                               sent

#   0  springstoryreporterbuilder does not expose all...  ...  [springstoryreporterbuilder does not expose al...

#   1  storymaps could link back to story source or c...  ...  [storymaps could link back to story source or ...

#   2  add crossreference report as contributed by th...  ...  [add crossreference report as contributed by t...

#   3  allow configuration of storytimeoutinsecs via ...  ...  [allow configuration of storytimeoutinsecs via...

#   4  provide property based embeddercontrols an ext...  ...  [provide property based embeddercontrols an ex...

#   

#   [5 rows x 4 columns]

"""
Finally, we can create a dataframe containing the sentences from different years:
"""

df.head(20)
# Output:
#                                                 summary  ...                                               sent

#   0   springstoryreporterbuilder does not expose all...  ...  [springstoryreporterbuilder does not expose al...

#   1   storymaps could link back to story source or c...  ...  [storymaps could link back to story source or ...

#   2   add crossreference report as contributed by th...  ...  [add crossreference report as contributed by t...

#   3   allow configuration of storytimeoutinsecs via ...  ...  [allow configuration of storytimeoutinsecs via...

#   4   provide property based embeddercontrols an ext...  ...  [provide property based embeddercontrols an ex...

#   5    composite step is executing successfully when...  ...  [ composite step is executing successfully whe...

#   6   generic parameter converter for enum classes i...  ...  [generic parameter converter for enum classes ...

#   7   support for weld context and dependency inject...  ...  [support for weld context and dependency injec...

#   8   add unoverridableembedder an extension of embe...  ...  [add unoverridableembedder an extension of emb...

#   9   rename run with annotated embedder goal to run...  ...  [rename run with annotated embedder goal to ru...

#   10  givenstories should be able to be loaded relat...  ...  [givenstories should be able to be loaded rela...

#   11  afterstories xml and beforestories xml being c...  ...  [afterstories xml and beforestories xml being ...

#   12  if story is ' excluded ' because of meta filte...  ...  [if story is ' excluded ' because of meta filt...

#   13  examplestablefactory should load resources fro...  ...  [examplestablefactory should load resources fr...

#   14  provide pending annotation to mark methods tha...  ...  [provide pending annotation to mark methods th...

#   15  null meta filters should be ignored in maven p...  ...  [null meta filters should be ignored in maven ...

#   16  ability to enqueue ad hoc stories asynchronous...  ...  [ability to enqueue ad hoc stories asynchronou...

#   17  ensure utility methods should be independent o...  ...  [ensure utility methods should be independent ...

#   18  should not require a space before a new line c...  ...  [should not require a space before a new line ...

#   19  should be able to have multiple scenarios in a...  ...  [should be able to have multiple scenarios in ...

#   

#   [20 rows x 4 columns]

df.loc[1,'sent']
# Output:
#   ['storymaps could link back to story source or colored htmlified story output refer http screencast com t g a ch x ca nt click the story name to go anywhere ']

# Create a dataframe containing sentences
df2 = pd.DataFrame(columns=['Sent','Year','Len'])

# List of sentences for new df
row_list = []

# for-loop to go over the df speeches
for i in range(len(df)):

    # for-loop to go over the sentences in the speech
    for sent in df.loc[i,'sent']:

        wordcount = len(sent.split())  # Word count
        year = df.loc[i,'type']  # Year
        dict1 = {'Year':year,'Sent':sent,'Len':wordcount}  # Dictionary
        row_list.append(dict1)  # Append dictionary to list

# Create the new df
df2 = pd.DataFrame(row_list)

df2.head()
# Output:
#             Year                                               Sent  Len

#   0  Improvement  springstoryreporterbuilder does not expose all...   30

#   1  Improvement  storymaps could link back to story source or c...   30

#   2  New Feature  add crossreference report as contributed by th...   24

#   3  Improvement  allow configuration of storytimeoutinsecs via ...   27

#   4  Improvement  provide property based embeddercontrols an ext...   17

df2.shape
# Output:
#   (226, 3)

"""
After performing this operation, we end up with 7150 sentences. Going over them and extracting information manually will be a difficult task. That’s why we are looking at Information Extraction using NLP techniques!
"""

"""
# Information Extraction using SpaCy
"""

"""
Now to get to the crux of the discussion at hand, Information Extraction. We will be using the [Spacy](https://www.analyticsvidhya.com/blog/2020/03/spacy-tutorial-learn-natural-language-processing/) library for working with the text data. It has all the necessary tools that we can exploit for all the tasks we need for Information Extraction.
"""

"""
Let me import the relevant SpaCy modules that we will require for the task ahead:
"""

!pip install visualise_spacy_tree
# Output:
#   Collecting visualise_spacy_tree

#     Downloading visualise_spacy_tree-0.0.6-py3-none-any.whl (5.0 kB)

#   Collecting pydot==1.4.1

#     Downloading pydot-1.4.1-py2.py3-none-any.whl (19 kB)

#   Requirement already satisfied: pyparsing>=2.1.4 in /usr/local/lib/python3.7/dist-packages (from pydot==1.4.1->visualise_spacy_tree) (3.0.7)

#   Installing collected packages: pydot, visualise-spacy-tree

#     Attempting uninstall: pydot

#       Found existing installation: pydot 1.3.0

#       Uninstalling pydot-1.3.0:

#         Successfully uninstalled pydot-1.3.0

#   Successfully installed pydot-1.4.1 visualise-spacy-tree-0.0.6


import spacy
from spacy.matcher import Matcher

from spacy import displacy
import visualise_spacy_tree
from IPython.display import Image, display

# load english language model
nlp = spacy.load('en_core_web_sm',disable=['ner','textcat'])

"""
We will need the spaCy Matcher class to create a pattern to match phrases in the text. We’ll also require the displaCy module for visualizing the dependency graph of sentences.

The **visualise_spacy_tree** library will be needed for creating a tree-like structure out of the Dependency graph. This helps in visualizing the graph in a better way. Finally, IPython Image and display classes are required to output the tree.

But you don’t need to worry about these too much. It will become clear as you look at the code.
"""

"""
## Information Extraction #1 – Finding Mentions of Prime Minister in the Speech
"""

"""
When working on Information extraction task, it is important to manually go over a subset of the dataset to understand what the text is like and determine if anything catches your attention at first glance.

When I first went over the speeches, I found out that many of them referred to what the Prime Minister had said, thought or achieved in the past. We know that a country is nothing without its leader. The destination a country ends up in is by and large the result of the able guidance of its leader. Therefore, I believe it is important to extract those sentences from the speeches that referred to Prime Ministers of India, and try and understand what their thinking and perspective was, and also try to unravel any common or differing beliefs over the years.

To achieve this task, I used [Spacy's Matcher class](https://spacy.io/api/matcher). It allows you to match a sequence of words based on certain patterns. For the current task, we know that whenever a Prime Minister is reffered to in the speech, it will be in one of the following ways:
* Prime Minister of [Country] ...
* Prime Minister [Name] ...

Using this general understanding one can come up with a pattern as follows:

pattern = [{'LOWER':'prime'},

           {'LOWER':'minister'},
              
           {'POS':'ADP','OP':'?'},
              
           {'POS':'PROPN'}]
        
Let me walk you through this pattern:     
* Here, each dictionary in the list is a unique word.
* The first and second words match the keyword "Prime Minister" irrespective of whether it is in uppercase or not, which is why I have included the key "LOWER".
* The third keyword matches a word that is a preposition. What I am looking for here is the word "of". Now, as discussed before, it may or may not be present in the pattern, therefore, an additional key, "OP", is mentioned to point out just that.
* Finally, the last keyword in the phrase should be a proper noun. This is either be the name of the country or name of the prime minister.
* The matched keywords have to be in continuation otherwise the pattern will not match the phrase.
"""

# Function to find sentences containing PMs of India
def find_names(text):

    names = []

    # Create a spacy doc
    doc = nlp(text)

    # Define the pattern
    pattern = [{'LOWER':'prime'},
              {'LOWER':'minister'},
              {'POS':'ADP','OP':'?'},
              {'POS':'PROPN'}]

    # Matcher class object
    matcher = Matcher(nlp.vocab)
    matcher.add("names", None, pattern)

    matches = matcher(doc)

    # Finding patterns in the text
    for i in range(0,len(matches)):

        # match: id, start, end
        token = doc[matches[i][1]:matches[i][2]]
        # append token to list
        names.append(str(token))

    # Only keep sentences containing Indian PMs
    for name in names:
        if (name.split()[2] == 'of') and (name.split()[3] != "India"):
                names.remove(name)

    return names

# Apply function
df2['PM_Names'] = df2['Sent'].apply(find_names)

"""
Here are some sample sentences from the year 1984 that matched our pattern:
"""

# look at sentences for a specific year
for i in range(len(df2)):
    if df2.loc[i,'Year'] in ['1984']:
        if len(df2.loc[i,'PM_Names'])!=0:
            print('->',df2.loc[i,'Sent'],'\n')

count=0
for i in range(len(df2)):
    if len(df2.loc[i,'PM_Names'])!=0:
        count+=1
print(count)
# Output:
#   0


"""
Now, since only 58 sentences out of 7150 total sentences gave an output that matched our pattern, I have summarised the relevant information from these outputs here:

- PM Indira Gandhi and PM Jawaharlal Nehru believed in working together in unity and with the principles of the UN
- PM Indira Gandhi believed in striking a balance between global production and consumption. She set out policies dedicated to national reconstruction and the consolidation of a secular and pluralistic political system
- PM Indira Gandhi emphasized that India does not intervene in the internal affairs of other countries. However, this stand on foreign policy took a U-turn under PM Rajiv Gandhi when he signed an agreement with the Sri Lankan Prime Minister which brought peace to Sri Lanka
- Both PM Indira Gandhi and PM Rajiv Gandhi believed in the link between economic development and protection of the environment
- PM Rajiv Gandhi advocated for the disarmament of nuclear weapons, a belief that was upheld by India over the years
- Indian, under different PMs, has always extended a hand of peace towards Pakistan over the years
- PM Narendra Modi believes that economic empowerment and upliftment of any nation involves the empowerment of its women
- PM Narendra Modi has launched several schemes that will help India achieve its SGD goals

Using information extraction, we were able to isolate only a few sentences that we required that gave us maximum results.
"""

"""
## Information Extraction #2 – Finding Initiatives
"""

"""
The second interesting thing I noticed while going through the speeches is that there were a lot of initiatives, schemes, agreements, conferences, programs, etc. that were mentioned in the speeches. For example, ‘Paris agreement’, ‘Simla Agreement’, ‘Conference on Security Council’, ‘Conference of Non Aligned Countries’, ‘International Solar Alliance’, ‘Skill India initiative’, etc.

Extracting these would give us an idea about what are the priorities for India and whether there is a pattern as to why they are mentioned quite often in the speeches.

I am going to refer to all the schemes, initiatives, conferences, programmes, etc. keywords as initiatives.

To extract initiatives from the text, the first thing I am going to do is identify those sentences that talk about the initiatives. For that, I will use simple regex to select only those sentences that contain the keyword ‘initiative’, ‘scheme’, ‘agreement’, etc. This will reduce our search for the initiative pattern that we are looking for:
"""

# Function to check if keyswords like 'programs','schemes', etc. present in sentences

def prog_sent(text):

    patterns = [r'\b(?i)'+'plan'+r'\b',
               r'\b(?i)'+'programme'+r'\b',
               r'\b(?i)'+'scheme'+r'\b',
               r'\b(?i)'+'campaign'+r'\b',
               r'\b(?i)'+'initiative'+r'\b',
               r'\b(?i)'+'conference'+r'\b',
               r'\b(?i)'+'agreement'+r'\b',
               r'\b(?i)'+'alliance'+r'\b']

    output = []
    flag = 0

    # Look for patterns in the text
    for pat in patterns:
        if re.search(pat, text) != None:
            flag = 1
            break
    return flag

# Apply function
df2['Check_Schemes'] = df2['Sent'].apply(prog_sent)
# Output:
#   /usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:19: DeprecationWarning: Flags not at the start of the expression '\\b(?i)plan\\b'

#   /usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:19: DeprecationWarning: Flags not at the start of the expression '\\b(?i)programme\\b'

#   /usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:19: DeprecationWarning: Flags not at the start of the expression '\\b(?i)scheme\\b'

#   /usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:19: DeprecationWarning: Flags not at the start of the expression '\\b(?i)campaign\\b'

#   /usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:19: DeprecationWarning: Flags not at the start of the expression '\\b(?i)initiative\\b'

#   /usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:19: DeprecationWarning: Flags not at the start of the expression '\\b(?i)conference\\b'

#   /usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:19: DeprecationWarning: Flags not at the start of the expression '\\b(?i)agreement\\b'

#   /usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:19: DeprecationWarning: Flags not at the start of the expression '\\b(?i)alliance\\b'


# Sentences that contain the initiative words
count = 0
for i in range(len(df2)):
    if df2.loc[i,'Check_Schemes'] == 1:
        count+=1
print(count)
# Output:
#   0


"""
Now, you might be thinking that our task is done here as we have already identified the sentences. We can easily look these up and determine what is being talked about in these sentences. But, think about it, not all of these will contain the initiative name. Some of these might be generally talking about initiatives but no initiative name might be present in them.

Therefore, we need to come up with a better solution that extracts only those sentences that contain the initiative names. For that, I am going to use the spaCy Matcher, once again, to come up with a pattern that matches these initiatives.

As you might have noticed, the initiative name is a proper noun that starts with a determiner and ends with either ‘initiative’/’programme’/’agreement’ etc. words in the end. It also includes an occasional preposition in the middle. I also noticed that most of the initiative names were between two to five words long. Keeping this in mind, I came up with the following pattern to match the initiative names:
"""

# To extract initiatives using pattern matching
def all_schemes(text,check):

    schemes = []

    doc = nlp(text)

    # Initiatives keywords
    prog_list = ['programme','scheme',
                 'initiative','campaign',
                 'agreement','conference',
                 'alliance','plan']

    # Define pattern to match initiatives names
    pattern = [{'POS':'DET'},
               {'POS':'PROPN','DEP':'compound'},
               {'POS':'PROPN','DEP':'compound'},
               {'POS':'PROPN','OP':'?'},
               {'POS':'PROPN','OP':'?'},
               {'POS':'PROPN','OP':'?'},
               {'LOWER':{'IN':prog_list},'OP':'+'}
              ]

    if check == 0:
        # return blank list
        return schemes

    # Matcher class object
    matcher = Matcher(nlp.vocab)
    matcher.add("matching", None, pattern)
    matches = matcher(doc)

    for i in range(0,len(matches)):

        # match: id, start, end
        start, end = matches[i][1], matches[i][2]

        if doc[start].pos_=='DET':
            start = start+1

        # matched string
        span = str(doc[start:end])

        if (len(schemes)!=0) and (schemes[-1] in span):
            schemes[-1] = span
        else:
            schemes.append(span)

    return schemes

# apply function
df2['Schemes1'] = df2.apply(lambda x:all_schemes(x.Sent,x.Check_Schemes),axis=1)

"""
Lets see how many of the sentences contain an initiative name.
"""

count = 0
for i in range(len(df2)):
    if len(df2.loc[i,'Schemes1'])!=0:
        count+=1
print(count)
# Output:
#   0


"""
62, not bad. Now lets have a look at a few of the outputs.
"""

year = '2021'
for i in range(len(df2)):
    if df2.loc[i,'Year']==year:
        if len(df2.loc[i,'Schemes1'])!=0:
            print('->',df2.loc[i,'Year'],',',df2.loc[i,'Schemes1'],':')
            print(df2.loc[i,'Sent'])

"""
But one thing I must point out here is that there were a lot more initiatives in the speeches that did not match our pattern. For example, in the year 2018, there were other initiatives too like “MUDRA”, ”Ujjwala”, ”Paris Agreement”, etc. So is there a better way to extract them?

Remember how we looked at dependencies at the beginning of the article? Well, we are going to use those to make some rules to match the initiative name. But before making a rule, you need to understand how a sentence is structured, only then can you come up with a general rule to extract relevant information.

To understand the structure of the sentence I am going to print the dependency graph of a sample example but in a tree fashion which gives a better intuition of the structure. Have a look below:
"""

# Printing dependency tree
doc = nlp(' Last year, I spoke about the Ujjwala programme , through which, I am happy to report, 50 million free liquid-gas connections have been provided so far')
png = visualise_spacy_tree.create_png(doc)
display(Image(png))
# Output:
#   <IPython.core.display.Image object>

"""
See how 'Ujjwala' is a child node of 'programme'. Have a look at another example.
"""

doc = nlp('Prime Minister Modi, together with the Prime Minister of France, launched the International Solar Alliance')
png = visualise_spacy_tree.create_png(doc)
display(Image(png))
# Output:
#   <IPython.core.display.Image object>

"""
Notice how the 'International Solar Alliance' is structured.

You must got the idea by now that the initiative names are usually children of nodes that containing words like 'initiative','programme',etc. Based on this knowledge we can develop our own rule.

The rule I am suggesting is pretty simple. Let me walk you through it.
* I am going to look for tokens in sentences that contain my initiative keywords.
* Then I am going to look at its subtree (or words dependent on it) using *token.subtree* and extract only those nodes/words that are proper nouns, since they are most likely going to contain the name of the initiative.
"""

# rule to extract initiative name
def sent_subtree(text):

    # pattern match for schemes or initiatives
    patterns = [r'\b(?i)'+'plan'+r'\b',
           r'\b(?i)'+'programme'+r'\b',
           r'\b(?i)'+'scheme'+r'\b',
           r'\b(?i)'+'campaign'+r'\b',
           r'\b(?i)'+'initiative'+r'\b',
           r'\b(?i)'+'conference'+r'\b',
           r'\b(?i)'+'agreement'+r'\b',
           r'\b(?i)'+'alliance'+r'\b']

    schemes = []
    doc = nlp(text)
    flag = 0
    # if no initiative present in sentence
    for pat in patterns:

        if re.search(pat, text) != None:
            flag = 1
            break

    if flag == 0:
        return schemes

    # iterating over sentence tokens
    for token in doc:

        for pat in patterns:

            # if we get a pattern match
            if re.search(pat, token.text) != None:

                word = ''
                # iterating over token subtree
                for node in token.subtree:
                    # only extract the proper nouns
                    if (node.pos_ == 'PROPN'):
                        word += node.text+' '

                if len(word)!=0:
                    schemes.append(word)

    return schemes

# derive initiatives
df2['Schemes2'] = df2['Sent'].apply(sent_subtree)

"""
Now let's see if we did any better than the last approach.
"""

count = 0
for i in range(len(df2)):
    if len(df2.loc[i,'Schemes2'])!=0:
        count+=1
print(count)
# Output:
#   0


"""
Wow, we matched 273 entries! That is a significant improvement over the previous result. Lets go over some of the sample outputs.
"""

year = '2021'
for i in range(len(df2)):
    if df2.loc[i,'Year']==year:
        if len(df2.loc[i,'Schemes2'])!=0:
            print('->',df2.loc[i,'Year'],',',df2.loc[i,'Schemes2'],':')
            print(df2.loc[i,'Sent'])

"""
Out of 7000+ sentences, we were able to zero down to just 282 sentences that talked about initiatives. I looped over these outputs and below is how I would summarise the output:

* There are a lot of different international initiatives or schemes that India has mentioned in its speeches. This goes to show that India has been an active member of the international community working towards building a better future by solving problems through these intiavtives.

* Another point to highlight here is that the initiatives mentioned in the initial years have been more focused on those that concern the international coomunity. However, during recent times, especially after 2014, a lot of domestic initiatives have been mentioned in the speeches like 'Ayushman Bharat', 'Pradhan Mantri Jan Dhan Yojana', etc. This shows a shift in how the country percevies its role in the community. By mentioning a lot of domestic initiatives, India has started to put more of the domestic work in front of the international community to witness and, probably, even follow in their footsteps.

Having said that, the results were definitely not perfect. There were instances when unwanted words were also got extracted with the initiative names. But the output derived by making our own rules was definitely better than the ones derived by using Spacy's pattern matcher. This goes to show the felixibility that one can achieve with making your own rules.
"""

"""
## Finding Patterns in the Speeches
"""

"""
So far, we extracted only that information that met our analytical eye when we skimmed over the data. But is there any other information hidden in our dataset? Surely there is and we are going to explore that by making our own rules using the dependency of the words, as we did in the previous section.

But before that, I want to point out two things.

First, when we are trying to understand the structure of the speech, we cannot look at the entire speech, that would take an eternity, and time is of the essence here. What we are going to do instead is look at random sentences from the dataset and then, based on their structure, try to come up with general rules to extract information.

But how do we test the validity of these rules? That’s where my second point comes in! Not all of the rules that we come up with will yield satisfactory results. So, to sift out the irrelevant rules, we can look at the percentage of sentences that matched our rule out of all sentences. This will give us a fair idea about how well the rule is performing, and whether, in fact, there is any such general structure in the corpus!

Another very important point that needs to be highlighted here is that any corpus is bound to contain long complex sentences. Working with these sentences to try and understand their structure will be a very difficult task. Therefore, we are going to look at smaller sentences. This will give us the opportunity to better understand their structure. So what’s the magic number? Let’s first look at how the sentence length varies in our corpus.
"""

plt.hist(df2['Len'],bins=20,range=[0,100])
plt.xticks(np.arange(0,100,5));
# Output:
#   <Figure size 432x288 with 1 Axes>

"""
Looking at the histogram, we can see that most of the sentences range from 15-20 words. So I am going to work with sentences that have no more than 15 words.
"""

row_list = []
# df2 contains all sentences from all speeches
for i in range(len(df2)):
    sent = df2.loc[i,'Sent']

    if (',' not in sent) and (len(sent.split()) <= 15):

        year = df2.loc[i,'Year']
        length = len(sent.split())

        dict1 = {'Year':year,'Sent':sent,'Len':length}
        row_list.append(dict1)

# df with shorter sentences
df3 = pd.DataFrame(columns=['Year','Sent',"Len"])
df3 = pd.DataFrame(row_list)

df3.head()
# Output:
#             Year                                               Sent  Len

#   0  Improvement  extract candidatesteps interface extract candi...   15

#   1  New Feature  add saucelabs contextview should use the webdr...   14

#   2  New Feature  flash support a webdriverprovider implementati...   14

"""
Now, lets come up with a function that will generate random sentences from this dataframe.
"""

from random import randint
def rand_sent(df):

    index = randint(0, len(df))
    print('Index = ',index)
    doc = nlp(df.loc[index,'Sent'][1:])
    displacy.render(doc, style='dep',jupyter=True)

    return index

rand_sent(df3)
# Output:
#   Index =  1

#   <IPython.core.display.HTML object>
#   1

"""
Finally, let's make a fucntion to evaluate the result of our rule.
"""

# function to check output percentage for a rule
def output_per(df,out_col):

    result = 0

    for out in df[out_col]:
        if len(out)!=0:
            result+=1

    per = result/len(df)
    per *= 100

    return per

"""
Right, let's get down to the business of making some rules!
"""

"""
## Information Extraction #3 – Rule on Noun-Verb-Noun Phrases
"""

"""
When you look at a sentence, it generally contains a **subject(noun), action(verb) and an object(noun)**. Rest of the words are just there to give us additional information about the entities. Therefore, we can leverage this basic structure to extract the main bits of information from the sentence. Take for example the following sentence:
"""

# To download dependency graphs to local folder
from pathlib import Path

text = df3.loc[9,'Sent'][1:]

doc = nlp(text)
img = displacy.render(doc, style='dep',jupyter=True)
img

# To save to folder
# output_path = Path("./img1.svg")
# output_path.open("w", encoding="utf-8").write(img)
# Output:
#   Error: KeyError: ignored

"""
What will be extracted is "countries face threats", which should give us a fair idea about what the sentence is trying to say.

So lets look at how this rule fairs what we run it against the short sentences that are working with.
"""

# Function for rule 1: noun(subject), verb, noun(object)
def rule1(text):

    doc = nlp(text)

    sent = []

    for token in doc:

        # If the token is a verb
        if (token.pos_=='VERB'):

            phrase =''

            # Only extract noun or pronoun subjects
            for sub_tok in token.lefts:

                if (sub_tok.dep_ in ['nsubj','nsubjpass']) and (sub_tok.pos_ in ['NOUN','PROPN','PRON']):

                    # Add subject to the phrase
                    phrase += sub_tok.text

                    # Save the root of the word in phrase
                    phrase += ' '+token.lemma_

                    # Check for noun or pronoun direct objects
                    for sub_tok in token.rights:

                        # Save the object in the phrase
                        if (sub_tok.dep_ in ['dobj']) and (sub_tok.pos_ in ['NOUN','PROPN']):

                            phrase += ' '+sub_tok.text
                            sent.append(phrase)

    return sent

"""
A continuacion creamos una regla nueva con los verbos modales:
"""

# Function for rule 1: noun(subject), verb, noun(object)
def rule11(text):

    doc = nlp(text)

    sent = []

    for token in doc:

        # If the token is a verb
        if (token.pos_=='VERB') and token.text in ('can', 'could', 'may', 'might', 'shall', 'should', 'will', 'would', 'must'):

            phrase =''

            #for sub_tok in token.rights:
            print (token[1])

            # Only extract noun or pronoun subjects
            for sub_tok in token.lefts:

                #if (sub_tok.dep_ in ['nsubj','nsubjpass']) and (sub_tok.pos_ in ['NOUN','PROPN','PRON']):
                #if (1 = 1): # (sub_tok.pos_ in ['VERB']):

                    print (sub_tok.text)
                    # Add subject to the phrase
                    phrase += sub_tok.text

                    # Save the root of the word in phrase
                    phrase += ' '+token.lemma_

                    sent.append(phrase)

                    # Check for noun or pronoun direct objects
                    for sub_tok in token.rights:

                        # Save the object in the phrase
                        #if (sub_tok.dep_ in ['dobj']) and (sub_tok.pos_ in ['NOUN','PROPN']):

                        phrase += ' '+sub_tok.text
                        sent.append(phrase)

    return sent

#rule1('allow annotation based configuration annotations based configuration can be preferable to some users investigate and implement annotation based alternatives to programmatic configuration which should always be possible')
#rule1('I hope you enjoy Prague.')
rule11('storymaps could link the sun back to story source or colored htmlified story output refer http screencast com t g a ch x ca nt click the story name to go anywhere')
# Output:
#   Error: TypeError: ignored

# Matcher is initialized with the shared vocab
from spacy.matcher import Matcher
# Each dict represents one token and its attributes
matcher = Matcher(nlp.vocab)
# Add with ID, optional callback and pattern(s)
#pattern = [{"LOWER": "new"}, {"LOWER": "york"}]
#pattern3 = [{"POS": "VERB", "OP": "?"}, {"POS": "VERB"}]
#pattern4 = [{"POS": "VERB"},{"LOWER": "'can', 'could', 'may', 'might', 'shall', 'should', 'will', 'would', 'must'"}, {"POS": "VERB"}]
pattern1 = [{"POS": "VERB","TEXT": "can"}, {"POS": "VERB"}]
pattern2 = [{"POS": "VERB","TEXT": "could"}, {"POS": "VERB"}]
pattern3 = [{"POS": "VERB","TEXT": "may"}, {"POS": "VERB"}]
pattern4 = [{"POS": "VERB","TEXT": "might"}, {"POS": "VERB"}]
pattern5 = [{"POS": "VERB","TEXT": "shall"}, {"POS": "VERB"}]
pattern6 = [{"POS": "VERB","TEXT": "should"}, {"POS": "VERB"}]
pattern7 = [{"POS": "VERB","TEXT": "will"}, {"POS": "VERB"}]
pattern8 = [{"POS": "VERB","TEXT": "would"}, {"POS": "VERB"}]
pattern9 = [{"POS": "VERB","TEXT": "must"}, {"POS": "VERB"}]
patternn = [{"POS": "VERB","TEXT": "can", "TEXT": "could"}, {"POS": "VERB"}]

matcher.add('P1', None, pattern1)
matcher.add('P2', None, pattern2)
matcher.add('P3', None, pattern3)
matcher.add('P4', None, pattern4)
matcher.add('P5', None, pattern5)
matcher.add('P6', None, pattern6)
matcher.add('P7', None, pattern7)
matcher.add('P8', None, pattern8)
matcher.add('P9', None, pattern9)

# Match by calling the matcher on a Doc object
#doc = nlp("I can live in New York")
doc = nlp("storymaps could link the sun back to story source or colored htmlified story output refer http screencast com t g a ch x ca nt click the story name to go anywhere")
matches = matcher(doc)
# Matches are (match_id, start, end) tuples
for match_id, start, end in matches:
     # Get the matched span by slicing the Doc
     span = doc[start:end]
     print(span.text)
# 'New York'
# Output:
#   could link


# Create a df containing sentence and its output for rule 1
row_list = []

for i in range(len(df3)):

    sent = df3.loc[i,'Sent']
    year = df3.loc[i,'Year']
    output = rule1(sent)
    dict1 = {'Year':year,'Sent':sent,'Output':output}
    row_list.append(dict1)

df_rule1 = pd.DataFrame(row_list)

# Rule 1 achieves 20% result on simple sentences
output_per(df_rule1,'Output')
# Output:
#   33.33333333333333

# Create a df containing sentence and its output for rule 1
#print(df3)
row_list = []

for i in range(len(df2)):

    sent = df2.loc[i,'Sent']
    year = df2.loc[i,'Year']
    output = rule11(sent)
    dict1 = {'Year':year,'Sent':sent,'Output':output}
    row_list.append(dict1)

df_rule1 = pd.DataFrame(row_list)
df_rule1.to_csv('prueba.csv')
#print(df_rule1)
# Rule 1 achieves 20% result on simple sentences
#output_per(df_rule1,'Output')
# Output:
#   z

#   delegatingwebdriverprovider


"""
We are getting more than 20% pattern match for our rule, we can check it for all the sentences in the corpus.
"""

# Create a df containing sentence and its output for rule 1
row_list = []

# df2 contains all the sentences from all the speeches
for i in range(len(df2)):

    sent = df2.loc[i,'Sent']
    year = df2.loc[i,'Year']
    output = rule1(sent)
    dict1 = {'Year':year,'Sent':sent,'Output':output}
    row_list.append(dict1)

df_rule1_all = pd.DataFrame(row_list)

# Check rule1 output on complete speeches
output_per(df_rule1_all,'Output')
# Output:
#   43.80530973451327

"""
We are getting more than a 30% match for our rules, which means 2226 out of 7150 sentences matched this pattern. Let’s form a new dataframe containing only those sentences that have an output and then segregate the verb from the nouns:
"""

# selecting non-empty output rows
df_show = pd.DataFrame(columns=df_rule1_all.columns)

for row in range(len(df_rule1_all)):

    if len(df_rule1_all.loc[row,'Output'])!=0:
        df_show = df_show.append(df_rule1_all.loc[row,:])

# reset the index
df_show.reset_index(inplace=True)
df_show.drop('index',axis=1,inplace=True)

df_rule1_all.shape, df_show.shape
# Output:
#   ((226, 3), (99, 3))

# separate subject, verb and object

verb_dict = dict()
dis_dict = dict()
dis_list = []

# iterating over all the sentences
for i in range(len(df_show)):

    # sentence containing the output
    sentence = df_show.loc[i,'Sent']
    # year of the sentence
    year = df_show.loc[i,'Year']
    # output of the sentence
    output = df_show.loc[i,'Output']

    # iterating over all the outputs from the sentence
    for sent in output:

        # separate subject, verb and object
        n1 = sent.split()[:1]
        v = sent.split()[1]
        n2 = sent.split()[2:]

        # append to list, along with the sentence
        dis_dict = {'Sent':sentence,'Year':year,'Noun1':n1,'Verb':v,'Noun2':n2}
        dis_list.append(dis_dict)

        # counting the number of sentences containing the verb
        verb = sent.split()[1]
        if verb in verb_dict:
            verb_dict[verb]+=1
        else:
            verb_dict[verb]=1

df_sep = pd.DataFrame(dis_list)

"""
We can seperate the verb from the subject noun and object noun. This will allows us to better analyse the result.
"""

df_sep.head()
# Output:
#                                                   Sent  ...         Noun2

#   0  springstoryreporterbuilder does not expose all...  ...  [properties]

#   1  generic parameter converter for enum classes i...  ...      [fields]

#   2  support for weld context and dependency inject...  ...     [support]

#   3  rename run with annotated embedder goal to run...  ...        [name]

#   4  if story is ' excluded ' because of meta filte...  ...         [way]

#   

#   [5 rows x 5 columns]

"""
Lets take a look at the top occuring verbs used in the sentences.
"""

sort = sorted(verb_dict.items(), key = lambda d:(d[1],d[0]), reverse=True)
# top 10 most used verbs in sentence
sort[:10]
# Output:
#   [('use', 7),

#    ('add', 7),

#    ('run', 6),

#    ('provide', 5),

#    ('need', 5),

#    ('contain', 5),

#    ('support', 4),

#    ('create', 4),

#    ('see', 3),

#    ('implement', 3)]

"""
Now we can look at specific verbs to see what kind of information is prsent. For example 'welcome' and 'support' could tell us what India encourages. And verbs like 'face' could maybe tell use what kind of problems we face in the real world.
"""

# support verb
df_sep[df_sep['Verb']=='support']
# Output:
#                                                    Sent  ...           Noun2

#   34  rename embeddedstory flag to givenstory in sto...  ...  [givenstories]

#   68  reports should support configurable multiple v...  ...         [types]

#   69  reports should support configurable multiple v...  ...          [tree]

#   71  finnish language support finnish language tran...  ...  [translations]

#   

#   [4 rows x 5 columns]

# face
df_sep[df_sep['Verb']=='face']
# Output:
#                                                    Sent Year  ...  Verb      Noun2

#   38  changing the value separator of jbehave i face...  Bug  ...  face  [problem]

#   

#   [1 rows x 5 columns]

"""
By looking at the output, we can try to make out what is the context of the sentence. For example, we can see that India supports ‘efforts’, ‘viewpoints’, ‘initiatives’, ‘struggles’, ‘desires, ‘aspirations’, etc. While India believes that the world faces ‘threat’, ‘conflicts’, ‘colonialism’, ‘pandemics’, etc.

We can select sentences to explore in-depth by looking at the output. This will definitely save us a lot of time than just going over the entire text.
"""

"""
## Information Extraction #4 – Rule on Adjective Noun Structure
"""

"""
In the previous rule that we made, we extracted the noun subjects and objects, but the information did not feel complete. This is because many nouns have an adjective or a word with a compound dependency that augments the meaning of a noun. Extracting these along with the noun will give us better information about the subject and the object.

Have a look at the sample sentence below.
"""

text = 'Our people are expecting a better life.'
print(text)
doc = nlp(text)
img = displacy.render(doc, style='dep',jupyter=True)
img

#output_path = Path("./img2.svg")
#output_path.open("w", encoding="utf-8").write(img)
# Output:
#   Our people are expecting a better life.

#   <IPython.core.display.HTML object>

"""
What we are looking to achieve here is: "people","expecting" and "better life".

The code for this rule is simple, but let me walk you through how it works:
* We look for tokens that have a Noun POS tag and have subject or object dependency
* Then we look at the child nodes of these tokens and append it to the phrase only if it modifies the noun
"""

# function for rule 2
def rule2(text):

    doc = nlp(text)

    pat = []

    # iterate over tokens
    for token in doc:
        phrase = ''
        # if the word is a subject noun or an object noun
        if (token.pos_ == 'NOUN')\
            and (token.dep_ in ['dobj','pobj','nsubj','nsubjpass']):

            # iterate over the children nodes
            for subtoken in token.children:
                # if word is an adjective or has a compound dependency
                if (subtoken.pos_ == 'ADJ') or (subtoken.dep_ == 'compound'):
                    phrase += subtoken.text + ' '

            if len(phrase)!=0:
                phrase += token.text

        if  len(phrase)!=0:
            pat.append(phrase)


    return pat

# Create a df containing sentence and its output for rule 2
row_list = []

for i in range(len(df3)):

    sent = df3.loc[i,'Sent']
    year = df3.loc[i,'Year']
    # Rule 2
    output = rule2(sent)

    dict1 = {'Year':year,'Sent':sent,'Output':output}
    row_list.append(dict1)

df_rule2 = pd.DataFrame(row_list)

df_rule2.head()
# Output:
#      Year  ...                  Output

#   0  2021  ...          [test project]

#   1  2021  ...  [report xml, end tags]

#   2  2021  ...                      []

#   3  2021  ...                      []

#   4  2021  ...                      []

#   

#   [5 rows x 3 columns]

# Rule 2 output
output_per(df_rule2,'Output')
# Output:
#   8.376905106240962

"""
51% of the short sentences match this rule. We can try now check it on the entire corpus.
"""

# create a df containing sentence and its output for rule 2
row_list = []

# df2 contains all the sentences from all the speeches
for i in range(len(df2)):

    sent = df2.loc[i,'Sent']
    year = df2.loc[i,'Year']
    output = rule2(sent)
    dict1 = {'Year':year,'Sent':sent,'Output':output}
    row_list.append(dict1)

df_rule2_all = pd.DataFrame(row_list)

# check rule output on complete speeches
output_per(df_rule2_all,'Output')
# Output:
#   94.24778761061947

df_rule2_all.head(10)
# Output:
#             Year  ...                                             Output

#   0  Improvement  ...                                                 []

#   1  Improvement  ...                         [story source, story name]

#   2  New Feature  ...  [crossreference report, geeknight folks, diffe...

#   3  Improvement  ...  [usingembedder annotation, story timeout, othe...

#   4  Improvement  ...                                [system properties]

#   5          Bug  ...  [composite step, sub steps, same manner, other...

#   6  Improvement  ...                        [enum classes, enum fields]

#   7  New Feature  ...                         [reference implementation]

#   8  Improvement  ...       [embedder methods, methods useconfiguration]

#   9  Improvement  ...  [embedder goal, goal name, other goal names, p...

#   

#   [10 rows x 3 columns]

df_rule2_all.shape,df_show2.shape
# Output:
#   ((226, 3), (213, 3))

"""
Out of 7150, 5470 sentences matched our pattern rule.
"""

# Selecting non-empty outputs
df_show2 = pd.DataFrame(columns=df_rule2_all.columns)

for row in range(len(df_rule2_all)):

    if len(df_rule2_all.loc[row,'Output'])!=0:
        df_show2 = df_show2.append(df_rule2_all.loc[row,:])

# Reset the index
df_show2.reset_index(inplace=True)
df_show2.drop('index',axis=1,inplace=True)

df_show2.head(10)
# Output:
#             Year  ...                                             Output

#   0  Improvement  ...                         [story source, story name]

#   1  New Feature  ...  [crossreference report, geeknight folks, diffe...

#   2  Improvement  ...  [usingembedder annotation, story timeout, othe...

#   3  Improvement  ...                                [system properties]

#   4          Bug  ...  [composite step, sub steps, same manner, other...

#   5  Improvement  ...                        [enum classes, enum fields]

#   6  New Feature  ...                         [reference implementation]

#   7  Improvement  ...       [embedder methods, methods useconfiguration]

#   8  Improvement  ...  [embedder goal, goal name, other goal names, p...

#   9          Bug  ...  [plugin afterstories xml, beforestories xml, i...

#   

#   [10 rows x 3 columns]

"""
Now we can combine this rule along with the rule that we created previously. This will give us a better perspective of what information in present in a sentence.
"""

def rule2_mod(text,index):

    doc = nlp(text)

    phrase = ''

    for token in doc:

        if token.i == index:

            for subtoken in token.children:
                if (subtoken.pos_ == 'ADJ'):
                    phrase += ' '+subtoken.text
            break

    return phrase

# rule 1 modified function
def rule1_mod(text):

    doc = nlp(text)

    sent = []

    for token in doc:
        # root word
        if (token.pos_=='VERB'):

            phrase =''

            # only extract noun or pronoun subjects
            for sub_tok in token.lefts:

                if (sub_tok.dep_ in ['nsubj','nsubjpass']) and (sub_tok.pos_ in ['NOUN','PROPN','PRON']):

                    adj = rule2_mod(text,sub_tok.i)

                    phrase += adj + ' ' + sub_tok.text

                    # save the root word of the word
                    phrase += ' '+token.lemma_

                    # check for noun or pronoun direct objects
                    for sub_tok in token.rights:

                        if (sub_tok.dep_ in ['dobj']) and (sub_tok.pos_ in ['NOUN','PROPN']):

                            adj = rule2_mod(text,sub_tok.i)

                            phrase += adj+' '+sub_tok.text
                            sent.append(phrase)

    return sent

# create a df containing sentence and its output for modified rule 1
row_list = []

# df2 contains all the sentences from all the speeches
for i in range(len(df2)):

    sent = df2.loc[i,'Sent']
    year = df2.loc[i,'Year']
    output = rule1_mod(sent)
    dict1 = {'Year':year,'Sent':sent,'Output':output}
    row_list.append(dict1)

df_rule1_mod_all = pd.DataFrame(row_list)
# check rule1 output on complete speeches
output_per(df_rule1_mod_all,'Output')
# Output:
#   43.80530973451327

df_rule1_mod_all.head(20)
# Output:
#              Year  ...                                           Output

#   0   Improvement  ...  [ springstoryreporterbuilder expose properties]

#   1   Improvement  ...                                               []

#   2   New Feature  ...                                               []

#   3   Improvement  ...                                               []

#   4   Improvement  ...                                               []

#   5           Bug  ...                                               []

#   6   Improvement  ...                            [ you convert fields]

#   7   New Feature  ...                       [ support provide support]

#   8   Improvement  ...                                               []

#   9   Improvement  ...                         [ we keep previous name]

#   10  Improvement  ...                                               []

#   11          Bug  ...                                               []

#   12  New Feature  ...                                   [ we need way]

#   13  Improvement  ...           [ examplestablefactory load resources]

#   14  New Feature  ...                                               []

#   15  Improvement  ...                                               []

#   16  New Feature  ...                            [ ability add method]

#   17  Improvement  ...                                               []

#   18          Bug  ...                                [ line give grid]

#   19  New Feature  ...                                               []

#   

#   [20 rows x 3 columns]

"""
In the previous rule that we made, we extracted the noun subjects and objects, but the information did not feel complete. This is because many nouns have an adjective or a word with a compound dependency that augments the meaning of a noun. Extracting these along with the noun will give us better information about the subject and the object.
"""

"""
## Information Extraction #5 – Rule on Prepositions
"""

"""
Thank god for preposistions. They tell us where or when something is in relationship with something else. For example, *The people **of** India believe **in** the priciples **of** United Nations.*. Clearly extarcting phrases inclusing prepositions will give us a lot of information from the sentence. This is exactly what we are going to achieve with this rule.

Let's try to understand how this rule works by going over it on a sample sentece - "India has once again shown faith in democracy."

* We iterate over all the tokens looking for prepositions. For example *in* in this sentence.
* On encountering a preposition, we check if it has a head word that is a noun. For example the word *faith* in this sentence.
* Then we look at the child tokens of the preposition token falling on its right side. For example, the word *democracy*.

This should finally extract the phrase *faith in democracy* from the sentence. Have a look at the dependency graph of the sentence below.
"""

text = "India has once again shown faith in democracy."
print(text)
doc = nlp(text)
img = displacy.render(doc, style='dep',jupyter=True)
img

#output_path = Path("./img3.svg")
# output_path.open("w", encoding="utf-8").write(img)
# displacy.render(doc, style='dep',jupyter=True)
# Output:
#   India has once again shown faith in democracy.

#   Error: NameError: ignored

"""
Now lets apply this rule to our short sentences.
"""

# rule 3 function
def rule3(text):

    doc = nlp(text)

    sent = []

    for token in doc:

        # look for prepositions
        if token.pos_=='ADP':

            phrase = ''

            # if its head word is a noun
            if token.head.pos_=='NOUN':

                # append noun and preposition to phrase
                phrase += token.head.text
                phrase += ' '+token.text

                # check the nodes to the right of the preposition
                for right_tok in token.rights:
                    # append if it is a noun or proper noun
                    if (right_tok.pos_ in ['NOUN','PROPN']):
                        phrase += ' '+right_tok.text

                if len(phrase)>2:
                    sent.append(phrase)

    return sent

# create a df containing sentence and its output for rule 4
row_list = []

for i in range(len(df2)): ##df3

    sent = df2.loc[i,'Sent']
    year = df2.loc[i,'Year']

    # Rule 3
    output = rule3(sent)

    dict1 = {'Year':year,'Sent':sent,'Output':output}
    row_list.append(dict1)

df_rule3 = pd.DataFrame(row_list)
# Rule 3 achieves 40% result
output_per(df_rule3,'Output')
# Output:
#   Error: NameError: ignored

"""
About 48% of the sentences follow this rule.
"""

df_rule3.head(10)

"""
We can test this pattern on the entire corpus since we have good amount of sentences matching the rule.
"""

# create a df containing sentence and its output for rule 1
row_list = []

# df2 contains all the sentences from all the speeches
for i in range(len(df2)):

    sent = df2.loc[i,'Sent']
    year = df2.loc[i,'Year']
    output = rule3(sent)  # Output
    dict1 = {'Year':year,'Sent':sent,'Output':output}
    row_list.append(dict1)

df_rule3_all = pd.DataFrame(row_list)
# check rule1 output on complete speeches
output_per(df_rule3_all,'Output')

df_rule3_all.head(10)

"""
Show only those sentences that have outputs
"""

# select non-empty outputs
df_show3 = pd.DataFrame(columns=df_rule3_all.columns)

for row in range(len(df_rule3_all)):

    if len(df_rule3_all.loc[row,'Output'])!=0:
        df_show3 = df_show3.append(df_rule3_all.loc[row,:])

# reset the index
df_show3.reset_index(inplace=True)
df_show3.drop('index',axis=1,inplace=True)

df_rule3_all.shape, df_show3.shape

# separate noun, preposition and noun

prep_dict = dict()
dis_dict = dict()
dis_list = []

# iterating over all the sentences
for i in range(len(df_show3)):

    # sentence containing the output
    sentence = df_show3.loc[i,'Sent']
    # year of the sentence
    year = df_show3.loc[i,'Year']
    # output of the sentence
    output = df_show3.loc[i,'Output']

    # iterating over all the outputs from the sentence
    for sent in output:

        # separate subject, verb and object
        n1 = sent.split()[0]
        p = sent.split()[1]
        n2 = sent.split()[2:]

        # append to list, along with the sentence
        dis_dict = {'Sent':sentence,'Year':year,'Noun1':n1,'Preposition':p,'Noun2':n2}
        dis_list.append(dis_dict)

        # counting the number of sentences containing the verb
        prep = sent.split()[1]
        if prep in prep_dict:
            prep_dict[prep]+=1
        else:
            prep_dict[prep]=1

df_sep3= pd.DataFrame(dis_list)

"""
The following dataframe shows the result of the rule on the entire corpus.
"""

df_sep3.head(10)

"""
We can look at the topmost occuring prepositions in the entire corpus.
"""

sort = sorted(prep_dict.items(), key = lambda d:(d[1],d[0]), reverse=True)
sort[:10]

"""
We look at certain prepositions to explore the sentences in detail. For example the preposition 'against'. It can give us information about what India does not support.
"""

# 'against'
df_sep3[df_sep3['Preposition']=='for']

"""
Skimming over the nouns, some important phrases like:

* efforts against proliferation
* fight against terrorism, action against terrorism, war against terrorism
* dsicrimination against women
* war against poverty
* struggle against colonialism

... and so on. This should give us a fair idea about which sentences we want to explore in detail. For exmaple, *efforts against proliferation* talks about efforts towards nuclear disarmament. Or the sentence on *struggle against colonialism* talks about the historical links between India and Africa borne out of their common struggle against colonialism.
"""

df_sep3.loc[1,'Sent']

df_sep3.loc[11513,'Sent']

df_sep3.loc[11618,'Sent']

df_sep3.loc[11859,'Sent']

"""
As you can see, prepositions give us an important relationship between two nouns. And with a little domain knowledge we can easily seive through the vast data and determine what India supports or does not support and much more.
"""

"""
But at some time the output seems a bit incomplete. For example, in the sentence *efforts against proliferation*, what kind of a *proliferation* are we talking about? Certainly we need to include the modifiers attached to the nouns in the phrase, like we did in rule 2. This would definitely increase the comprehensibility of the extracted phrase.

This rule can be easily modified to include the new change. I have created a new function to extract the noun modifiers for nouns that we extracted from rule 3.
"""

# rule 0
def rule0(text, index):

    doc = nlp(text)

    token = doc[index]

    entity = ''

    for sub_tok in token.children:
        if (sub_tok.dep_ in ['compound','amod']):# and (sub_tok.pos_ in ['NOUN','PROPN']):
            entity += sub_tok.text+' '

    entity += token.text

    return entity

"""
All we have to do is call this function whenever we encounter a noun in our phrase.
"""

# rule 3 function
def rule3_mod(text):

    doc = nlp(text)

    sent = []

    for token in doc:

        if token.pos_=='ADP':

            phrase = ''
            if token.head.pos_=='NOUN':

                # appended rule
                append = rule0(text, token.head.i)
                if len(append)!=0:
                    phrase += append
                else:
                    phrase += token.head.text
                phrase += ' '+token.text

                for right_tok in token.rights:
                    if (right_tok.pos_ in ['NOUN','PROPN']):

                        right_phrase = ''
                        # appended rule
                        append = rule0(text, right_tok.i)
                        if len(append)!=0:
                            right_phrase += ' '+append
                        else:
                            right_phrase += ' '+right_tok.text

                        phrase += right_phrase

                if len(phrase)>2:
                    sent.append(phrase)


    return sent

# create a df containing sentence and its output for rule 3
row_list = []

# df2 contains all the sentences from all the speeches
for i in range(len(df_show3)):

    sent = df_show3.loc[i,'Sent']
    year = df_show3.loc[i,'Year']
    output = rule3_mod(sent)
    dict1 = {'Year':year,'Sent':sent,'Output':output}
    row_list.append(dict1)

df_rule3_mod = pd.DataFrame(row_list)

df_rule3_mod

"""
This definitely has more information than before. For example, 'impediments in economic development' instead of 'impediments in development' and 'greater transgressor of human rights' rather than 'transgressor of rights'.

Once again combining rules has given us more power and flexibility to explore only those sentences in detail that have a meaningful extracted phrase.
"""

# separate noun, preposition and noun

prep_dict = dict()
dis_dict = dict()
dis_list = []

# iterating over all the sentences
for i in range(len(df_rule3_mod)):

    # sentence containing the output
    sentence = df_rule3_mod.loc[i,'Sent']
    # year of the sentence
    year = df_rule3_mod.loc[i,'Year']
    # output of the sentence
    output = df_rule3_mod.loc[i,'Output']

    # iterating over all the outputs from the sentence
    for sent in output:

        # separate subject, verb and object
        n1 = sent[0]
        p = sent[1]
        n2 = sent[2:]

        # append to list, along with the sentence
        dis_dict = {'Sent':sentence,'Year':year,'Noun1':n1,'Preposition':p,'Noun2':n2}
        dis_list.append(dis_dict)

        # counting the number of sentences containing the verb
        prep = sent[1]
        if prep in prep_dict:
            prep_dict[verb]+=1
        else:
            prep_dict[verb]=1

df_sep3_mod = pd.DataFrame(dis_list)

df_sep3

"""
# End Notes

Information extraction is by no means an easy NLP task to perform. You need to spend time with the data to better understand its structure and what it has to offer.

In this article, we used theoretical knowledge and put it to practical use. We worked with a text dataset and tried to extract the information using traditional information extraction techniques.

We looked for key phrases and relationships in the text data to try and extract the information from the text. This type of approach requires a combination of computer and human effort to extract relevant information.
"""

