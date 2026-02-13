Repository: francotejada/automatic-traceability
File: 1_Classification_binary_2025.ipynb
Lines: 729

Estimated tokens: 13.7k

Directory structure:
└── 1_Classification_binary_2025.ipynb


"""
<a href="https://colab.research.google.com/github/francotejada/Automatic-Traceability/blob/main/Classification/1_Classification_binary_2025.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
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

from transformers import AutoTokenizer # BertTokenizer
from torch.utils.data import TensorDataset

from transformers import AutoModelForSequenceClassification #BertForSequenceClassification

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
    df = pd.read_csv('jboss features.csv')
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
#                                                summary                    type

#   0  BASubordinateCrashDuringComplete.txt byteman s...  non-feature containing

#   1  Can't find resource for bundle key com.arjuna....  non-feature containing

#   2  Could not invoke deployment method: It looks l...  non-feature containing

#   3  failure The root cause of the failure is that ...  non-feature containing

#   4  When is used with JTS then no participant for ...      feature containing

df['type'].value_counts()
# Output:
#   non-feature containing    1212

#   feature containing         379

#   Name: type, dtype: int64

possible_labels = df.type.unique()

label_dict = {}
for index, possible_label in enumerate(possible_labels):
    label_dict[possible_label] = index
label_dict
# Output:
#   {'non-feature containing': 0, 'feature containing': 1}

df['label'] = df.type.replace(label_dict)

df.head()
# Output:
#                                                summary                    type  \

#   0  BASubordinateCrashDuringComplete.txt byteman s...  non-feature containing   

#   1  Can't find resource for bundle key com.arjuna....  non-feature containing   

#   2  Could not invoke deployment method: It looks l...  non-feature containing   

#   3  failure The root cause of the failure is that ...  non-feature containing   

#   4  When is used with JTS then no participant for ...      feature containing   

#   

#      label  

#   0      0  

#   1      0  

#   2      0  

#   3      0  

#   4      1  

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
#                                           summary

#   type                   label data_type         

#   feature containing     1     train          322

#                                val             57

#   non-feature containing 0     train         1030

#                                val            182

#tokenizer = BertTokenizer.from_pretrained('distilbert-base-uncased',#'allenai/scibert_scivocab_uncased') # 'bert-base-uncased',
#                                          do_lower_case=True)
tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-base',#'allenai/scibert_scivocab_uncased') # 'bert-base-uncased',
                                                do_lower_case=True)


text_file = open("vocab.txt", "r")
new_tokens = text_file.readlines()
print(new_tokens)
print(len(new_tokens))
text_file.close()
# Output:
#   ['Acceptance \n', 'Acceptance criteria\n', 'Acceptance test\n', 'Activity\n', 'Activity model\n', 'Actor\n', 'Adequacy\n', 'Agile\n', 'Ambiguity\n', 'Artifact\n', 'Association\n', 'Attribute\n', 'Backlog\n', 'Baseline\n', 'Behavior\n', 'Behavior model\n', 'Bug\n', 'Burndown chart\n', 'Business requirement\n', 'Cardinality\n', 'Change management\n', 'Change request\n', 'Changeability\n', 'Class\n', 'Class diagram\n', 'Class model\n', 'Completeness\n', 'Composition\n', 'Conflict\n', 'Requirements conflict\n', 'Consistency\n', 'Constraint\n', 'Context\n', 'Context boundary\n', 'Context diagram\n', 'Context model\n', 'Control flow\n', 'Correctness\n', 'Customer\n', 'specification\n', 'customer\n', 'Decision table\n', 'Defect\n', 'Design\n', 'Document template\n', 'Domain\n', 'Domain model\n', 'Domain requirement\n', 'Effectiveness\n', 'Efficiency\n', 'Elaboration\n', 'Elicitation\n', 'Entity\n', 'diagram\n', 'Error\n', 'Evolutionary prototype\n', 'Exploratory prototype\n', 'Fault\n', 'Fault tolerance\n', 'Feasibility\n', 'Feature\n', 'Feature diagram\n', 'Feature model\n', 'Functional requirement\n', 'Functionality\n', 'Glossary\n', 'Goal\n', 'Goal model\n', 'Increment\n', 'Iteration\n', 'Kind of requirement\n', 'requirements\n', 'Mock-up\n', 'Model\n', 'Modeling language\n', 'Modifiability\n', 'Natural language\n', 'Necessity\n', 'requirement\n', 'Object\n', 'Object model\n', 'Performance\n', 'Persona\n', 'Priority\n', 'Prioritization\n', 'Problem\n', 'Process\n', 'Process model\n', 'Product\n', 'Product backlog\n', 'Product line\n', 'Product owner\n', 'Prototype\n', 'Prototyping\n', 'Quality\n', 'requirements\n', 'functional requirements\n', 'Refactoring\n', 'Redundancy\n', 'Release\n', 'Reliability\n', 'Requirements analysis\n', 'Requirements baseline\n', 'Requirements branching\n', 'Requirements conflict\n', 'Requirements discovery\n', 'elicitation\n', 'Requirements document\n', 'Requirements elicitation\n', 'Requirements Engineer\n', 'Engineering\n', 'stakeholders\n', 'system\n', 'management\n', 'Requirements model\n', 'negotiation\n', 'Requirements source\n', 'Requirements specification\n', 'Review\n', 'Risk\n', 'Role\n', 'Safety\n', 'Scope\n', 'Security\n', 'Semantics\n', 'Sequence diagram\n', 'Service\n', 'Specification\n', 'Sprint\n', 'Stakeholder\n', 'Standard\n', 'State machine\n', 'State machine diagram\n', 'Statechart\n', 'Synonym\n', 'Syntax\n', 'System\n', 'System boundary\n', 'System context\n', 'System requirement\n', 'specification\n', 'Tool\n', 'Traceability\n', 'UML\n', 'Unambiguity\n', 'Understandability\n', 'Usability\n', 'Use case\n', 'Use case diagram\n', 'User\n', 'User story\n', 'Validation\n', 'Variability\n', 'Variant\n', 'Variation\n', 'Verifiability\n', 'Verification\n', 'Version\n', 'View\n', 'Viewpoint\n', 'Vision\n', 'Walkthrough\n', 'Wireframe\n', 'ADC\n', 'ALU\n', 'ANSI\n', 'ASCII\n', 'abstraction\n', 'access\n', 'access time\n', 'accident\n', 'accuracy\n', 'accuracy study processor\n', 'actuator\n', 'adaptive maintenance\n', 'address\n', 'addressing exception\n', 'algorithm\n', 'algorithm analysis\n', 'alphanumeric\n', 'American National Standards Institute\n', 'American Standard Code for Information Interchange\n', 'analog\n', 'analog device\n', 'analog-to-digital converter\n', 'analysis\n', 'anomaly\n', 'application program\n', 'application software\n', 'architectural design\n', 'architecture\n', 'archival database\n', 'archive\n', 'archive file\n', 'arithmetic logic unit\n', 'arithmetic overflow\n', 'arithmetic underflow\n', 'array\n', 'as built\n', 'assemble\n', 'assembler\n', 'assembling\n', 'assembly code\n', 'assembly language\n', 'assertion\n', 'assertion checking\n', 'asynchronous\n', 'asynchronous transmission\n', 'audit\n', 'audit trail\n', 'auxiliary storage\n', 'band\n', 'bandwidth\n', 'bar code\n', 'baseline\n', 'BASIC\n', 'basic input/output system\n', 'batch\n', 'batch processing\n', 'baud\n', 'benchmark\n', 'bias\n', 'binary\n', 'BIOS\n', 'bit\n', 'bits per second\n', 'black-box testing\n', 'block\n', 'block check\n', 'block diagram\n', 'block length\n', 'block transfer\n', 'blocking factor\n', 'blueprint\n', 'bomb\n', 'boolean\n', 'boot\n', 'bootstrap\n', 'boundary value\n', 'boundary value analysis\n', 'box diagram\n', 'bps\n', 'branch\n', 'branch analysis\n', 'branch coverage\n', 'bubble chart\n', 'buffer\n', 'bug\n', 'bus\n', 'byte\n', 'C\n', 'C++\n', 'CAD\n', 'calibration\n', 'call graph\n', 'CAM\n', 'CASE\n', 'cathode ray tube\n', 'cause effect graph\n', 'cause effect graphing\n', 'CCITT\n', 'CD-ROM\n', 'central processing unit\n', 'certification\n', 'change control\n', 'change tracker\n', 'check summation\n', 'checksum\n', 'chip\n', 'CISC\n', 'client-server\n', 'clock\n', 'CMOS\n', 'CO-AX\n', 'coaxial cable\n', 'COBOL\n', 'code\n', 'code audit\n', 'code auditor\n', 'code inspection\n', 'code review\n', 'code walkthrough\n', 'coding\n', 'coding standards\n', 'comment\n', 'compact disc - read only memory\n', 'comparator\n', 'compatibility\n', 'compilation\n', 'compile\n', 'compiler\n', 'compiling\n', 'complementary metal-oxide semiconductor\n', 'completeness\n', 'complex instruction set computer\n', 'complexity\n', 'component\n', 'computer\n', 'computer aided design\n', 'computer aided manufacturing\n', 'computer aided software engineering\n', 'computer instruction set\n', 'computer language\n', 'computer program\n', 'computer science\n', 'computer system\n', 'computer system audit\n', 'computer system security\n', 'computer word\n', 'computerized system\n', 'concept phase\n', 'condition coverage\n', 'configurable\n', 'configuration\n', 'configuration audit\n', 'configuration control\n', 'configuration identification\n', 'configuration item\n', 'configuration management\n', 'consistency\n', 'consistency checker\n', 'constant\n', 'constraint analysis\n', 'Consultive Committee for International Telephony and Telegraphy\n', 'Contrast with software item\n', 'control bus\n', 'control flow\n', 'control flow analysis\n', 'control flow diagram\n', 'Control Program for Microcomputers\n', 'controller\n', 'conversational\n', 'coroutine\n', 'corrective maintenance\n', 'correctness\n', 'COTS\n', 'coverage analysis\n', 'CP/M\n', 'CPU\n', 'crash\n', 'CRC\n', 'critical control point\n', 'critical design review\n', 'criticality\n', 'criticality analysis\n', 'cross-assembler\n', 'cross-compiler\n', 'CRT\n', 'cursor\n', 'cyclic redundancy [check] code\n', 'cyclomatic complexity\n', 'DAC\n', 'data\n', 'data analysis\n', 'data bus\n', 'data corruption\n', 'data dictionary\n', 'data element\n', 'data exception\n', 'data flow analysis\n', 'data flow diagram\n', 'data integrity\n', 'data item\n', 'data set\n', 'data sink\n', 'data structure\n', 'data structure centered design\n', 'data structure diagram\n', 'data validation\n', 'database\n', 'database analysis\n', 'database security\n', 'dead code\n', 'debugging\n', 'decision coverage\n', 'decision table\n', 'default\n', 'default value\n', 'defect\n', 'defect analysis\n', 'delimiter\n', 'demodulate\n', 'demodulation\n', 'dependability\n', 'design\n', 'design description\n', 'design level\n', 'design of experiments\n', 'design phase\n', 'design requirement\n', 'design review\n', 'design specification\n', 'design standards\n', 'desk checking\n', 'detailed design\n', 'developer\n', 'development methodology\n', 'development standards\n', 'DFD\n', 'diagnostic\n', 'different software system analysis\n', 'digital\n', 'digital-to-analog converter\n', 'direct memory access\n', 'directed graph\n', 'disk\n', 'disk drive\n', 'disk operating system\n', 'diskette\n', 'DMA\n', 'documentation\n', 'documentation plan\n', 'documentation\n', 'DOS\n', 'drift\n', 'driver\n', 'duplex transmission\n', 'dynamic analysis\n', 'EBCDIC\n', 'editing\n', 'EEPROM\n', 'electrically erasable programmable read only memory\n', 'electromagnetic interference\n', 'electronic media\n', 'electrostatic discharge\n', 'embedded computer\n', 'embedded software\n', 'EMI\n', 'emulation\n', 'emulator\n', 'encapsulation\n', 'end user\n', 'enhanced small device interface\n', 'entity relationship diagram\n', 'environment\n', 'EPROM\n', 'equivalence class partitioning\n', 'erasable programmable read only memory\n', 'error\n', 'error analysis\n', 'error detection\n', 'error guessing\n', 'error seeding\n', 'ESD\n', 'ESDI\n', 'event table\n', 'evolutionary development\n', 'exception\n', 'exception\n', 'exception conditions/responses table\n', 'execution trace\n', 'extended ASCII\n', 'extended binary coded decimal interchange code\n', 'extremal test data\n', 'Fagan inspection\n', 'fail-safe\n', 'failure\n', 'failure analysis\n', 'Failure Modes and Effects Analysis\n', 'Failure Modes and Effects Criticality Analysis\n', 'fault\n', 'fault seeding\n', 'Fault Tree Analysis\n', 'FDD\n', 'feasibility study\n', 'Federal Information Processing Standards\n', 'fiber optics\n', 'field\n', 'file\n', 'file maintenance\n', 'file transfer protocol\n', 'FIPS\n', 'firmware\n', 'flag\n', 'flat file\n', 'floppy disk\n', 'floppy disk drive\n', 'flowchart or flow diagram\n', 'FMEA\n', 'FMECA\n', 'formal qualification review\n', 'FORTRAN\n', 'FTA\n', 'FTP\n', 'full duplex\n', 'function\n', 'functional analysis\n', 'functional configuration audit\n', 'functional decomposition\n', 'functional design\n', 'functional requirement\n', 'GB\n', 'gigabyte\n', 'graph\n', 'graphic software specifications\n', 'half duplex\n', 'handshake\n', 'hard copy\n', 'hard disk drive\n', 'hard drive\n', 'hardware\n', 'hazard\n', 'hazard analysis\n', 'hazard probability\n', 'hazard severity\n', 'HDD\n', 'hertz\n', 'hexadecimal\n', 'hierarchical decomposition\n', 'hierarchy of input-processing-output\n', 'hierarchy of input-processing-output chart\n', 'high-level language\n', 'HIPO\n', 'Hz\n', 'I/0\n', 'I/O port\n', 'IC\n', 'IDE\n', 'IEC\n', 'IEEE\n', 'implementation\n', 'implementation phase\n', 'implementation requirement\n', 'incremental development\n', 'incremental integration\n', 'industry standard\n', 'infeasible path\n', 'information hiding\n', 'input-process-output chart\n', 'input-processing-output\n', 'input/output\n', 'inspection\n', 'installation\n', 'installation and checkout phase\n', 'installation qualification\n', 'Institute of Electrical and Electronic Engineers\n', 'instruction\n', 'instruction set\n', 'instrumentation\n', 'integrated circuit\n', 'integrated drive electronics\n', 'interactive\n', 'interface\n', 'interface analysis\n', 'interface requirement\n', 'International Electrotechnical Commission\n', 'International Organization for Standardization\n', 'International Standards Organization\n', 'International Telecommunications Union - Telecommunications Standards Section\n', 'interpret\n', 'interpreter\n', 'interrupt\n', 'interrupt analyzer\n', 'invalid inputs\n', 'ISO\n', 'ITU-TSS\n', 'JCL\n', 'job\n', 'job control language\n', 'KB\n', 'Kermit\n', 'key\n', 'key element\n', 'kilobyte\n', 'KLOC\n', 'ladder logic\n', 'LAN\n', 'language\n', 'large scale integration\n', 'latency\n', 'latent defect\n', 'life cycle\n', 'life cycle methodology\n', 'linkage editor\n', 'loader\n', 'local area network\n', 'logic analysis\n', 'longitudinal redundancy check\n', 'low-level language\n', 'LSI\n', 'machine code\n', 'machine language\n', 'macro\n', 'macroinstruction\n', 'main memory\n', 'main program\n', 'mainframe\n', 'maintainability\n', 'maintenance\n', 'MAN\n', 'MB\n', 'Mb\n', 'mean time between failures\n', 'mean time to failure\n', 'mean time to repair\n', 'measurable\n', 'measure\n', 'measurement\n', 'medium scale integration\n', 'megabit\n', 'megabyte\n', 'megahertz\n', 'memory\n', 'menu\n', 'metal-oxide semiconductor\n', 'metal-oxide semiconductor field effect transistor\n', 'metric based test data generation\n', 'metric\n', 'metropolitan area network\n', 'MHz\n', 'microcode\n', 'microcomputer\n', 'microprocessor\n', 'million instructions per second\n', 'minicomputer\n', 'MIPS\n', 'mishap\n', 'mnemonic\n', 'modeling\n', 'modem\n', 'modem access\n', 'modifiability\n', 'modular decomposition\n', 'modular software\n', 'modularity\n', 'modulate\n', 'modulation\n', 'module\n', 'module interface table\n', 'MOS\n', 'MOSFET\n', 'MSI\n', 'MTBF\n', 'MTTF\n', 'MTTR\n', 'multi-processing\n', 'multi-programming\n', 'multi-tasking\n', 'multiple condition coverage\n', 'multiplexer\n', 'multipurpose systems\n', 'mutation analysis\n', 'n-channel MOS\n', 'National Bureau of Standards\n', 'National Institute for Standards and Technology\n', 'NBS\n', 'network\n', 'network database\n', 'nibble\n', 'NIST\n', 'NMI\n', 'NMOS\n', 'node\n', 'non-maskable interrupt\n', 'noncritical code analysis\n', 'nonincremental integration\n', 'null\n', 'null data\n', 'null string\n', 'object\n', 'object code\n', 'object oriented design\n', 'object oriented language\n', 'object oriented programming\n', 'object program\n', 'OCR\n', 'octal\n', 'OEM\n', 'on-line\n', 'OOP\n', 'operating system\n', 'operation and maintenance phase\n', 'operation exception\n', 'operator\n', 'optical character recognition\n', 'optical fiber\n', 'optimization\n', 'Oracle\n', 'original equipment manufacturer\n', 'overflow\n', 'overflow exception\n', 'paging\n', 'PAL\n', 'parallel\n', 'parallel processing\n', 'parameter\n', 'parity\n', 'parity bit\n', 'parity check\n', 'Pascal\n', 'password\n', 'patch\n', 'path\n', 'path analysis\n', 'path coverage\n', 'PC\n', 'PCB\n', 'PDL\n', 'perfective maintenance\n', 'performance requirement\n', 'peripheral device\n', 'peripheral equipment\n', 'personal computer\n', 'physical configuration audit\n', 'physical requirement\n', 'pixel\n', 'PLA\n', 'platform\n', 'PLD\n', 'PMOS\n', 'polling\n', 'positive channel MOS\n', 'precision\n', 'preliminary design\n', 'preliminary design review\n', 'printed circuit board\n', 'production database\n', 'program\n', 'program design language\n', 'program mutation\n', 'programmable array logic\n', 'programmable logic array\n', 'programmable logic device\n', 'programmable read only memory\n', 'programming language\n', 'programming standards\n', 'programming style analysis\n', 'project plan\n', 'PROM\n', 'PROM programmer\n', 'proof of correctness\n', 'protection exception\n', 'protocol\n', 'prototyping\n', 'pseudocode\n', 'QA\n', 'QC\n', 'qualification\n', 'quality assurance\n', 'quality control\n', 'radiofrequency interference\n', 'RAM\n', 'random access memory\n', 'range check\n', 'rapid prototyping\n', 'read only memory\n', 'real time\n', 'real time processing\n', 'record\n', 'record of change\n', 'recursion\n', 'reduced instruction set computer\n', 'region\n', 'register\n', 'regression analysis and testing\n', 'relational database\n', 'release\n', 'reliability\n', 'reliability assessment\n', 'requirement\n', 'requirements analysis\n', 'requirements phase\n', 'requirements review\n', 'retention period\n', 'retrospective trace\n', 'revalidation\n', 'review\n', 'revision number\n', 'RFI\n', 'RISC\n', 'risk\n', 'risk assessment\n', 'robustness\n', 'ROM\n', 'routine\n', 'RS-232-C\n', 'safety\n', 'safety critical\n', 'safety critical computer software components\n', 'SCSI\n', 'security\n', 'sensor\n', 'serial\n', 'server\n', 'service program\n', 'servomechanism\n', 'severity\n', 'side effect\n', 'simulation\n', 'simulation analysis\n', 'simulator\n', 'sizing\n', 'sizing and timing analysis\n', 'small computer systems interface\n', 'small scale integration\n', 'software\n', 'software audit\n', 'software characteristic\n', 'software configuration item\n', 'software design description\n', 'software developer\n', 'software development notebook\n', 'software development plan\n', 'software development process\n', 'software diversity\n', 'software documentation\n', 'software element\n', 'software element analysis\n', 'software engineering\n', 'software engineering environment\n', 'software hazard analysis\n', 'software item\n', 'software life cycle\n', 'software reliability\n', 'software requirements specification\n', 'software review\n', 'software safety change analysis\n', 'software safety code analysis\n', 'software safety design analysis\n', 'software safety requirements analysis\n', 'software safety test analysis\n', 'SOPs\n', 'source code\n', 'source program\n', 'spaghetti code\n', 'special test data\n', 'specification\n', 'specification analysis\n', 'specification tree\n', 'specification\n', 'spiral model\n', 'SQL\n', 'SSI\n', 'ST-506\n', 'standard operating procedures\n', 'state\n', 'state diagram\n', 'state-transition table\n', 'statement coverage\n', 'static analysis\n', 'static analyzer\n', 'stepwise refinement\n', 'storage device\n', 'string\n', 'structure chart\n', 'structured design\n', 'structured programming\n', 'structured query language\n', 'stub\n', 'subprogram\n', 'subroutine\n', 'subroutine trace\n', 'support software\n', 'symbolic execution\n', 'symbolic trace\n', 'synchronous\n', 'synchronous transmission\n', 'syntax\n', 'system\n', 'system administrator\n', 'system analysis\n', 'system design\n', 'system design review\n', 'system documentation\n', 'system integration\n', 'system life cycle\n', 'system manager\n', 'system safety\n', 'system software\n', 'tape\n', 'TB\n', 'TCP/IP\n', 'telecommunication system\n', 'terabyte\n', 'terminal\n', 'test\n', 'test case\n', 'test case generator\n', 'test design\n', 'test documentation\n', 'test driver\n', 'test harness\n', 'test incident report\n', 'test item\n', 'test log\n', 'test phase\n', 'test plan\n', 'test procedure\n', 'test readiness review\n', 'test report\n', 'test result analyzer\n', 'testability\n', 'testing\n', 'time sharing\n', 'timing\n', 'timing analyzer\n', 'timing and sizing analysis\n', 'top-down design\n', 'touch screen\n', 'touch sensitive\n', 'trace\n', 'traceability\n', 'traceability analysis\n', 'traceability matrix\n', 'transaction\n', 'transaction analysis\n', 'transaction flowgraph\n', 'transaction matrix\n', 'transform analysis\n', 'translation\n', 'transmission control protocol\n', 'Internet protocol\n', 'trojan horse\n', 'truth table\n', 'tuning\n', 'twisted pair\n', 'unambiguous\n', 'underflow\n', 'underflow exception\n', 'unit\n', 'UNIX\n', 'usability\n', 'user\n', "user's guide\n", 'utility program\n', 'utility software\n', 'V&V\n', 'valid\n', 'valid input\n', 'validate\n', 'validation\n', 'validation protocol\n', 'variable\n', 'variable trace\n', 'VAX\n', 'vendor\n', 'verifiable\n', 'verification\n', 'verify\n', 'version\n', 'version number\n', 'very large scale integration\n', 'virtual address extension\n', 'virtual memory system\n', 'virus\n', 'VLSI\n', 'VMS\n', 'volume\n', 'VV&T\n', 'walkthrough\n', 'WAN\n', 'watchdog timer\n', 'waterfall model\n', 'white-box testing\n', 'wide area network\n', 'word\n', 'workaround\n', 'workstation\n', 'worm\n', 'Xmodem\n', 'Ymodem\n', 'Zmodem\n', 'A/B testing\n', 'abnormal end\n', 'abuse case\n', 'acceptance  criteria\n', 'acceptance  testing\n', 'acceptance test-driven  development\n', 'accessibility\n', 'account harvesting\n', 'accountability\n', 'acting\n', 'actual result\n', 'adaptability\n', 'adversarial  example\n', 'adversarial testing\n', 'Agile Manifesto\n', 'Agile software development\n', 'Agile testing\n', 'agile testing quadrants\n', 'alpha testing\n', 'analytical test strategy\n', 'analyzability\n', 'anomaly\n', 'anti-malware\n', 'anti-pattern\n', 'API testing\n', 'appropriateness recognizability\n', 'assessment  report\n', 'assessor\n', 'atomic  condition\n', 'attack vector\n', 'attacker\n', 'audio testing\n', 'audit\n', 'authentication\n', 'authorization\n', 'automation code defect density\n', 'automotive  safety integrity level\n', 'automotive SPICE\n', 'availability\n', 'back-to-back  testing\n', 'balanced scorecard\n', 'behavior-driven development\n', 'beta testing\n', 'black-box test technique\n', 'botnet\n', 'boundary  value\n', 'boundary  value analysis\n', 'branch\n', 'bug hunting\n', 'build verification  test\n', 'call graph\n', 'Capability  Maturity Model Integration\n', 'capacity\n', 'capacity testing\n', 'capture/playback\n', 'cause-effect  diagram\n', 'cause-effect  graph\n', 'certification\n', 'change management\n', 'change-related testing\n', 'checklist-based review\n', 'checklist-based testing\n', 'classification tree\n', 'classification tree technique\n', 'CLI testing\n', 'closed-loop-system\n', 'code injection\n', 'codependent  behavior\n', 'coding standard\n', 'combinatorial testing\n', 'command-line interface\n', 'commercial off-the-shelf\n', 'compatibility\n', 'complexity\n', 'compliance\n', 'compliance testing\n', 'component integration testing\n', 'computer  forensics\n', 'concurrency\n', 'concurrency  testing\n', 'condition coverage\n', 'condition testing\n', 'confidence interval\n', 'confidentiality\n', 'configuration  management\n', 'configuration item\n', 'confirmation  testing\n', 'connectivity\n', 'consultative  test strategy\n', 'content-based  model\n', 'context  of use\n', 'continuous  integration\n', 'continuous  representation\n', 'continuous testing\n', 'contractual  acceptance testing\n', 'control chart\n', 'control flow\n', 'control flow analysis\n', 'control flow testing\n', 'convergence  metric\n', 'corporate  dashboard\n', 'cost of quality\n', 'coverage\n', 'coverage  item\n', 'coverage criteria\n', 'critical success factor\n', 'Critical Testing Processes\n', 'cross-browser  compatibility\n', 'cross-site  scripting\n', 'crowd testing\n', 'custom  tool\n', 'cyclomatic  complexity\n', 'dashboard\n', 'data  privacy\n', 'data flow analysis\n', 'data obfuscation\n', 'data-driven testing\n', 'debugging\n', 'debugging  tool\n', 'decision\n', 'decision coverage\n', 'decision table testing\n', 'decision testing\n', 'defect\n', 'defect  density\n', 'Defect Detection Percentage\n', 'defect management\n', 'defect management committee\n', 'defect report\n', 'defect taxonomy\n', 'defect-based  test technique\n', 'definition-use pair\n', 'demilitarized  zone\n', 'Deming cycle\n', 'denial of service\n', 'device-based testing\n', 'diagnosing\n', 'driver\n', 'dynamic  analysis\n', 'dynamic testing\n', 'Effect Analysis\n', 'effectiveness\n', 'efficiency\n', 'egon at\n', 'egon at\n', 'emotional intelligence\n', 'emulator\n', 'encryption\n', 'end-to-end testing\n', 'endurance testing\n', 'entry criteria\n', 'environment  model\n', 'epic\n', 'equivalence  partition\n', 'equivalence  partitioning\n', 'equivalent manual test effort\n', 'error\n', 'error guessing\n', 'escaped defect\n', 'establishing\n', 'ethical hacker\n', 'European Foundation for Quality Management  excellence  model\n', 'exit criteria\n', 'expected  result\n', 'experience-based test technique\n', 'experience-based testing\n', 'expert usability  review\n', 'exploratory testing\n', 'Extreme Programming\n', 'failed\n', 'failover\n', 'failure\n', 'Failure blade and Effect Analysis\n', 'failure mode\n', 'failure rate\n', 'false-negative result\n', 'false-positive result\n', 'fault injection\n', 'fault seeding\n', 'fault seeding tool\n', 'fault tolerance\n', 'Fault Tree  Analysis\n', 'feature-driven development\n', 'field testing\n', 'finding\n', 'firewall\n', 'follow-up test case\n', 'formal review\n', 'formative evaluation\n', 'functional  testing\n', 'functional appropriateness\n', 'functional completeness\n', 'functional correctness\n', 'functional safety\n', 'functional suitability\n', 'fuzz testing\n', 'generic test automation architecture\n', 'Goal Question  Metric\n', 'graphical user interface\n', 'GUI  testing\n', 'hacker\n', 'hardware in the  loop\n', 'hashing\n', 'heuristic\n', 'heuristic  evaluation\n', 'high-level test case\n', 'human-centered design\n', 'hyperlink\n', 'hyperlink test tool\n', 'IDEAL\n', 'impact analysis\n', 'incremental development  model\n', 'independence  of testing\n', 'independent  test lab\n', 'indicator\n', 'informal review\n', 'information assurance\n', 'initiating\n', 'input  data testing\n', 'insider threat\n', 'insourced  testing\n', 'inspection\n', 'installability\n', 'integration  testing\n', 'integrity\n', 'interface testing\n', 'interoperability\n', 'intrusion detection  system\n', 'iterative development  model\n', 'keyword-driven testing\n', 'lead assessor\n', 'learnability\n', 'learning\n', 'level of intrusion\n', 'level test plan\n', 'linear scripting\n', 'load generation\n', 'load generator\n', 'load management\n', 'load profile\n', 'load testing\n', 'low-level test case\n', 'maintenance\n', 'maintenance testing\n', 'malware\n', 'malware scanning\n', 'management  review\n', 'manufacturing-based quality\n', 'master test plan\n', 'math testing\n', 'maturity\n', 'maturity level\n', 'maturity model\n', 'MBT model\n', 'mean time between failures\n', 'mean time to failure\n', 'mean time to repair\n', 'measure\n', 'measurement\n', 'memory leak\n', 'metamorphic  relation\n', 'metamorphic  testing\n', 'method table\n', 'methodical  test strategy\n', 'metric\n', 'mind map\n', 'ML functional performance\n', 'ML functional performance criteria\n', 'ML functional performance metrics\n', 'ML model\n', 'ML model testing\n', 'model coverage\n', 'model in the  loop\n', 'model-based test strategy\n', 'model-based testing\n', 'moderator\n', 'modifiability\n', 'modified condition/decision coverage\n', 'modified condition/decision testing\n', 'modularity\n', 'monitoring  tool\n', 'multiplayer testing\n', 'multiple condition coverage\n', 'multiple condition testing\n', 'Myers-Briggs  Type Indicator\n', 'N-switch coverage\n', 'negative testing\n', 'neighborhood  integration testing\n', 'network zone\n', 'neuron coverage\n', 'non-functional testing\n', 'non-repudiation\n', 'offline MBT\n', 'online MBT\n', 'open-loop-system\n', 'operability\n', 'operational  profile\n', 'operational  profiling\n', 'operational acceptance  testing\n', 'outsourced  testing\n', 'pair testing\n', 'pairwise  integration  testing\n', 'par  sheet testing\n', 'Pareto analysis\n', 'pass/fail criteria\n', 'passed\n', 'password  cracking\n', 'path\n', 'path testing\n', 'peer review\n', 'penetration testing\n', 'performance  efficiency\n', 'performance  indicator\n', 'performance  testing\n', 'performance  testing tool\n', 'perspective-based reading\n', 'pharming\n', 'phase containment\n', 'phishing\n', 'planning poker\n', 'player perspective testing\n', 'pointer\n', 'portability\n', 'post-release  testing\n', 'postcondition\n', 'precondition\n', 'priority\n', 'PRISMA\n', 'probe effect\n', 'process assessment\n', 'process model\n', 'process-compliant test strategy\n', 'process-driven scripting\n', 'product risk\n', 'product-based  quality\n', 'project risk\n', 'proximity-based testing\n', 'pseudo-oracle\n', 'quality\n', 'quality  risk\n', 'quality assurance\n', 'quality characteristic\n', 'quality control\n', 'quality function  deployment\n', 'quality management\n', 'RACI matrix\n', 'ramp-down\n', 'ramp-up\n', 'random testing\n', 'Rational Unified Process\n', 'reactive test  strategy\n', 'reactive testing\n', 'reconnaissance\n', 'recoverability\n', 'regression testing\n', 'regression-averse test strategy\n', 'regulatory acceptance testing\n', 'reliability\n', 'reliability growth model\n', 'remote test lab\n', 'replaceability\n', 'requirement\n', 'requirements-based testing\n', 'resource  utilization\n', 'retrospective  meeting\n', 'reusability\n', 'review\n', 'review plan\n', 'reviewer\n', 'risk\n', 'risk analysis\n', 'risk assessment\n', 'risk identification\n', 'risk impact\n', 'risk level\n', 'risk likelihood\n', 'risk management\n', 'risk mitigation\n', 'risk-based testing\n', 'robustness\n', 'role-based  review\n', 'root cause\n', 'root cause analysis\n', 'safety integrity level\n', 'salting\n', 'scalability\n', 'scalability testing\n', 'scenario-based review\n', 'scribe\n', 'script kiddie\n', 'scripted testing\n', 'scrum\n', 'security\n', 'security  testing\n', 'security attack\n', 'security audit\n', 'security policy\n', 'security procedure\n', 'security risk\n', 'security vulnerability\n', 'sequential development  model\n', 'service virtualization\n', 'session-based  test  management\n', 'session-based  testing\n', 'severity\n', 'short-circuiting\n', 'sign change coverage\n', 'sign-sign coverage\n', 'simulator\n', 'smoke test\n', 'social engineering\n', 'softwa  e development  lifecycle\n', 'software  in the  loop\n', 'software  lifecycle\n', 'software  process  improvement\n', 'software qualification  test\n', 'Software Usability Measurement  Inventory\n', 'source test case\n', 'specification  by  example\n', 'spike testing\n', 'SQL injection\n', 'staged representation\n', 'standard\n', 'standard-compliant test strategy\n', 'state transition testing\n', 'statement\n', 'statement coverage\n', 'statement testing\n', 'static  analysis\n', 'static testing\n', 'stress testing\n', 'structural coverage\n', 'structured  scripting\n', 'stub\n', 'summative  evaluation\n', 'System  Usability  Scale\n', 'system hardening\n', 'system integration testing\n', 'system of  systems\n', 'system qualification test\n', 'system testing\n', 'system under test\n', 'Systematic  Test  and Evaluation  Process\n', 'technical review\n', 'test\n', 'test adaptation layer\n', 'test analysis\n', 'test approach\n', 'test architect\n', 'test automation\n', 'test automation  architecture\n', 'test automation  engineer\n', 'test automation  framework\n', 'test automation  manager\n', 'test automation  solution\n', 'test automation  strategy\n', 'test basis\n', 'test case\n', 'test case explosion\n', 'test charter\n', 'test closure\n', 'test completion\n', 'test completion report\n', 'test condition\n', 'test control\n', 'test cycle\n', 'test data\n', 'test data preparation\n', 'test data preparation  tool\n', 'test definition layer\n', 'test design\n', 'test design specification\n', 'test director\n', 'test environment\n', 'test estimation\n', 'test execution\n', 'test execution  schedule\n', 'test execution automation\n', 'test execution layer\n', 'test execution tool\n', 'test generation  layer\n', 'test harness\n', 'test hook\n', 'test implementation\n', 'test improvement  plan\n', 'test infrastructure\n', 'test item\n', 'test leader\n', 'test level\n', 'test log\n', 'test logging\n', 'test management\n', 'test management tool\n', 'test manager\n', 'Test Maturity Model integration\n', 'test mission\n', 'test model\n', 'test monitoring\n', 'test object\n', 'test objective\n', 'test plan\n', 'test planning\n', 'Test Point Analysis\n', 'test policy\n', 'test procedure\n', 'test process\n', 'test process  improvement  manifesto\n', 'test process improvement\n', 'test process improver\n', 'test progress report\n', 'test pyramid\n', 'test report\n', 'test reporting\n', 'test result\n', 'test run\n', 'test schedule\n', 'test script\n', 'test selection criteria\n', 'test session\n', 'test specification\n', 'test strategy\n', 'test suite\n', 'test technique\n', 'test tool\n', 'test type\n', 'test-driven development\n', 'test-first approach\n', 'testability\n', 'tester\n', 'testing\n', 'testware\n', 'think aloud usability testing\n', 'think time\n', 'threshold  coverage\n', 'time behavior\n', 'Total Quality  Management\n', 'tour\n', 'TPI  Next\n', 'traceability\n', 'traceability  matrix\n', 'transactional analysis\n', 'transcendent-based quality\n', 'unit test framework\n', 'usability\n', 'usability  evaluation\n', 'usability  lab\n', 'usability  requirement\n', 'usability test participant\n', 'usability test script\n', 'usability test session\n', 'usability test task\n', 'usability testing\n', 'user  interface\n', 'user acceptance  testing\n', 'user error  protection\n', 'user experience\n', 'user interface aesthetics\n', 'user interface guideline\n', 'user story\n', 'user story testing\n', 'user survey\n', 'user-agent based testing\n', 'user-based  quality\n', 'V-model\n', 'validation\n', 'value change coverage\n', 'value-based  quality\n', 'values of several parameters\n', 'verification\n', 'virtual test environment\n', 'virtual user\n', 'visual testing\n', 'vulnerability  scanner\n', 'walkthrough\n', 'Web Content  Accessibility  Guidelines\n', 'Website Analysis and Measurement  Inventory\n', 'white-box  test technique\n', 'white-box testing\n', 'Wideband  Delphi\n', 'wild pointer\n', 'XiL  test environment\n']

#   1543


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
# Output:
#   Truncation was not explicitly activated but `max_length` is provided a specific value, please use `truncation=True` to explicitly truncate examples to max length. Defaulting to 'longest_first' truncation strategy. If you encode pairs of sequences (GLUE-style) with the tokenizer you can select this strategy more precisely by providing a specific strategy to `truncation`.

#   /usr/local/lib/python3.10/dist-packages/transformers/tokenization_utils_base.py:2393: FutureWarning: The `pad_to_max_length` argument is deprecated and will be removed in a future version, use `padding=True` or `padding='longest'` to pad to the longest sequence in the batch, or use `padding='max_length'` to pad to a max length. In this case, you can give a specific length with `max_length` (e.g. `max_length=45`) or leave max_length to None to pad to the maximal input size of the model (e.g. 512 for Bert).

#     warnings.warn(


dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)
dataset_val = TensorDataset(input_ids_val, attention_masks_val, labels_val)

len(dataset_train), len(dataset_val)
# Output:
#   (1352, 239)

#model = BertForSequenceClassification.from_pretrained("distilbert-base-uncased", #"allenai/scibert_scivocab_uncased", #"bert-base-uncased",
#                                                      num_labels=len(label_dict),
#                                                      output_attentions=False,
#                                                      output_hidden_states=False)
model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-v3-base", #"allenai/scibert_scivocab_uncased", #"bert-base-uncased",
                                                            num_labels=len(label_dict),
                                                            output_attentions=False,
                                                            output_hidden_states=False)
# Output:
#   Some weights of DistilBertForSequenceClassification were not initialized from the model checkpoint at distilbert-base-uncased and are newly initialized: ['pre_classifier.bias', 'classifier.weight', 'classifier.bias', 'pre_classifier.weight']

#   You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.


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
# Output:
#   /usr/local/lib/python3.10/dist-packages/transformers/optimization.py:411: FutureWarning: This implementation of AdamW is deprecated and will be removed in a future version. Use the PyTorch implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this warning

#     warnings.warn(


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
#   cuda


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
filter2 = a["type"]=="feature containing"
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

