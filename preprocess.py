import nltk
nltk.download('punkt')      # for tokenization
import pandas as pd
import re
from nltk.tokenize import sent_tokenize

root = './data'    # File path for an original text file
file_name = 'pg2542.txt'
# Step 1: Read the text file with 'utf-8' encoding
with open(f'{root}/{file_name}', 'r', encoding='utf-8') as file:
    text = file.read()

# Step 2: Split the text by two newline characters to get each character's line
sections = text.split("\n\n")

# Step 3: Initialize empty lists to store data for LinebyLine dataframe
character = []
sentences = []

# Iterate through the sections
for i, section in enumerate(sections):
    # Check if the section is a character's speech
    if re.match(r'^[A-Z\s]+\.', section):
        char, sentence = section.split('\n', 1)
        character.append(char.strip().rstrip('.'))
        sentence = sentence.strip()
    else:
        character.append('DIRECTION')
        sentence = section.replace('\n', ' ').strip()
    sentences.append(sentence)

# Create a DataFrame LinebyLine
df_linebyline = pd.DataFrame({'character': character, 'sentences': sentences})
df_linebyline.reset_index(level=0, inplace=True)

# Step 4: Create SentbySent dataframe
character = []
line_type = []
sentences = []
pattern = r'(_\[.*?\]_\.?)'

for i, row in df_linebyline.iterrows():
    if row['character'] == 'DIRECTION':
        # It's a direction, so add it to the list as is
        # Tokenize into sentences
        for sent in sent_tokenize(row['sentences'].replace('\n', ' ')):
            character.append(row['character'])
            line_type.append('direction')
            sentences.append(sent)
    else:
        # It's a speech, so we need to handle directions within the speech and separate them
        # Preprocess the line to replace newlines with spaces
        line = row['sentences'].replace('\n', ' ')
        parts = re.split(pattern, line)

        for part in parts:
            part = part.strip()
            if part.startswith("_["):
                # It's a direction within the speech
                # Tokenize into sentences
                for sent in sent_tokenize(part):
                    character.append(row['character'])
                    line_type.append('direction')
                    sentences.append(sent)
            else:
                # It's a part of the speech
                # Tokenize into sentences
                for sent in sent_tokenize(part):
                    character.append(row['character'])
                    line_type.append('speech')
                    sentences.append(sent)

# Create a DataFrame SentbySent
df_sentbysent = pd.DataFrame({'character': character, 'type': line_type, 'sentences': sentences})
# Delete rows where "sentences" only includes "]_"
df_sentbysent = df_sentbysent[df_sentbysent['sentences'] != "]_"]

# Remove "_[" from "sentences"
df_sentbysent['sentences'] = df_sentbysent['sentences'].str.replace("_\[|\]_", "", regex=True)
df_sentbysent.reset_index(level=0, inplace=True)

# Write to Excel
df_linebyline.to_excel(f'{root}/LinebyLine.xlsx', index=False)
df_sentbysent.to_excel(f'{root}/SentbySent.xlsx', index=False)
