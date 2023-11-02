import numpy as np
import pandas as pd
import spacy 

db = pd.read_csv('../database/train_B_text_processed_distance.csv')

db = db.drop(['Id'], axis=1)
db['Fake/Real'] = db['Fake/Real'].replace({'real': 0, 'fake': 1})

def process_title(title):
    quotes = title.count('"')
    s_quotes = title.count("'")
    title = title.replace('"','').replace("'",'')
    dots = title.count("...")
    semicolon = title.count(";")
    parenthesis = title.count("(") + title.count(")")
    words = title.split()
    
    upper_letters = 0
    for word in words:
        if word[0].isupper():
            upper_letters += 1

    return quotes,s_quotes,upper_letters

db[['quotes', 's_quotes', 'upper_letters']] = db['Title'].apply(process_title).apply(pd.Series)

#print(db)

# get a dictionary with the value of column title: closest_emb 
titles = db['Title'].to_list()
closest_emb = db['closest_emb'].to_list()
distances_dict = dict(zip(titles, closest_emb))


######################### Just playing

# Turn db['Title'] into a python list of tuples (string, fake/real)
# [(title1, fake/real), (title2, fake/real), ...]
titles = db['Title'].to_list()
fake_real = db['Fake/Real'].to_list()
titles = list(zip(titles, fake_real))

print("Total: ", len(titles))

filter = []
count = 0
for i in range(len(titles)):
    title, label = titles[i]
    if '"' == title[0] and label == 1:
        count += 1
        pass
    elif '"' == title[0] and label != 1:
        print(title, label)
        filter.append(titles[i])
    else:
        filter.append(titles[i])
        
print('Count ": ', count)

print("......................................................")

titles = filter
filter = []
count = 0
for i in range(len(titles)):
    title, label = titles[i]
    if "..." in title and label == 0:
        count += 1
        pass
    elif "..." in title and label != 0:
        print(title, label)
        filter.append(titles[i])
    else:
        filter.append(titles[i])
        
print('Count ...: ', count)

print("......................................................")



'''
titles = filter
filter = []
count = 0
for i in range(len(titles)):
    title, label = titles[i]
    if "'" in title and label == 0:
        count += 1
        pass
    elif "'" in title and label != 0:
        print(title, label)
        filter.append(titles[i])
    else:
        filter.append(titles[i])
        
print("Count ': ", count)

print("......................................................")

titles = filter
filter = []
count = 0
for i in range(len(titles)):
    title, label = titles[i]
    if any(char.isdigit() for char in title):
        if label == 0:
            count += 1
            pass
        elif label != 0:
            print(title, label)
            filter.append(titles[i])
    else:
        filter.append(titles[i])
        
print("Count number: ", count)

print("......................................................")

titles = filter
filter = []
count = 0
for i in range(len(titles)):
    title, label = titles[i]
    if ',' in title:
        if label == 0:
            count += 1
            pass
        elif label != 0:
            print(title, label)
            filter.append(titles[i])
    else:
        filter.append(titles[i])
        
print("Count ,: ", count)

print("......................................................")

titles = filter
filter = []
count = 0
for i in range(len(titles)):
    title, label = titles[i]
    if "'s" in title:
        if label == 1:
            count += 1
            pass
        elif label != 1:
            print(title, label)
            filter.append(titles[i])
    else:
        filter.append(titles[i])
        
print("Count 's: ", count)

print("......................................................")

titles = filter
filter = []
count = 0
for i in range(len(titles)):
    title, label = titles[i]
    if ":" in title:
        if label == 0:
            count += 1
            pass
        elif label != 0:
            print(title, label)
            filter.append(titles[i])
    else:
        filter.append(titles[i])
        
print("Count : : ", count)

print("......................................................")

titles = filter
filter = []
count = 0
for i in range(len(titles)):
    title, label = titles[i]
    if "-" in title:
        if label == 0:
            count += 1
            pass
        elif label != 0:
            print(title, label)
            filter.append(titles[i])
    else:
        filter.append(titles[i])
        
print("Count : : ", count)

print("......................................................")

titles = filter
filter = []
count = 0
for i in range(len(titles)):
    title, label = titles[i]
    words = title.split()
    all_upper = True
    for word in words:
        # check only when it is an alphabetical character
        if word[0].isalpha():
            if not word[0].isupper():
                all_upper = False
                break
    if not all_upper:
        if label == 0:
            count += 1
            pass
        elif label != 0:
            print(title, label)
            filter.append(titles[i])
    else:
        filter.append(titles[i])
        
print("Count noun lower ", count)
'''
print("......................................................")

nlp = spacy.load("en_core_web_sm") 

titles = filter
filter = []
count = 0
for i in range(len(titles)):
    title, label = titles[i]
    # check if a noun or verb are writte in lowercase
    lower_noun = False
    for token in nlp(title):
        if token.pos_ == 'NOUN' or token.pos_ == 'VERB':
            if token.text[0].islower():
                lower_noun = True
                break
    if lower_noun:
        if label == 0:
            count += 1
            pass
        elif label != 0:
            print(title, label)
            filter.append(titles[i])
    else:
        filter.append(titles[i])
        
print("Count lower noun/verb : ", count)

print("......................................................")

titles = filter
filter = []
count = 0
for i in range(len(titles)):
    title, label = titles[i]
    if len(title.split()) > 12:
        if label == 0:
            count += 1
            pass
        elif label != 0:
            print(title, label)
            filter.append(titles[i])
    else:
        filter.append(titles[i])

print("Count length > 12 : ", count)

print("......................................................")

titles = filter
filter = []
count = 0
for i in range(len(titles)):
    title, label = titles[i]
    if any(char.isdigit() for char in title):
        if label == 0:
            count += 1
            pass
        elif label != 0:
            print(title, label)
            filter.append(titles[i])
    else:
        filter.append(titles[i])
        
print("Count number: ", count)

print("......................................................")

for i in range(len(titles)):
    title, label = titles[i]
    if distances_dict[title] > 0.07 and label == 1:
        # compute mean of words length
        words = title.split()
        mean = 0
        for word in words:
            mean += len(word)
        mean /= len(words)
        print(mean, title, label, distances_dict[title] )

'''

print("Remaining: ", len(filter))
'''




'''
# Count sentences that have all first letters uppercase
count_real = 0
count_fake = 0
for i in range(len(filter)):
    title, label = filter[i]
    if # condition
        if label == 0:
            count_real += 1
        else:
            count_fake += 1
            print(title, label)
print("Count real: ", count_real, " fake: ", count_fake)
'''

'''
def count_spacy_token(spacy_tokens, target_token):
    count = 0
    for token in spacy_tokens:
        if str(token.pos_) == target_token:
            count += 1
    return count




true_titles = []
false_titles = []
for i in range(len(filter)):
    title, label = filter[i]
    if label == 0:
        true_titles.append(title)
    else:
        false_titles.append(title)

for title in true_titles:
    print(title)
    aux = nlp(title.lower())
    #print("    ", {'ADJ': count_spacy_token(aux, 'ADJ'), 'ADV': count_spacy_token(aux, 'ADV'), 'NOUN': count_spacy_token(aux, 'NOUN'), 'VERB': count_spacy_token(aux, 'VERB'), 'PROPN': count_spacy_token(aux, 'PROPN'), 'PUNCT': count_spacy_token(aux, 'PUNCT')})
print("   -----------------------------------------------------------   ")
for title in false_titles:
    print(title)
    aux = nlp(title.lower())
    #print("    ", {'ADJ': count_spacy_token(aux, 'ADJ'), 'ADV': count_spacy_token(aux, 'ADV'), 'NOUN': count_spacy_token(aux, 'NOUN'), 'VERB': count_spacy_token(aux, 'VERB'), 'PROPN': count_spacy_token(aux, 'PROPN'), 'PUNCT': count_spacy_token(aux, 'PUNCT')})
print("len true: ", len(true_titles), " len false: ", len(false_titles))

'''
