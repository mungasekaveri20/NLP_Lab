
import spacy
from nltk.stem import PorterStemmer
import nltk
nltk.download('punkt')



print("-------------Tokenization----------------")
nlp = spacy.load("en_core_web_sm")
about_text = (
    "This is Kaveri Vitthal Mungase"
    " From Final year I.T. department"
    " Sanjivani College of engineering, Kopargaon"
    " My hometown is Nashik which is famous for grapes and also known as a Grape City"
    " Always think positive, ie Nothing is impossible "
    " All you can belive in Truth!!!!"
) 
about_doc = nlp(about_text)

for token in about_doc:
    print (token, token.idx)
    



print("-------------------Stemming--------------------------")

porter_stemmer = PorterStemmer()

# Tokenizing text for stemming using nltk
tokens = nltk.word_tokenize(about_text)

for token in tokens:
    stemmed = porter_stemmer.stem(token)
    if token != stemmed:
        print(f"{token:>20} : {stemmed}")

    


print("-------------Lemmatization----------------")
about_doc = nlp(about_doc)
for token in about_doc:
    if str(token) != str(token.lemma_):
        print(f"{str(token):>20} : {str(token.lemma_)}")


print("-------------Stop Words----------------")

spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS
len(spacy_stopwords)
for stop_word in spacy_stopwords:
    print(stop_word)
  
    

print("-----------------------------------------------")
for token in about_doc:
    if token.is_stop ==True:
        print(token)
        



print("--------------------------------------------------")
about_doc = nlp(about_text)

# Remove stop words
filtered_tokens = [token.text for token in about_doc if not token.is_stop]

# Print the filtered text
filtered_text = ' '.join(filtered_tokens)
print("Text after removing stop words (spaCy):")
print(filtered_text)
    


print("----------------------REMOVE PUNCTUATION---------------------")
doc = nlp(about_doc)

# Identify punctuation
punctuation_tokens = [token.text for token in doc if token.is_punct]

# Print punctuation
print("Punctuation found (spaCy):")
print(punctuation_tokens)


        
print("----------------------------------------------")

doc = nlp(about_doc)

# Remove punctuation
filtered_tokens = [token.text for token in doc if not token.is_punct]

# Join tokens back into a string
filtered_text = ' '.join(filtered_tokens)
print("Text after removing punctuation (spaCy):")
print(filtered_text)


print("---------------Part of Speech and tagging------------------")



about_doc = nlp(about_text)
for token in about_doc:
    print(
        f"""
TOKEN: {str(token)}
=====
TAG: {str(token.tag_):10} POS: {token.pos_}
EXPLANATION: {spacy.explain(token.tag_)}"""
    )