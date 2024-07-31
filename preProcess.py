import gensim
from gensim.utils import simple_preprocess
from gensim.corpora import Dictionary

# Example Wikipedia text (replace with your scraped data)

with open('text.txt', 'r') as file:
    lines = file.readlines()
file.close()

wikipedia_text = lines

# Step 1: Tokenization
tokenized_text = [simple_preprocess(sentence) for sentence in wikipedia_text]

# Step 2: Create a Gensim Dictionary
dictionary = Dictionary(tokenized_text)

# Step 3: Convert tokenized text to bag-of-words format
corpus = [dictionary.doc2bow(tokens) for tokens in tokenized_text]

# Now you have a Gensim corpus ready for modeling!
# You can proceed to train your language model.

# Optional: Remove common stop words (if not done during tokenization)
# Example:
# from gensim.parsing.preprocessing import STOPWORDS
# tokenized_text = [[word for word in tokens if word not in STOPWORDS] for tokens in tokenized_text]
