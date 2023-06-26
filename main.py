from hazm import *
from scipy import spatial
from sent2vec.vectorizer import Vectorizer
import string
from hazm import *
# from nltk.treeprettyprinter import * (deprecated now)
from nltk.tree import TreePrettyPrinter
from nltk.draw import TreeView
from nltk import Tree
from collections import Counter
import numpy as np
import math
from googletrans import Translator
# from datasets import load_dataset

# dataset = load_dataset("pn_summary")

with open("stopwords.txt", "r") as f:
  stop_words = f.read()
  
with open("sample.txt", "r") as sample:
  sample_text = sample.read()

# sample_text = dataset['validation'][0]['article']

normalizer = Normalizer()
tagger = POSTagger(model='resources/pos_tagger.model')
chunker = Chunker(model='resources/chunker.model')
lemmatizer = Lemmatizer()
parser = DependencyParser(tagger=tagger, lemmatizer=lemmatizer)

def sent_similarity(sents):
  vectorizer = Vectorizer(pretrained_weights='distilbert-base-multilingual-cased')
  sentences = []
  for sent in sents:
    c = ' '.join([word for word in sent.split() if word not in stop_words])
    sentences.append(c)
  vectorizer.run(sentences)
  vectors_w2v = vectorizer.vectors
  p = vectors_w2v[0]

  for i in range(1, len(sentences)):
    q = vectors_w2v[i]
    cos_alpha = 1 - spatial.distance.cosine(p, q)
    alpha = np.arccos(cos_alpha)*180/math.pi
    # print(f"cos_alpha: {cos_alpha}, alpha: {alpha}")
    if alpha <= 10:
      return True
  return False

def calc_word_freq(text):
  translate_table = {ord(char): None for char in string.punctuation}
  s = text.translate(translate_table)

  words = [word for word in word_tokenize(s) if word not in stop_words]
  return Counter(words)


def generate_summary(text, word_freq, word_limit):
  sentences = sent_tokenize(text)
  sentence_scores = {}

  for sent in sentences:
    sentence_words = [word for word in word_tokenize(sent) if word not in stop_words]
    sentence_score = sum(word_freq[word] for word in sentence_words)
    if len(sentence_words) < word_limit:
      sentence_scores[sent] = sentence_score

  summary_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)
  summary = []

  for sent in summary_sentences:
    if not summary:
      summary.append(sent)
      word_limit = word_limit - len(sent)
    elif len(sent) <= word_limit:
      l = [sent, *summary]
      if not sent_similarity(l):
        summary.append(sent)
        word_limit = word_limit - len(sent)

  return ' '.join(summary)

sentences = normalizer.normalize(sample_text)
word_freq = calc_word_freq(sample_text)

summary = generate_summary(sentences, word_freq, 400)
summary_sentences = summary.split('. ')

formatted_summary = '.\n'.join(summary_sentences)
print(formatted_summary)

# paraphrasing
translator = Translator()
tr = translator.translate(formatted_summary, dest='en') # for the loop back we can use both Arabic
# and English cause Persian translations to these languages are more precise
trb = translator.translate(tr.text, dest='fa')

print(tr.text)
print(normalizer.normalize(trb.text))