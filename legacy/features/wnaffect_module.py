import nltk
import numpy as np
import xml.etree.ElementTree as ET
from nltk.corpus import wordnet as wn
from sklearn.metrics.pairwise import cosine_similarity

def preprocess_sentence(sentence):
    stop_words = set(nltk.corpus.stopwords.words('english'))
    words = nltk.word_tokenize(sentence.lower())
    words = [word for word in words if word.isalnum() and word not in stop_words]
    return words

def load_asynsets(corpus_path):
    tree = ET.parse(corpus_path)
    root = tree.getroot()

    asynsets = {}
    for pos in ["noun", "adj", "verb", "adv"]:
        asynsets[pos] = {}
        for elem in root.findall(f".//{pos}-syn-list//{pos}-syn"):
            (p, offset) = elem.get("id").split("#")
            if not offset:
                continue

            asynsets[pos][offset] = {"offset16": offset, "pos": pos}
            if elem.get("categ"):
                asynsets[pos][offset]["categ"] = elem.get("categ")

    return asynsets

def find_similar_word(word, model, asynsets):
    if word in model:
        return word, 1.0

    synsets = wn.synsets(word)
    if not synsets:
        return None, 0

    max_similarity = 0
    best_word = None

    for syn in synsets:
        for lemma in syn.lemmas():
            lemma_name = lemma.name()
            if lemma_name in model:
                for pos, synset_data in asynsets.items():
                    for offset, info in synset_data.items():
                        affect_word = info.get("categ")
                        if affect_word and affect_word in model:
                            similarity = cosine_similarity([model[lemma_name]], [model[affect_word]])[0][0]
                            if similarity > max_similarity:
                                max_similarity = similarity
                                best_word = affect_word

    return best_word, max_similarity

def create_emotion_embeddings(asynsets, model):
    emotion_embeddings = {}
    for pos, synsets in asynsets.items():
        for offset, info in synsets.items():
            word = info.get("categ")
            if word and word in model:
                emotion_embeddings[word] = model[word]
    return emotion_embeddings

def text_to_embedding(text, model, asynsets):
    words = preprocess_sentence(text)
    embeddings = []
    for word in words:
        if word in model:
            word_embedding = model[word]
        else:
            similar_word, similarity = find_similar_word(word, model, asynsets)
            if similar_word:
                word_embedding = model[similar_word] * similarity
            else:
                word_embedding = np.zeros(model.vector_size)

        embeddings.append(word_embedding)

    if not embeddings:
        return np.zeros(model.vector_size)
    return np.mean(embeddings, axis=0)

class WordNetAffectEmbedder:
    def __init__(self, wn_model, corpus_path):
        self.wn_model = wn_model
        self.asynsets = load_asynsets(corpus_path)
        self.emotion_embeddings = create_emotion_embeddings(self.asynsets, self.wn_model)

    def get_embedding(self, text):
        return text_to_embedding(text, self.wn_model, self.asynsets)