import re
import nltk
import spacy
#import torch
import pickle
import timeit
import scipy as sc
from numpy import dot
#from scipy import spatial
from numpy.linalg import norm
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from nltk.tag import StanfordPOSTagger
from gensim.models import KeyedVectors
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity


spacy.util.fix_random_seed(0)
is_using_gpu = spacy.prefer_gpu()
if is_using_gpu:
  torch.set_default_tensor_type("torch.cuda.FloatTensor")

stanford_dir = "Data/stanford-postagger-full-2018-10-16"
modelfile = stanford_dir+"/models/english-bidirectional-distsim.tagger"
jarfile = stanford_dir+"/stanford-postagger.jar"
tagger=StanfordPOSTagger(model_filename=modelfile, path_to_jar=jarfile)

stop_words = set(stopwords.words('english'))
verbPOS = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']

filename1 = 'Data/GoogleNews-vectors-negative300.bin' # Google New Model
# filename2 = 'Data/numberbatch-en.txt' # ConceptNet Numberbatch Model
# filename3 = 'Data/crawl-300d-2M.vec' # Facebook Fasttext Model

model1 = KeyedVectors.load_word2vec_format(filename1, binary=True) # For Google News
# model2 = KeyedVectors.load_word2vec_format(filename2, encoding= 'unicode_escape') # For Conceptnet Number batch
# model3 = KeyedVectors.load_word2vec_format(filename3) # For FastText

termexpansion = {}

def getSentences(text):

	sentences = nltk.sent_tokenize(text)

	return sentences

def getExpendedWordList(word):

	if(word in termexpansion.keys()):
		return termexpansion[word]

	expendedList = []
	expendedList.append(word)
	try:
		result = model1.most_similar(word.replace(' ', ''), topn=10)
		for res in result:
			if(res[0] not in expendedList):
				expendedList.append(res[0])
		termexpansion[word] = expendedList
	except Exception as e:
		word = word

	# try:
	# 	result = model2.most_similar(word.replace(' ', ''), topn=10)
	# 	for res in result:
	# 		if(res[0] not in expendedList):
	# 			expendedList.append(res[0])
	# except Exception as e:
	# 	word = word

	# try:
	# 	result = model3.most_similar(word.replace(' ', ''), topn=10)
	# 	for res in result:
	# 		if(res[0] not in expendedList):
	# 			expendedList.append(res[0])
	# except Exception as e:
	# 	word = word
	
	return expendedList

def getWordSynonyms(wordList):

	synonymList = []

	for word in wordList:
		synonymList.append(word)
		synList = wordnet.synsets(word)

		for syn in synList:

			for l in syn.lemmas():
				if l.name() not in synonymList:
					synonymList.append(l.name())

	return synonymList

def getCausalTriple(sentences):

	tripples = []
	regex = r"<e1>(.*)<\/e1>(.*)<e2>(.*)<\/e2>"

	for sent in sentences:
		#print('sent')
		#print(sent)
		tokens = nltk.word_tokenize(sent)
		tokensPOS = tagger.tag(tokens)
		matches = re.finditer(regex, sent, re.MULTILINE)

		for matchNum, match in enumerate(matches, start=1):
			innerText =  match.group(2)
			tokenizedInnerText = nltk.word_tokenize(innerText)

			for pos in tokensPOS:
				if(pos[0] in tokenizedInnerText and pos[1] in verbPOS ):
					
					expansion = getExpendedWordList(pos[0])
					
					for trigger in expansion:
						triple = match.group(1)+' '+trigger+' '+match.group(3)
						if(triple not in tripples):
							tripples.append(triple)

	return tripples

def getCausalCandidateTriple(sentences):

	candidateTriple = []
	regex = r"<e1>(.*)<\/e1>(.*)<e2>(.*)<\/e2>"

	for sentence in sentences:

		tokens = nltk.word_tokenize(sentence)
		tokensPOS = tagger.tag(tokens)
		matches = re.finditer(regex, sentence, re.MULTILINE)

		for matchNum, match in enumerate(matches, start=1):
			innerText =  match.group(2)
			tokenizedInnerText = nltk.word_tokenize(innerText)
			entity1 = ''
			entity2 = ''
			trigger = ''

			for pos in tokensPOS:
				if(pos[0] in tokenizedInnerText and pos[1] in verbPOS):
					entity1 = match.group(1)
					entity2 = match.group(3)
					trigger = pos[0]

					if ((entity1 + ' ' + trigger + ' ' + entity2) not in candidateTriple):
						candidateTriple.append(entity1 + ' ' + trigger + ' ' + entity2)

	return candidateTriple

trainingText = open('Data/TRAIN_FILE-Cuase-effect.TXT', 'r').read()		

sentences = getSentences(trainingText)
senteceCausalTriple = getCausalTriple(sentences)

filehandler = open('sentenceCausalTripleTrainingWithStopWord.txt', 'wb')
pickle.dump(senteceCausalTriple, filehandler)

testingText = open('Data/TEST_FILE_Processed.TXT', 'r').read()		
sentences = getSentences(testingText)
candidate = getCausalCandidateTriple(sentences)

filehandler2 = open('sentenceCausalTripleTestingWithStopWord.txt', 'wb')
pickle.dump(candidate, filehandler2)

from sentence_transformers import SentenceTransformer

model = SentenceTransformer('bert-base-nli-mean-tokens')
print('model loaded')
#model = SentenceTransformer('bert-large-nli-mean-tokens')
# model = SentenceTransformer('bert-base-nli-max-tokens')
#model = SentenceTransformer('bert-large-nli-max-tokens')
# model = SentenceTransformer('bert-base-nli-cls-token')
#model = SentenceTransformer('bert-large-nli-cls-token')

trigger_embeddings = model.encode(senteceCausalTriple)
print('trigger embedded')

candidate_embeddings = model.encode(candidate)
print('candidate embedded')

identifiedCausalTerms = []


from numba import jit

@jit(nopython=True)
def cosine_similarity_numba(u:np.ndarray, v:np.ndarray):
    assert(u.shape[0] == v.shape[0])
    uv = 0
    uu = 0
    vv = 0
    for i in range(u.shape[0]):
        uv += u[i]*v[i]
        uu += u[i]*u[i]
        vv += v[i]*v[i]
    cos_theta = 1
    if uu!=0 and vv!=0:
        cos_theta = uv/np.sqrt(uu*vv)
    return cos_theta


for cand, candidate_embedding in zip(candidate, candidate_embeddings):
	for sentence, trigger_embedding in zip(senteceCausalTriple, trigger_embeddings):
		numba_distance = cosine_similarity_numba(candidate_embedding, trigger_embedding)
		similarity = abs(numba_distance)
		if (similarity) > 0.8 and cand not in identifiedCausalTerms:
			identifiedCausalTerms.append(cand)


print('identifiedCausalTerms')
print(len(identifiedCausalTerms))
print(identifiedCausalTerms)

