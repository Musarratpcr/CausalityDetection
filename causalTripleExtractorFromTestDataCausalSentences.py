import re
import nltk
import spacy
from numpy import dot
from numpy.linalg import norm
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from nltk.tag import StanfordPOSTagger
from gensim.models import KeyedVectors
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity


stanford_dir = "Data/stanford-postagger-full-2018-10-16"
modelfile = stanford_dir+"/models/english-bidirectional-distsim.tagger"
jarfile = stanford_dir+"/stanford-postagger.jar"
tagger=StanfordPOSTagger(model_filename=modelfile, path_to_jar=jarfile)

stop_words = set(stopwords.words('english'))
verbPOS = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']

def getCausalCandidateTriple(sentences):

	candidateTriple = []
	regex = r"<e1>(.*)<\/e1>(.*)<e2>(.*)<\/e2>"

	for text in sentences:
		#print('tex')

		textSplited = text.split('\n')
		sentence = textSplited[0]
		tokens = nltk.word_tokenize(sentence)
		tokensPOS = tagger.tag(tokens)
		matches = re.finditer(regex, sentence, re.MULTILINE)

		for matchNum, match in enumerate(matches, start=1):
			#print('checking regex')
			innerText =  match.group(2)

			tokenizedInnerText = nltk.word_tokenize(innerText)
			entity1 = ''
			entity2 = ''
			trigger = ''

			for pos in tokensPOS:
				if(pos[0] in tokenizedInnerText and pos[1] in verbPOS and pos[0] not in stop_words):

					entity1 = match.group(1)
					entity2 = match.group(3)
					trigger = pos[0]
					#print('append: ', entity1 + ' ' + trigger + ' ' + entity2)
					candidateTriple.append(entity1 + ' ' + trigger + ' ' + entity2)
					# candidateTriple.append({'triple' : entity1 + ' ' + trigger + ' ' + entity2, 'sentence' : sentence})
					#candidateTriple.append({'triple' : entity1 + ' ' + trigger + ' ' + entity2, 'sentence' : sentence})

	return candidateTriple


sentences = open('TEST_FILE_Cause-effect.TXT', 'r').read().split('\n\n')
senteceCausalTriple = getCausalCandidateTriple(sentences)

print('senteceCausalTriple')
print(len(senteceCausalTriple))
print(senteceCausalTriple)