import nltk
from nltk.corpus import stopwords
from nltk.tag import StanfordPOSTagger



stanford_dir = "Data/stanford-postagger-full-2018-10-16"
modelfile = stanford_dir+"/models/english-bidirectional-distsim.tagger"
jarfile = stanford_dir+"/stanford-postagger.jar"
tagger=StanfordPOSTagger(model_filename=modelfile, path_to_jar=jarfile)

stop_words = set(stopwords.words('english'))
verbPOS = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']

def getCausalCandidateTriple(sentences):

	candidateTriple = []
	index = 1

	for text in sentences:
		
		print('index: ', index)
		index = index + 1
		
		textSplited = text.split('\n')
		sentence = textSplited[0]
		tokens = nltk.word_tokenize(sentence)
		tokensPOS = tagger.tag(tokens)
		
		pos1a = text.lower().find('<e1>') + len('<e1>')
		pos1b = text.lower().find('</e1>')
		entity1 = text[pos1a:pos1b]

		pos2a = text.lower().find('<e2>') + len('<e2>')
		pos2b = text.lower().find('</e2>')
		entity2 = text[pos2a:pos2b]

		pos1 = text.lower().find(entity1) + len(entity1) + 5
		pos2 = text.lower().find(entity2) - 4

		innerText = text[pos1:pos2]
		tokenizedInnerText = nltk.word_tokenize(innerText)

		for pos in tokensPOS:
			if(pos[0] in tokenizedInnerText and pos[1] in verbPOS and pos[0] not in stop_words):

				trigger = pos[0]
				triple = entity1 + ' ' + trigger + ' ' + entity2
				causal = False
				if(textSplited[1].find('Cause-Effect') >= 0):
					causal = True
				candidateTriple.append({'triple': triple, 'causal' : causal})

	return candidateTriple


sentences = open('TEST_FILE_Result.TXT', 'r').read().split('\n\n')
senteceCausalTriple = getCausalCandidateTriple(sentences)

print('senteceCausalTriple')
print(len(senteceCausalTriple))
print(senteceCausalTriple)