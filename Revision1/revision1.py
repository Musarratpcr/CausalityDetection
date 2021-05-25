import re
import time
import nltk
import torch
import copy
import pickle
import itertools
import numpy as np
from os import path
import pandas as pd
from numba import jit, cuda
from matplotlib import pyplot
from numpy import sqrt, argmax
import torch.nn.functional as F
from scipy.spatial import distance
from gensim.models import KeyedVectors
from nltk.tag import StanfordPOSTagger
from nltk.stem.porter import PorterStemmer
from nltk.corpus import wordnet, stopwords
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
from upsetplot import UpSet, generate_counts, plot, from_memberships
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, roc_auc_score, precision_recall_curve
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

stopWords = set(stopwords.words('english'))
verbPOS = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
bertModels = ['bert-base-nli-mean-tokens', 'bert-large-nli-mean-tokens', 'bert-base-nli-max-tokens', 'bert-large-nli-max-tokens', 'bert-base-nli-cls-token', 'bert-large-nli-cls-token']
# bertModels = ['bert-base-nli-mean-tokens', 'bert-large-nli-mean-tokens', 'bert-base-nli-max-tokens', 'bert-large-nli-max-tokens', 'bert-base-nli-cls-token', 'bert-large-nli-cls-token', 'bioBert']
#with ROC 
#thresholds = [0.90, 0.65, 0.80, 0.92, 0.93, 0.94]
# with PRC
thresholds = [0.87, 0.88, 0.91, 0.92, 0.85, 0.87, 0.96]
termexpansion = {}

# Function to print the settings
def print_settings():
    print('Using device:', device)
    # Additional Info when using cuda
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_cached(0) / 1024 ** 3, 1), 'GB')

def readPickle(url):

    data = []
    with open(url, "rb") as fp:
        data = pickle.load(fp)

    return data

def writePickle(url, data):

    filehandler = open(url, 'wb')
    pickle.dump(data, filehandler) 
    filehandler.close() 

def getCausalTriple(sentences):

    triples = []

    stanford_dir = "Data/stanford-postagger-full-2018-10-16"
    modelfile = stanford_dir+"/models/english-bidirectional-distsim.tagger"
    jarfile = stanford_dir+"/stanford-postagger.jar"
    tagger=StanfordPOSTagger(model_filename=modelfile, path_to_jar=jarfile)



    causalVerbFilePath = 'Data/initialTriplesForSemEvalTrainingData.pb'

    if(path.exists(causalVerbFilePath)):

        triples = readPickle(causalVerbFilePath)
        
    else:

        regex = r"<e1>(.*)<\/e1>(.*)<e2>(.*)<\/e2>"

        for index, sent in enumerate(sentences):

            tokens = nltk.word_tokenize(sent)
            #tokensPOS = nltk.pos_tag(tokens)
            tokensPOS = tagger.tag(tokens)
            matches = re.finditer(regex, sent, re.MULTILINE)

            for matchNum, match in enumerate(matches, start=1):
                innerText = match.group(2)
                tokenizedInnerText = nltk.word_tokenize(innerText)
                
                for pos in tokensPOS:
                    if (pos[0] in tokenizedInnerText and pos[1] in verbPOS):

                        triple = [match.group(1),  pos[0],  match.group(3)]
                        if triple not in triples:
                            triples.append(triple)

        filehandler = open('Data/initialTriplesForSemEvalTrainingData.pb', 'wb')
        pickle.dump(triples, filehandler)
        filehandler.close()

    return triples

def getTermExpansion(term, model):

    if (term in termexpansion.keys()):
        return termexpansion[term]

    expendedList = []
    expendedList.append(term)

    try:
        result = model.most_similar(term.replace(' ', ''), topn=10)
        #result = model.most_similar(word.replace(' ', ''))
        for res in result:
            if (res[0] not in expendedList):
                expendedList.append(res[0])
    except Exception as e:
        #print(' term: ', term)
        term = term

    # Synonyms Identification

    # synList = wordnet.synsets(term)
    # for syn in synList:
    #     for l in syn.lemmas():
    #         if l.name() not in expendedList:
    #             expendedList.append(l.name())

    # termexpansion[term] = expendedList

    return expendedList

def getCausalTriplesExpansion(triples):

    allExpandedTriples = []

    # Model based Expansion

    filename1 = 'Data/GoogleNews-vectors-negative300.bin' # Google New Model
    # filename2 = 'Data/numberbatch-en.txt' # ConceptNet Numberbatch Model
    # filename3 = 'Data/crawl-300d-2M.vec' # Facebook Fasttext Model

    model1 = KeyedVectors.load_word2vec_format(filename1, binary=True) # For Google News
    # model2 = KeyedVectors.load_word2vec_format(filename2, encoding= 'unicode_escape') # For Conceptnet Number batch
    # model3 = KeyedVectors.load_word2vec_format(filename3) # For FastText

    # models = [model1, model2, model3]
    models = [model1]

    for index, model in enumerate(models):

        print('================== model: ', index, " ==================")

        modelExpansion = []

        expandedTriplesFilePath = 'Data/top-10-model-'+str(index)+'-without-synonym-expandedTriplesForSemEvalTrainingData.pb'

        #if(path.exists(expandedTriplesFilePath)):
        if(1 == 3):
            
            modelExpansion = readPickle(expandedTriplesFilePath)
        else:
            
            for tripleIndex, triple in enumerate(triples):

                noun1Expansion = getTermExpansion(triple[0], model)
                verbExpansion = getTermExpansion(triple[1], model)
                noun2Expansion = getTermExpansion(triple[2], model)

                expansion = set(list(itertools.product(noun1Expansion, verbExpansion, noun2Expansion)))

                modelExpansion.extend(expansion)

            modelExpansion = list(set(modelExpansion))

            print("model  " + str(index) + " Exansion: ", len(modelExpansion))

            writePickle(expandedTriplesFilePath, modelExpansion)

            #print(modelExpansion)

        
    
    return modelExpansion

def generateEmbeddings():


    # modelNames = ['bert-base-nli-mean-tokens', 'bert-large-nli-mean-tokens', 'bert-base-nli-max-tokens', 'bert-large-nli-max-tokens', 'bert-base-nli-cls-token', 'bert-large-nli-cls-token']
    modelNames = ['bert-large-nli-mean-tokens', 'bert-base-nli-max-tokens', 'bert-large-nli-max-tokens', 'bert-base-nli-cls-token', 'bert-large-nli-cls-token']

    fileURLs = ['top-10-model-0-without-synonym-expandedTriplesForSemEvalTrainingData.pb', 'top-10-model-1-without-synonym-expandedTriplesForSemEvalTrainingData.pb', 'top-10-model-2-without-synonym-expandedTriplesForSemEvalTrainingData.pb']

    for  url in fileURLs: 

        triples = readPickle("Data/"+str(url))  
    
        triplePhrases = [" ".join(t) for t in triples]

        
        for modelName in modelNames:

            print('modelName: ', modelName)

            model = SentenceTransformer(modelName, device="cpu")
            #model = SentenceTransformer(['test is moving', 'good are free'])

            print('model loaded')

            triggerEmbeddings = model.encode(triplePhrases)
            

            filehandler = open('Embeddings/'+ str(modelName) +  str(url) +'V0.1.pb', 'wb')
            pickle.dump(triggerEmbeddings, filehandler) 
            filehandler.close()    

def getSemEvalCausalCandiate(sentences):

    candidatefilePath = 'Data/SemEvalCandidateTriples.pb'

    if(path.exists(candidatefilePath)):

        candidateTriples = readPickle(candidatefilePath)
    else:   

        stanford_dir = "Data/stanford-postagger-full-2018-10-16"
        modelfile = stanford_dir+"/models/english-bidirectional-distsim.tagger"
        jarfile = stanford_dir+"/stanford-postagger.jar"
        tagger=StanfordPOSTagger(model_filename=modelfile, path_to_jar=jarfile)

        stop_words = set(stopwords.words('english'))
        verbPOS = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']

        candidateTriples = []
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
                    causalObj = {'triple': triple, 'causal' : causal}
                    if(causalObj not in candidateTriples):
                        candidateTriples.append(causalObj)

        writePickle(candidatefilePath, candidateTriples)   

    return candidateTriples

def generateTestTriplesEmbeddings(triples, dataset):

    #bertModels = [ 'bert-large-nli-mean-tokens', 'bert-base-nli-max-tokens', 'bert-large-nli-max-tokens', 'bert-base-nli-cls-token', 'bert-large-nli-cls-token']

    for modelName in bertModels:

        
        modelStartsTime = time.time()
        
        print('modelName: ', modelName)
        model = SentenceTransformer(modelName)
        embeddings = model.encode(triples, device='cpu')

        filehandler = open('Embeddings/'+ str(dataset) + "-" + str(modelName) + 'V0.2.pb', 'wb')
        pickle.dump(embeddings, filehandler) 
        filehandler.close()

def preprocessADSentece(sentence):
    
    sentence = re.sub(r"\((.*?)\)", "", sentence)
    
    return sentence

def getADCandidateTriple(sentences):

    candidatefilePath = 'Data/ADDataset/ADDatasetCandidateTriplesV0.1.pb'
    
    if(path.exists(candidatefilePath)):
        print('file ' + candidatefilePath + ' already exist')

        candidateTriples = readPickle(candidatefilePath)
    else:


        stanford_dir = "Data/stanford-postagger-full-2018-10-16"
        modelfile = stanford_dir+"/models/english-bidirectional-distsim.tagger"
        jarfile = stanford_dir+"/stanford-postagger.jar"
        tagger=StanfordPOSTagger(model_filename=modelfile, path_to_jar=jarfile)

        stop_words = set(stopwords.words('english'))
        verbPOS = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']


        candidateTriples = []
        
        for text in sentences:
            text = text.split("\n")
            sentence = text[0]
            sentence = preprocessADSentece(sentence)
            entities  = text[1]

            leftTerm = ''
            direction = ''
            rightTerm = ''

            regex = r"(.*)(->|-|\?)(.*)"
            matches = re.finditer(regex, entities, re.MULTILINE)

            for matchNum, match in enumerate(matches, start=1):
                leftTerm = match.group(1).strip()
                direction = match.group(2).strip()
                rightTerm = match.group(3).strip()

                if(sentence.find(leftTerm) >= 0 and sentence.find(rightTerm) >= 0 ):
                
                    regex2 = r".*("+leftTerm+"|"+rightTerm+")(.*)("+leftTerm+"|"+rightTerm+")"
                    matches2 = re.finditer(regex2, sentence, re.MULTILINE)

                    for matchNum2, match2 in enumerate(matches2, start=1):
                        tokenizedSentence =  nltk.word_tokenize(sentence)
                        sentencePOS = nltk.pos_tag(tokenizedSentence)
                        innerText = match2.group(2)
                        tokenizedInnerText = nltk.word_tokenize(innerText)

                        for token in sentencePOS:
                            if (token[0] in tokenizedInnerText and token[1] in verbPOS and token[0] not in stop_words):
                                triple = leftTerm+' '+token[0]+' '+rightTerm
                                causal = False
                                if(direction == '->' or direction == '-'):
                                    causal = True
                                tripleObj = {'triple': triple, 'causal' : causal}

                                if (tripleObj not in candidateTriples):
                                    candidateTriples.append(tripleObj)

                                # if(triple not in candidateTriples):
                                #   candidateTriples.append(triple)
        
        writePickle(candidatefilePath, candidateTriples)
    
    return candidateTriples

def mapTermWithSentence(term, sentence):
    identifiedTerm = ''
    stemmer = PorterStemmer()

    termStem = stemmer.stem(term)

    if(sentence.find(term) >= 0):
        identifiedTerm = term
    else:
        tokens = nltk.word_tokenize(sentence)
        
        for token in tokens:
            tokenStem = stemmer.stem(token)

            if(termStem == tokenStem):

                identifiedTerm = token
                break

    return identifiedTerm
    

def getAsianNetCandidateTriple(sentences):

    # candidatefilePath = 'Data/AsianNet/AsianNetCandidateTriplesWithSentencesV0.1.pb'

    # if(path.exists(candidatefilePath)):
    if( 1 == 3):
        print('file ' + candidatefilePath + ' already exist')

        candidateTriples = readPickle(candidatefilePath)
    else:

        stanford_dir = "Data/stanford-postagger-full-2018-10-16"
        modelfile = stanford_dir+"/models/english-bidirectional-distsim.tagger"
        jarfile = stanford_dir+"/stanford-postagger.jar"
        tagger=StanfordPOSTagger(model_filename=modelfile, path_to_jar=jarfile)

        stop_words = set(stopwords.words('english'))
        verbPOS = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
        
        candidateTriples = []

        regex = r"(.*)(\n?)(.*)(->|-|\.)(.*)(\n?)(consistency?)"

        matches = re.finditer(regex, sentences, re.MULTILINE)

        count = 0

        for matchNum, match in enumerate(matches, start=1):

            sentence = match.group(1)
            leftTerm = match.group(3).strip()
            direction = match.group(4).strip()
            rightTerm = match.group(5).strip()

            # candidateTriples.append([sentence, True if direction == '->' or direction == '-' else False])
            candidateTriples.append({'triple': sentence, 'causal' : True if direction == '->' or direction == '-' else False})
            continue
            

            sentence = sentence.lower()
            leftTerm = leftTerm.lower()
            rightTerm = rightTerm.lower()

            sentenceLeftTerm = mapTermWithSentence(leftTerm, sentence)
            sentenceRightTerm = mapTermWithSentence(rightTerm, sentence)

            if(sentenceLeftTerm != '' and sentenceRightTerm != ''):
                count += 1
                print('count: ', count)
                continue
            else:
                print('sentence: ', sentence)
                print('leftTerm: ', leftTerm)
                print('rightTerm: ', rightTerm)
                continue

            exit()



            if(sentence.find(leftTerm) >= 0 and sentence.find(rightTerm) >= 0 ):

                count += 1
                print('count: ', count)
                continue
            else:
                print('sentence: ', sentence)
                print('leftTerm: ', leftTerm)
                print('rightTerm: ', rightTerm)
                continue
                
                regex2 = r".*("+leftTerm+"|"+rightTerm+")(.*)("+leftTerm+"|"+rightTerm+")"
                matches2 = re.finditer(regex2, sentence, re.MULTILINE)

                for matchNum2, match2 in enumerate(matches2, start=1):
                    tokenizedSentence =  nltk.word_tokenize(sentence)
                    sentencePOS = nltk.pos_tag(tokenizedSentence)
                    innerText = match2.group(2)
                    tokenizedInnerText = nltk.word_tokenize(innerText)

                    for token in sentencePOS:
                        if (token[0] in tokenizedInnerText and token[1] in verbPOS and token[0] not in stop_words):
                            triple = leftTerm+' '+token[0]+' '+rightTerm
                            causal = False
                            if(direction == '->' or direction == '-'):
                                causal = True
                            tripleObj = {'triple': triple, 'causal' : causal, 'sentence' : sentence}
                            
                            candidateTriples.append(tripleObj)
        
        # writePickle(candidatefilePath, candidateTriples)         
    
    return candidateTriples

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

def PredictLables(candidates, candidateEmbeddings, triggers, triggerEmbeddings, threshold):

    predictedClass = []

    index = 0

    for cand, candidateEmbedding in zip(candidates, candidateEmbeddings):

        causal = 0
        #print('index: ', index)

        for trigger, triggerEmbedding in zip(triggers, triggerEmbeddings):

            # candTorch = torch.tensor(candidateEmbedding)
            # trigTorch = torch.tensor(triggerEmbedding)
            
            # similarity1 = F.cosine_similarity(candTorch, trigTorch, dim=0).to(device)
            # print('cand: ', cand, ' trigger: ', trigger, ' similarity: ', similarity1)
            similarity = cosine_similarity_numba(candidateEmbedding, triggerEmbedding)

            if(similarity > threshold):
                causal = 1
                break
        
        predictedClass.append(causal)
        index += 1

    return predictedClass

def PredictSimilarity(candidates, candidateEmbeddings, triggers, triggerEmbeddings):

    predictedSimlarities = []

    index = 0

    for cand, candidateEmbedding in zip(candidates, candidateEmbeddings):

        similarity = 0
        # print('index: ', index)

        for trigger, triggerEmbedding in zip(triggers, triggerEmbeddings):

            sim = cosine_similarity_numba(candidateEmbedding, triggerEmbedding)
            
            if(sim > similarity):
                similarity = sim
        
        predictedSimlarities.append(similarity)
        index += 1

    return predictedSimlarities

def PredictBioBertSimilarity(candidateEmbeddings, triggerEmbeddings):

    predictedSimlarities = []

    index = 0


    candidateEmbeddingsCPU = [t.cpu().numpy() for t in candidateEmbeddings]
    triggerEmbeddingsCPU = [t.cpu().numpy() for t in triggerEmbeddings]


    for candidateEmbedding in candidateEmbeddingsCPU:

        similarity = 0
        #print('index: ', index)

        for triggerEmbedding in triggerEmbeddingsCPU:

            #sim = cosine_similarity_numba(candidateEmbedding.cpu().numpy(), triggerEmbedding.cpu().numpy())
            sim = cosine_similarity_numba(candidateEmbedding, triggerEmbedding)
           
            if(sim > similarity):
                similarity = sim
        
        predictedSimlarities.append(similarity)
        index += 1

    return predictedSimlarities

def calculateSemEvalTestSimilarity():
    
    #expansionModels = ['model-0', 'model-1', 'model-2']
    expansionModels = ['model-0', 'model-2']
    #bertModels = ['bert-base-nli-mean-tokens', 'bert-large-nli-mean-tokens', 'bert-base-nli-max-tokens', 'bert-large-nli-max-tokens', 'bert-base-nli-cls-token', 'bert-large-nli-cls-token']


    semEvalTesting = readPickle('Data/SemEvalCandidateTriples.pb')
    acutalClasses = [t['causal'] for t in semEvalTesting]


    for expModel in expansionModels:
        print('expModel: ', expModel)

        triggers = readPickle('Data/top-10-'+ expModel+'-without-synonym-expandedTriplesForSemEvalTrainingData.pb')

        for model in bertModels:

            print('model: ', model)
            
            triggerEmbeddings = readPickle('Embeddings/'+model+'top-10-'+expModel+'-without-synonym-expandedTriplesForSemEvalTrainingData.pbV0.1.pb')
            semEvalTestingEmbeddings = readPickle('Embeddings/SemEvalTesting-'+model+'V0.1.pb')
            
            #thresholds = [0.2, 0.5, 0.6, 0.7, 0.8, 0.85, 0.86, 0.9]
            thresholds = [0.87, 0.88]

            for threshold in thresholds:

                print('threshold: ', threshold)

                filePath = 'PredictedClasses/'+model+'-'+expModel+'-SemEvalTesting-'+str(threshold)+'-V0.1.pb'

                if(path.exists(filePath)):

                    print('file ' + filePath + ' Already Exist')
                    continue
                else:
                    predictedClass = PredictLables(semEvalTesting, semEvalTestingEmbeddings, triggers, triggerEmbeddings, threshold, device)                    
                    writePickle('PredictedClasses/'+model+'-'+expModel+'-SemEvalTesting-'+str(threshold)+'-V0.1.pb', predictedClass)

def calculateSemEvalTestBioBertSimilarity():
    

    semEvalTesting = readPickle('Data/SemEvalCandidateTriples.pb')
    semEvalTestingEmbeddings = readPickle('Embeddings/SemEvalTesting-BioBertV0.1.pb')
    acutalClasses = [t['causal'] for t in semEvalTesting]

    print('semEvalTesting: ', len(semEvalTesting), ' semEvalTestingEmbeddings: ', len(semEvalTestingEmbeddings))

    triggers = readPickle('Data/top-10-model-0-without-synonym-expandedTriplesForSemEvalTrainingData.pb')

    triggerEmbeddingsName = ['model-0-BioBertTriggersEmbeddings-V0.1.pb', 'model-0-BioBertTriggersEmbeddings-V0.2.pb', 'model-0-BioBertTriggersEmbeddings-V0.3.pb', 'model-0-BioBertTriggersEmbeddings-V0.4.pb', 'model-0-BioBertTriggersEmbeddings-V0.5.pb', 'model-0-BioBertTriggersEmbeddings-V0.6.pb', 'model-0-BioBertTriggersEmbeddings-V0.7.pb', 'model-0-BioBertTriggersEmbeddings-V0.8.pb', 'model-0-BioBertTriggersEmbeddings-V0.9.pb', 'model-0-BioBertTriggersEmbeddings-V0.10.pb', 'model-0-BioBertTriggersEmbeddings-V0.11.pb', 'model-0-BioBertTriggersEmbeddings-V0.12.pb', 'model-0-BioBertTriggersEmbeddings-V0.13.pb']

    for triggerName in triggerEmbeddingsName:
        print('triggerName: ', triggerName)
        exit()




    triggerEmbeddings = readPickle('Embeddings/model-0-BioBertAllTriggerEmbeddings-V0.1.pb')

    print('triggers: ', len(triggers), ' triggerEmbeddings: ', len(triggerEmbeddings))

    filePath = 'PredictedSimilarities/SemEvalTestingBioBertV0.1.pb'

    if(path.exists(filePath)):

        print('file ' + filePath + ' Already Exist')
    else:
        predictedSimilarities = PredictSimilarity(semEvalTesting, semEvalTestingEmbeddings, triggers, triggerEmbeddings, device)                    
        writePickle(filePath, predictedClass)

def evaluatePerformance():

    expansionModels = ['model-0', 'model-1', 'model-2']
    #bertModels = ['bert-base-nli-mean-tokens', 'bert-large-nli-mean-tokens', 'bert-base-nli-max-tokens', 'bert-large-nli-max-tokens', 'bert-base-nli-cls-token', 'bert-large-nli-cls-token']
    thresholds = [0.2, 0.5, 0.6, 0.7, 0.8, 0.85, 0.86, 0.9]
    

    semEvalTesting = readPickle('Data/SemEvalCandidateTriples.pb')
    acutalClasses = [t['causal'] for t in semEvalTesting] 
    acutalClasses = [1 if t == True else 0 for t in acutalClasses] # converting True false to 1,0



    for expModel in expansionModels:
        #print('expModel: ', expModel)

        for model in bertModels:
            #print('mode: ', model)

            for threshold in thresholds:
                #print('threshold: ', threshold)

                predictedClass = readPickle('PredictedClasses/'+model+'-'+expModel+'-SemEvalTesting-'+str(threshold)+'-V0.1.pb')

                accuracy = accuracy_score(acutalClasses, predictedClass) 
                precision = precision_score(acutalClasses, predictedClass) 
                recall = recall_score(acutalClasses, predictedClass) 
                f1 = f1_score(acutalClasses, predictedClass)

                print('expModel: ', expModel, ' model: ', model, ' threshold: ', threshold, ' accuracy :', accuracy, ' precision: ', precision, ' recall: ', recall, ' f1: ', f1)   

def showROCGraphs():

    semEvalTesting = readPickle('Data/SemEvalCandidateTriples.pb')
    acutalClasses = [t['causal'] for t in semEvalTesting] 
    actualClasses = [1 if t == True else 0 for t in acutalClasses] # converting True false to 1,0


    #expansionModels = ['model-0', 'model-1', 'model-2']
    expansionModels = ['model-0']
    bertModels = ['bert-base-nli-mean-tokens', 'bert-large-nli-mean-tokens', 'bert-base-nli-max-tokens', 'bert-large-nli-max-tokens', 'bert-base-nli-cls-token', 'bert-large-nli-cls-token', 'BioBert']
    # bertModels = ['BioBert']
    colors = ["#A2142F", "#EDB120", "#ED553B", "#20639B", "#3CAEA3", "#7E2F8E", "#09943e"]    
    
    for expModel in expansionModels:
        
        for index, model in enumerate(bertModels):
            
            predictedSimilarities = readPickle('PredictedSimilarities/SemEvalTesting'+model+'V0.1.pb')

            predictedSimilarities = [float(round(t, 2)) for t in predictedSimilarities]
            

            fpr, tpr, thresholds = roc_curve(actualClasses, predictedSimilarities)
            roc_auc = auc(fpr, tpr)
            gmeans = sqrt(tpr * (1-fpr))
            ix = argmax(gmeans)



            pyplot.scatter(fpr[ix], tpr[ix], marker='o', edgecolor='black', color=colors[index], s=100, linewidth=3)
            pyplot.text(fpr[ix]+.03, tpr[ix]+.03, 'Threshold ' + str(predictedSimilarities[ix]), fontdict=dict(color='black',size=10), bbox=dict(facecolor='yellow',alpha=0.5))

            pyplot.plot(fpr, tpr, marker='.', label='Model ' + str(model) +' (area = %0.2f)' % roc_auc, color=colors[index], linewidth=2)
            print('fpr[ix]', fpr[ix], ' tpr[ix]', tpr[ix], ' ix: ', ix, ' similarity: ', predictedSimilarities[ix])
            

            
            pyplot.plot([0, 1], [0, 1], color='navy',  linestyle='--')
            pyplot.xlabel('False Positive Rate')
            pyplot.ylabel('True Positive Rate')
            #pyplot.title('ROC for model ' + model)
            pyplot.legend()
            pyplot.show()

def showROCGraphForTestDataset():
    semEvalTesting = readPickle('Data/SemEvalCandidateTriples.pb')
    acutalClasses = [t['causal'] for t in semEvalTesting] 
    actualClasses = [1 if t == True else 0 for t in acutalClasses] # converting True false to 1,0


    #expansionModels = ['model-0', 'model-1', 'model-2']
    
    colors = ["#A2142F", "#EDB120", "#ED553B", "#20639B", "#3CAEA3", "#7E2F8E"]    
    
    
        
    for index, model in enumerate(bertModels):
        
        predictedSimilarities = readPickle('PredictedSimilarities/SemEvalTesting'+model+'V0.1.pb')

        predictedSimilarities = [float(round(t, 2)) for t in predictedSimilarities]
        

        fpr, tpr, thresholds = roc_curve(actualClasses, predictedSimilarities)
        roc_auc = auc(fpr, tpr)
        gmeans = sqrt(tpr * (1-fpr))
        ix = argmax(gmeans)



        pyplot.scatter(fpr[ix], tpr[ix], marker='o', edgecolor='black', color=colors[index], s=100, linewidth=3)
        pyplot.text(fpr[ix]+.03, tpr[ix]+.03, 'Threshold ' + str(predictedSimilarities[ix]), fontdict=dict(color='black',size=10), bbox=dict(facecolor='yellow',alpha=0.5))

        pyplot.plot(fpr, tpr, marker='.', label='Model ' + str(model) +' (area = %0.2f)' % roc_auc, color=colors[index], linewidth=2)
        print('fpr[ix]', fpr[ix], ' tpr[ix]', tpr[ix], ' ix: ', ix, ' similarity: ', predictedSimilarities[ix])
        

        
        pyplot.plot([0, 1], [0, 1], color='navy',  linestyle='--')
        pyplot.xlabel('False Positive Rate')
        pyplot.ylabel('True Positive Rate')
        #pyplot.title('ROC for model ' + model)
        pyplot.legend()
        pyplot.show()

def showPRCGraphs():

    semEvalTesting = readPickle('Data/SemEvalCandidateTriples.pb')
    acutalClasses = [t['causal'] for t in semEvalTesting] 
    actualClasses = [1 if t == True else 0 for t in acutalClasses] # converting True false to 1,0


    #expansionModels = ['model-0', 'model-1', 'model-2']
    expansionModels = ['model-0']
    bertModels = ['bert-base-nli-mean-tokens', 'bert-large-nli-mean-tokens', 'bert-base-nli-max-tokens', 'bert-large-nli-max-tokens', 'bert-base-nli-cls-token', 'bert-large-nli-cls-token', 'BioBert']
    # bertModels = ['bert-base-nli-mean-tokens']
    colors = ["#A2142F", "#EDB120", "#ED553B", "#20639B", "#3CAEA3", "#7E2F8E", "#09943e"]    
    
    for expModel in expansionModels:
        
        for index, model in enumerate(bertModels):
            
            predictedSimilarities = readPickle('PredictedSimilarities/SemEvalTesting'+model+'V0.1.pb')
            predictedSimilarities = [float(round(t, 2)) for t in predictedSimilarities]
            
            precision, recall, thresholds = precision_recall_curve(actualClasses, predictedSimilarities)

            areas = []
            for thrIndex, threshold in enumerate(thresholds):
                area = auc([0, recall[thrIndex], 1], [1, precision[thrIndex], 0])

                areas.append(area)

            ix = argmax(areas)

            print(' Threshold', thresholds[ix], ' area: ', areas[ix])
            
            pyplot.scatter(recall[ix], precision[ix], marker='o', edgecolor='black', color=colors[index], s=100, linewidth=3, label='Best threshold')
            pyplot.plot([0, recall[ix], 1], [1, precision[ix], 0 ], ls="--", color='black', linewidth=1)
            #pyplot.text(recall[ix]+.03, precision[ix]+.03, 'Threshold: ' + str(thresholds[ix]) + '\n AUC: ' + str(round(areas[ix], 2)), fontdict=dict(color='black',size=10), bbox=dict(facecolor='yellow',alpha=0.5))
            pyplot.text(recall[ix]-.26, precision[ix]-.12, 'Threshold: ' + str(thresholds[ix]) + '\n AUC: ' + str(round(areas[ix], 2)), fontdict=dict(color='black',size=10), bbox=dict(facecolor='yellow',alpha=0.5))

            pyplot.plot(recall, precision, marker='o', label='Model: ' + str(model), color=colors[index], linewidth=2)
            
            pyplot.xlabel('Recall')
            pyplot.ylabel('Precision')
            pyplot.legend()
            pyplot.show()

def generateTestDatasetQuads(candidateTriplesPath, dataset):

    triggers = readPickle('Data/top-10-model-0-without-synonym-expandedTriplesForSemEvalTrainingData.pb')
    print('triggers: ', len(triggers))

    candidateObj = readPickle(candidateTriplesPath)
    print('candidateTriples: ', len(candidateObj))

    candidateTriples = [t['triple'] for t in candidateObj]

    #bertModels = ['bert-base-nli-mean-tokens']
    
    for index, modelName in enumerate(bertModels):

        print('model: ', modelName)

        quadsFile = 'Quads/'+dataset+'-'+modelName+'-V0.2.pb'

        if(not path.exists(quadsFile)):

            quads = []

            filePath = 'PredictedSimilarities/'+dataset+'-' + modelName + '-V0.2.pb'
            if(path.exists(filePath)):
                similarities = readPickle(filePath)
            else:

                similarities = []
                
                triggerEmbeddings = readPickle('Embeddings/'+modelName+'-top-10-model-0-without-synonym-expandedTriplesForSemEvalTrainingDataV0.1.pb')
                print('triggerEmbeddings: ', len(triggerEmbeddings))

                candidateEMbeddingsPath = 'Embeddings/'+dataset+'-'+modelName+'V0.2.pb'

                if(path.exists(candidateEMbeddingsPath)):
                    candidateEmbeddings = readPickle(candidateEMbeddingsPath)
                else:
                    model = SentenceTransformer(modelName, device="cpu")
                    candidateEmbeddings = model.encode(candidateTriples)

                print("candidateEmbeddings: ", len(candidateEmbeddings))

                similarities = PredictSimilarity(candidateTriples, candidateEmbeddings, triggers, triggerEmbeddings)                    
                writePickle(filePath, similarities)

            for tripleIndex, triple in enumerate(candidateObj):
                #print('tripleIndex: ', tripleIndex, ' triple: ', triple)

                quad = triple

                quad['triple'] = triple['triple'] + ' ' + str(similarities[tripleIndex])

                quads.append(quad)
                
            writePickle('Quads/'+dataset+'-'+modelName+'-V0.2.pb', quads)
        else:
            print('quads file ' + quadsFile + ' already exists')

def generateFeedbackTestDatasetQuads(candidateTriplesPath, dataset, iteration):

    print('dataset: ', dataset, ' iteration: ', iteration)

   
    for index, model in enumerate(bertModels):

        print('model: ', model)

        triggers = readPickle('Data/Iteration'+ str(iteration) +'/' + model + '-SemEvalTrainingData-V0.1.pb')
        print('triggers: ', len(triggers))


        candidateObj = readPickle(candidateTriplesPath)
        print('candidateTriples: ', len(candidateObj))

        candidateTriples = [t['triple'] for t in candidateObj]


        quadsFile = 'Quads/Iteration'+ str(iteration) +'/' + dataset+'-'+model+'-V0.1.pb'

        if(not path.exists(quadsFile)):

            quads = []

            filePath = 'PredictedSimilarities/Iteration'+str(iteration) + '/' + dataset+'-' + model + '-V0.1.pb'
            if(path.exists(filePath)):
            
                similarities = readPickle(filePath)
            else:

                similarities = []
                
                triggerEmbeddings = readPickle('Embeddings/Iteration' + str(iteration) + '/'+ model+'-SemEvalTrainingTripelEmbeddings-V0.1.pb')
                print('triggerEmbeddings: ', len(triggerEmbeddings))

                candiateEmbeddignsPath = 'Embeddings/'+dataset+'-'+model+'V0.1.pb'
                if(path.exists(candiateEmbeddignsPath)):
                    candidateEmbeddings = readPickle('Embeddings/'+dataset+'-'+model+'V0.1.pb')
                else:
                    bertModel = SentenceTransformer(model)
                    candidateEmbeddings = bertModel.encode(candidateTriples, device='cpu')
                    writePickle(candiateEmbeddignsPath, candidateEmbeddings)



                print("candidateEmbeddings: ", len(candidateEmbeddings))

                similarities = PredictSimilarity(candidateTriples, candidateEmbeddings, triggers, triggerEmbeddings)                    
                writePickle(filePath, similarities)

            for tripleIndex, triple in enumerate(candidateObj):
                #print('tripleIndex: ', tripleIndex, ' triple: ', triple)

                quad = triple

                quad['triple'] = triple['triple'] + ' ' + str(similarities[tripleIndex])

                quads.append(quad)
                
            writePickle(quadsFile, quads)
        else:
            print('quads file ' + quadsFile + ' already exists')

def predictBioBertTestDatasetClasses(candidateTriplesPath, dataset):

    finalSimilarities = []
    bioBertTreshold = 0.96
    classes = []

    classesPath = 'PredictedClasses/bioBert-' + dataset +'-model-0-'+str(bioBertTreshold)+'-V0.1.pb'

    if(not path.exists(classesPath)):
        

        candidateEmbeddings = readPickle('Embeddings/'+dataset+'-BioBertV0.1.pb')
        print('candidateEmbeddings: ', len(candidateEmbeddings))

        for n in range(1, 14):
            print('N: ', n)

            triggerEmbeddings = readPickle('Embeddings/model-0-BioBertTriggersEmbeddings-V0.'+str(n)+'.pb')
            
            filePath = 'PredictedSimilarities/'+ dataset +'BioBertV0.1.pb'

            if(path.exists(filePath)):

                print('file ' + filePath + ' Already Exist')
                finalSimilarities = readPickle(filePath)
            
            predictedSimilarities = PredictBioBertSimilarity(candidateEmbeddings, triggerEmbeddings) 

            if len(finalSimilarities) > 0:

                for index, similarity in enumerate(finalSimilarities):
                    if similarity < predictedSimilarities[index]:
                        finalSimilarities[index] = predictedSimilarities[index]
            else:
                finalSimilarities = predictedSimilarities                

            print('finalSimilarities: ', len(finalSimilarities))
            
            writePickle(filePath, finalSimilarities)
            writePickle('Data/bioBert'+ dataset +'Number.pb', n)

        classes =  [1 if t > bioBertTreshold else 0 for t in finalSimilarities]
        writePickle(classesPath, classes)
    else:
        print('file already exist')

def predictSemEvalTestDatasetSimilarity():

    triggers = readPickle('Data/top-10-model-0-without-synonym-expandedTriplesForSemEvalTrainingData.pb')
    print('triggers: ', len(triggers))

    candidateObj = readPickle('Data/SemEvalCandidateTriples.pb')
    print('candidateTriples: ', len(candidateObj))

    candidateTriples = [t['triple'] for t in candidateObj]

    
    for model in bertModels:

        print('model: ', model)

        triggerEmbeddings = readPickle('Embeddings/'+model+'-top-10-model-0-without-synonym-expandedTriplesForSemEvalTrainingDataV0.1.pb')
        print('triggerEmbeddings: ', len(triggerEmbeddings))

        candidateEmbeddings = readPickle('Embeddings/SemEvalTesting-'+model+'V0.1.pb')
        print("candidateEmbeddings: ", len(candidateEmbeddings))



        filePath = 'PredictedSimilarities/SemEvalTesting'+model+'V0.1.pb'

        if(path.exists(filePath)):

            print('file ' + filePath + ' Already Exist')
            continue
        else:
            predictedSimilarities = PredictSimilarity(candidateTriples, candidateEmbeddings, triggers, triggerEmbeddings, device)                    
            writePickle(filePath, predictedSimilarities)

def predictSemEvalTestDatasetBioBertSimilarity():

    finalSimilarities = []
    
    semEvalTesting = readPickle('Data/SemEvalCandidateTriples.pb')
    semEvalTestingEmbeddings = readPickle('Embeddings/SemEvalTesting-BioBertV0.1.pb')
    
    embeddingsFiles = range(1, 14)

    for n in embeddingsFiles:
        print('N: ', n)

        triggerEmbeddings = readPickle('Embeddings/model-0-BioBertTriggersEmbeddings-V0.'+str(n)+'.pb')
        
        filePath = 'PredictedSimilarities/SemEvalTestingBioBertV0.1.pb'

        if(path.exists(filePath)):

            print('file ' + filePath + ' Already Exist')
            finalSimilarities = readPickle(filePath)
        
        predictedSimilarities = PredictBioBertSimilarity(semEvalTestingEmbeddings, triggerEmbeddings, device) 

        if len(finalSimilarities) > 0:

            for index, similarity in enumerate(finalSimilarities):
                if similarity < predictedSimilarities[index]:
                    finalSimilarities[index] = predictedSimilarities[index]
        else:
            finalSimilarities = predictedSimilarities                

        print('finalSimilarities: ', len(finalSimilarities))
        
        writePickle(filePath, finalSimilarities)
        writePickle('Data/bioBertNumber.pb', n)

def evaluateTestDatasetPerformance(candidateTriplesPath, dataset, iteration = 0):

    candidateObj = readPickle(candidateTriplesPath)
    print('candidateTriples: ', len(candidateObj))

    acutalClasses = [t['causal'] for t in candidateObj] 
    acutalClasses = [1 if t == True else 0 for t in acutalClasses] # converting True false to 1,0

    for index, model in enumerate(bertModels):

        # print('model: ', model)

        predictedClass = []

        if iteration == 0:

            quads = readPickle('Quads/'+ dataset + '-' + model + '-V0.1.pb')
        else:
            quads = readPickle('Quads/Iteration'+ str(iteration) + '/' + dataset + '-' + model + '-V0.1.pb')
        
        
        for quad in quads:
            similarity = quad['triple'].split()[-1]
            
            predictedClass.append(1 if float(similarity) >= thresholds[index] else 0)
        
        
        accuracy = accuracy_score(acutalClasses, predictedClass) 
        precision = precision_score(acutalClasses, predictedClass) 
        recall = recall_score(acutalClasses, predictedClass) 
        f1 = f1_score(acutalClasses, predictedClass)

        TP, FP, TN, FN = perf_measure(acutalClasses, predictedClass)
                
        print('TP: ', TP, ' FP: ', FP, ' TN: ', TN, ' FN: ', FN)
        
        print('dataset: ', dataset, 'iteration: ', iteration, ' model: ', model, ' accuracy :', accuracy, ' precision: ', precision, ' recall: ', recall, ' f1: ', f1)   

def evaluateBioBertTestDatasetPerformance(candidateTriplesPath, dataset):

    candidateObj = readPickle(candidateTriplesPath)
    print('candidateTriples: ', len(candidateObj))

    acutalClasses = [t['causal'] for t in candidateObj] 
    acutalClasses = [1 if t == True else 0 for t in acutalClasses] # converting True false to 1,0

    predictedClass = readPickle('PredictedClasses/bioBert-model-0-'+dataset+'-0.96-V0.1.pb')


        
        
    accuracy = accuracy_score(acutalClasses, predictedClass) 
    precision = precision_score(acutalClasses, predictedClass) 
    recall = recall_score(acutalClasses, predictedClass) 
    f1 = f1_score(acutalClasses, predictedClass)

    TP, FP, TN, FN = perf_measure(acutalClasses, predictedClass)
            
    print('TP: ', TP, ' FP: ', FP, ' TN: ', TN, ' FN: ', FN)
    
    print('dataset: ', dataset,  ' accuracy :', accuracy, ' precision: ', precision, ' recall: ', recall, ' f1: ', f1)   

def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1

    return(TP, FP, TN, FN)    

def evaluateMultimodelPerformance(candidateTriplesPath, dataset, degree, iteration = 0):

    multimodelClasses = []
    candidateObj = readPickle(candidateTriplesPath)
    acutalClasses = [t['causal'] for t in candidateObj] 
    acutalClasses = [1 if t == True else 0 for t in acutalClasses] # converting True false to 1,0
    
    allPredictedClasses = []
    
    for index, model in enumerate(bertModels):

        predictedClasses = []

        if iteration == 0:
            quads = readPickle('Quads/'+ dataset + '-' + model + '-V0.2.pb')
        else:
            quads = readPickle('Quads/Iteration'+  str(iteration) + '/' + dataset + '-' + model + '-V0.1.pb')
        
        for quad in quads:
            similarity = quad['triple'].split()[-1]
            predictedClasses.append(1 if float(similarity) > thresholds[index] else 0)

        allPredictedClasses.append(predictedClasses)

    print('allPredictedClasses: ', len(allPredictedClasses))

    for actualIndex, actualLabel in enumerate(acutalClasses):
        
        labelSum = 0
        
        for modelIndex, predClasses in enumerate(allPredictedClasses):
            labelSum += predClasses[actualIndex]
        
        multimodelClasses.append(1 if labelSum >= degree else 0)

    
    accuracy = accuracy_score(acutalClasses, multimodelClasses) 
    precision = precision_score(acutalClasses, multimodelClasses) 
    recall = recall_score(acutalClasses, multimodelClasses) 
    f1 = f1_score(acutalClasses, multimodelClasses)

    TP, FP, TN, FN = perf_measure(acutalClasses, multimodelClasses)
            
    print('TP: ', TP, ' FP: ', FP, ' TN: ', TN, ' FN: ', FN)
    
    print('dataset: ', dataset, 'iteration: ', iteration, ' accuracy :', accuracy, ' precision: ', precision, ' recall: ', recall, ' f1: ', f1)

def showUpsetGraph():
    adData = from_memberships(
    [   
    [],
    ['LargeMean'],
    ['BaseMax'],
    ['LargeMax'],
    ['BaseCls'],
    ['LargeCls'],
    ['BaseMean', 'BaseCls'],
    ['LargeMean', 'LargeMax'],
    ['LargeMean', 'BaseCls'],
    ['LargeMean', 'LargeCls'],
    ['BaseMax', 'BaseCls'],
    ['LargeMax', 'BaseCls'],
    ['LargeMax', 'LargeCls'],
    ['BaseCls', 'LargeCls'],
    ['BaseMean', 'LargeMean', 'BaseCls'],
    ['BaseMean', 'BaseMax', 'BaseCls'],
    ['BaseMean', 'BaseCls', 'LargeCls'],
    ['LargeMean', 'BaseMax', 'BaseCls'],
    ['LargeMean', 'BaseMax', 'LargeCls'],
    ['LargeMean', 'LargeMax', 'BaseCls'],
    ['LargeMean', 'LargeMax', 'LargeCls'],
    ['LargeMean', 'BaseCls', 'LargeCls'],
    ['BaseMax', 'BaseCls', 'LargeCls'],
    ['LargeMax', 'BaseCls', 'LargeCls'],
    ['BaseMean', 'LargeMean', 'BaseMax', 'BaseCls'],
    ['BaseMean', 'LargeMean', 'LargeMax', 'LargeCls'],
    ['BaseMean', 'LargeMean', 'BaseCls', 'LargeCls'],
    ['BaseMean', 'BaseMax', 'LargeMax', 'BaseCls'],
    ['BaseMean', 'BaseMax', 'BaseCls', 'LargeCls'],
    ['BaseMean', 'LargeMax', 'BaseCls', 'LargeCls'],
    ['LargeMean', 'BaseMax', 'BaseCls', 'LargeCls'],
    ['LargeMean', 'LargeMax', 'BaseCls', 'LargeCls'],
    ['BaseMax', 'LargeMax', 'BaseCls', 'LargeCls'],
    ['BaseMean', 'LargeMean', 'BaseMax', 'BaseCls', 'LargeCls'],
    ['BaseMean', 'LargeMean', 'LargeMax', 'BaseCls', 'LargeCls'],
    ['BaseMean', 'BaseMax', 'LargeMax', 'BaseCls', 'LargeCls'],
    ['LargeMean', 'BaseMax', 'LargeMax', 'BaseCls', 'LargeCls'],
    ['BaseMean', 'LargeMean', 'BaseMax', 'LargeMax', 'BaseCls', 'LargeCls'],
    ],
    data=[196, 10, 2, 2, 65, 74, 6, 2, 26, 45, 7, 2, 12, 56, 5, 5, 6, 1, 1, 2, 34, 53, 18, 14, 1, 2, 11, 2, 11, 1, 22, 67, 10, 18, 12, 9, 31, 109]
    )

    #print(example)
    plot(adData, show_counts='%d')
    pyplot.show()

def getPredictedCausalTriples(candidateTriplesPath, dataset, degree, iteration=0):

    predictedCausalTriples = []
    multimodelClasses = []
    candidateObj = readPickle(candidateTriplesPath)
    actualClasses = [t['causal'] for t in candidateObj] 
    actualClasses = [1 if t == True else 0 for t in actualClasses] # converting True false to 1,0

    print('actualClasses: ', len(actualClasses))
    
    allPredictedClasses = []
    
    for index, model in enumerate(bertModels):

        predictedClasses = []

        if iteration == 0:
            print('in iteration 0')
            quads = readPickle('Quads/'+ dataset + '-' + model + '-V0.1.pb')
        else:
            print('in iteration ' + str(iteration))
            quads = readPickle('Quads/iteration'+ str(iteration) + '/' + dataset + '-' + model + '-V0.1.pb')
        
        for quad in quads:
            similarity = quad['triple'].split()[-1]
            predictedClasses.append(1 if float(similarity) > thresholds[index] else 0)

        allPredictedClasses.append(predictedClasses)

    for actualIndex, actualLabel in enumerate(actualClasses):
        
        labelSum = 0
        
        for modelIndex, predClasses in enumerate(allPredictedClasses):
            labelSum += predClasses[actualIndex]
        
        multimodelClasses.append(1 if labelSum >= degree else 0)

    print('multimodelClasses: ', len(multimodelClasses))

    for classIndex, lab in enumerate(multimodelClasses):

        if lab == 1:
            predictedCausalTriples.append(candidateObj[classIndex])

    return predictedCausalTriples

def cleanTrainedEmbeddings(iteration):
    
    blockList = readPickle('BlockLists/blocklist-V0.'+ str(iteration) + '.pb')
    print('blockList: ', len(blockList))

    for modelIndex, modelName in enumerate(bertModels):

        print('modelName: ', modelName)

        blockListEmbeddingsPath = 'Embeddings/BlockList/Iteration'+ str(iteration) + '/' +modelName+'-blockList-V0.1.pb'

        if(path.exists(blockListEmbeddingsPath)):

            blockListEmbeddings = readPickle(blockListEmbeddingsPath)
        else:

            model = SentenceTransformer(modelName)
            blockListEmbeddings = model.encode(blockList)
            writePickle(blockListEmbeddingsPath, blockListEmbeddings)

        if(iteration == 1) : 
            previousTriggersPath = 'Data/top-10-model-0-without-synonym-expandedTriplesForSemEvalTrainingData.pb'
            previousTriggerEmbeddingsPath = 'Embeddings/'+modelName+'-top-10-model-0-without-synonym-expandedTriplesForSemEvalTrainingDataV0.1.pb'
        else:
            previousTriggersPath = 'Data/Iteration'+ str(iteration - 1)+'/'+modelName+'-SemEvalTrainingData-V0.1.pb'
            previousTriggerEmbeddingsPath = 'Embeddings/Iteration'+ str(iteration - 1)+'/'+modelName+'-SemEvalTrainingTripelEmbeddings-V0.1.pb'


        newTriggersPath =  'Data/Iteration'+ str(iteration)+'/'+modelName+'-SemEvalTrainingData-V0.1.pb'
        newTriggerEmbeddingsPath = 'Embeddings/Iteration'+ str(iteration)+'/'+modelName+'-SemEvalTrainingTripelEmbeddings-V0.1.pb'

        
        triggers = readPickle(previousTriggersPath)
        triggerEmbeddings = readPickle(previousTriggerEmbeddingsPath)

        print('triggers: ', len(triggers), ' triggerEmbeddings: ', len(triggerEmbeddings))

        similarities = PredictSimilarity(triggers, triggerEmbeddings, blockList, blockListEmbeddings)

        newTriggers = triggers
        newTriggerEmbeddings = triggerEmbeddings

        deleteIndex = []

        for simIndex, sim in enumerate(similarities):
            
            if(sim > thresholds[modelIndex]):
                deleteIndex.append(simIndex)

        print('deleteIndex: ', len(deleteIndex))

        newTriggers = [t for tIndex, t, in enumerate(triggers)  if tIndex not in deleteIndex ]
        newTriggerEmbeddings = [t for tIndex, t, in enumerate(triggerEmbeddings)  if tIndex not in deleteIndex ]

        print('newTriggers: ', len(newTriggers), ' newTriggerEmbeddings: ', len(newTriggerEmbeddings))

        newAddedTriples = readPickle('PositiveCasual/PositiveCasual-V0.'+ str(iteration) + '.pb')
        newAddedEmbeddingsPath = 'Embeddings/PositiveCasual/Iteration'+ str(iteration) + '/' +modelName+'-PositiveCasual-V0.1.pb'

        if(path.exists(newAddedEmbeddingsPath)):
            newAddedTriggerEmbeddings = readPickle(newAddedEmbeddingsPath)
        else:
            model = SentenceTransformer(modelName)
            newAddedTriggerEmbeddings = model.encode(newAddedTriples)
            writePickle(newAddedEmbeddingsPath, newAddedTriggerEmbeddings)

        newTriggers = newTriggers + newAddedTriples

        newTriggerEmbeddings.extend(newAddedTriggerEmbeddings)

        print('newTriggers: ', len(newTriggers), ' newTriggerEmbeddings: ', len(newTriggerEmbeddings))

        writePickle(newTriggersPath, newTriggers)
        writePickle(newTriggerEmbeddingsPath, newTriggerEmbeddings)

def evaluateAsianNetMultimodelPerformance(candidateTriplesPath, dataset, degree, iteration = 0):
    
    multimodelClasses = []

    

    candidateObj = readPickle(candidateTriplesPath)
    print('candidateTriples: ', len(candidateObj))

    candidateTriples = [t['triple'] for t in candidateObj]

    acutalClasses = [t['causal'] for t in candidateObj] 
    acutalClasses = [1 if t == True else 0 for t in acutalClasses] # converting True false to 1,0

    #bertModels = ['bert-base-nli-mean-tokens']
    model = {}

    for modelName in bertModels:
        model[modelName] = {}
        model[modelName]['model'] = SentenceTransformer(modelName, device="cpu")
        model[modelName]["triggers"]  =   readPickle('Data/iteration3/' + modelName + '-SemEvalTrainingData-V0.1.pb')         
        model[modelName]["triggerEmbeddings"]  =   readPickle('Embeddings/iteration3/'+modelName+'-SemEvalTrainingTripelEmbeddings-V0.1.pb')         


    for tripleIndex, triple in enumerate(candidateTriples):

        print('tripleIndex: ', tripleIndex)

        causal = 0

        for index, modelName in enumerate(bertModels):
            
            candidateEmbeddings = model[modelName]['model'].encode(triple)

            similarity = PredictSimilarity([triple], [candidateEmbeddings], model[modelName]["triggers"], model[modelName]["triggerEmbeddings"])
            
            if similarity[0] >= thresholds[index]:
                
                causal = 1
                break

        print("causal:",causal)
        multimodelClasses.append(causal)


    
    accuracy = accuracy_score(acutalClasses, multimodelClasses) 
    precision = precision_score(acutalClasses, multimodelClasses) 
    recall = recall_score(acutalClasses, multimodelClasses) 
    f1 = f1_score(acutalClasses, multimodelClasses)

    TP, FP, TN, FN = perf_measure(acutalClasses, multimodelClasses)
            
    print('TP: ', TP, ' FP: ', FP, ' TN: ', TN, ' FN: ', FN)
    
    print('dataset: ', dataset, 'iteration: ', iteration, ' accuracy :', accuracy, ' precision: ', precision, ' recall: ', recall, ' f1: ', f1)
                
            

#print_settings()

# ================= For extraction initial triples =============

# trainingText = open('Data/SemEval2010TrainingCauseEffect.txt', 'r').read()
# sentences = trainingText.splitlines() # 1003 causal sentences
# print('sentences: ', len(sentences))

# causalTriples = getCausalTriple(sentences)
# print('causalTriples: ', len(causalTriples))


# =================== Expanding Causal Triples ==============

# expandedCausaltriples = getCausalTriplesExpansion(causalTriples)

# print('expandedCausaltriples: ', len(expandedCausaltriples))



# =================== Embedding Expanding Causal Triples ==============

#generateEmbeddings()

# =======================================================
#                                                       =
#               SemEval Test Dataset                    =
#                                                       =
#========================================================


# testsentences = open('Data/SemEval2010Testing.txt', 'r').read().split('\n\n')
# candidateTriples = getSemEvalCausalCandiate(testsentences)
# print('candidateTriples: ', len(candidateTriples))

# actualClasses = [1 if t['causal'] == True else 0 for t in candidateTriples]

# print('actualClasses: ', len(actualClasses))
# totalCausal = np.sum(actualClasses)
# print('totalCausal: ', totalCausal)


#==== Generating SemEval Test tripels embeddgins ========

#triplesObj = readPickle('Data/SemEvalCandidateTriples.pb')
#triples = [t['triple'] for t in triplesObj]
#generateTestTriplesEmbeddings(triples, 'SemEvalTesting')

# =======================================================
#                                                       =
#               Threshold Selection                     =
#                                                       =
#========================================================

#calculateSemEvalTestSimilarity(device)

#calculateSemEvalTestBioBertSimilarity(device)

#evaluatePerformance()
#predictSemEvalTestDatasetSimilarity(device)
# predictSemEvalTestDatasetBioBertSimilarity(device)
# print(readPickle('Data/bioBertNumber.pb'))
# showROCGraphs()
# showPRCGraphs()


# =======================================================
#                                                       =
#               AD Test triples                         =
#                                                       =
#========================================================

# print('==================== AD Test Data =============')

# sentences = open('Data/ADDataset/ADDataset.txt', 'r').read().split("\n\n")
# triplesObj = getADCandidateTriple(sentences)
# print('triplesObj: ', len(triplesObj))
# actualClasses = [1 if t['causal'] == True else 0 for t in triplesObj]

# print('actualClasses: ', len(actualClasses))
# totalCausal = np.sum(actualClasses)
# print('totalCausal: ', totalCausal)


#==== Generating AD tripels embeddgins ========

# triplesObj = readPickle('Data/ADDataset/ADDatasetCandidateTriplesV0.1.pb')
# triples = [t['triple'] for t in triplesObj]
# print('triples: ', len(triples))



# generateTestTriplesEmbeddings(triples, 'AD')
# generateTestDatasetQuads('Data/ADDataset/ADDatasetCandidateTriplesV0.1.pb', 'AD', 0)

# predictBioBertTestDatasetClasses('Data/ADDataset/ADDatasetCandidateTriplesV0.1.pb', 'AD')
# evaluateBioBertTestDatasetPerformance('Data/ADDataset/ADDatasetCandidateTriplesV0.1.pb', 'AD')
# evaluateBioBertTestDatasetPerformance('Data/AsianNet/AsianNetCandidateTriplesV0.1.pb', 'AsianNet')

# evaluateTestDatasetPerformance('Data/ADDataset/ADDatasetCandidateTriplesV0.1.pb', 'AD')
#multimodelClasses = predictMultimodelClassess('Data/ADDataset/ADDatasetCandidateTriplesV0.1.pb', 'AD', 3)
# evaluateMultimodelPerformance('Data/ADDataset/ADDatasetCandidateTriplesV0.1.pb', 'AD', 1)

# predictedCausalTriples = getPredictedCausalTriples('Data/ADDataset/ADDatasetCandidateTriplesV0.1.pb', 'AD', 1)
# print('predictedCausalTriples: ', len(predictedCausalTriples))


# print(predictedCausalTriples)



# =======================================================
#                                                       =
#               AsianNet Test Tripels                   =
#                                                       =
#========================================================

# print('==================== AsiaNet Test Dataset =============')

# text = open('Data/AsianNet/AsiaDatasetFormated.txt', 'r').read()
# tripleObj = getAsianNetCandidateTriple(text)
# print('tripleObj', len(tripleObj))


# actualClasses = [1 if t['causal'] == True else 0 for t in tripleObj]
# print('acutalClasses ', len(actualClasses))


# print('actualClasses: ', len(actualClasses))
# totalCausal = np.sum(actualClasses)
# print('totalCausal: ', totalCausal)

#==== Generating AsianNet tripels embeddgins ========

# triplesObj = readPickle('Data/AsianNet/AsianNetCandidateTriplesV0.1.pb')
# triples = [t['triple'] for t in tripleObj]
# actualClasses = [1 if t['causal'] == True else 0 for t in triplesObj]
# print('triples: ', len(triples))

# totalCausal = np.sum(actualClasses)
# print('totalCausal: ', totalCausal)



# generateTestTriplesEmbeddings(triples, 'AsianNet')

# generateTestDatasetQuads('Data/AsianNet/AsianNetCandidateTriplesV0.2.pb', 'AsianNet', 0)

# evaluateTestDatasetPerformance('Data/AsianNet/AsianNetCandidateTriplesV0.1.pb', 'AsianNet')

evaluateAsianNetMultimodelPerformance('Data/AsianNet/AsianNetCandidateTriplesV0.2.pb', 'AsianNet', 1)
# causalClassifiedTriples = getPredictedCausalTriples('Data/AsianNet/AsianNetCandidateTriplesV0.1.pb', 'AsianNet', 3)
# print('causalClassifiedTriples: ', len(causalClassifiedTriples))

# predictedCausalTriples = getPredictedCausalTriples('Data/AsianNet/AsianNetCandidateTriplesV0.1.pb', 'AsianNet', 1)
# print('predictedCausalTriples: ', len(predictedCausalTriples))

# print(predictedCausalTriples)



# ====================== Upset Graph =====================

# showUpsetGraph()



# =================== Combined Analysis and Result ================

# combinedTriplObj = readPickle('Data/AsianNet/AsianNetCandidateTriplesV0.1.pb') + readPickle('Data/ADDataset/ADDatasetCandidateTriplesV0.1.pb')
# print('combinedTriplObj: ', len(combinedTriplObj))

# combinedEmbeddings


# =======================================================
#                                                       =
#                  FeedbackLoop loop                    =
#                                                       =
#========================================================


# =================== Spliting AD Dataset =============

# adTriplesObj = readPickle('Data/ADDataset/ADDatasetCandidateTriplesV0.1.pb')

# triples = [t['triple'] for t in adTriplesObj]
# acutalClasses = [t['causal'] for t in adTriplesObj] 

# X_train, X_test, y_train, y_test = train_test_split(triples, acutalClasses, test_size=0.50, random_state=42)
# print('X_train: ', len(X_train), ' X_test: ', len(X_test), ' y_train: ', len(y_train), ' y_test: ', len(y_test))

# split1 = [{'triple': t, 'causal': y_train[index]} for index, t in enumerate(X_train)]
# split2 = [{'triple': t, 'causal': y_test[index]} for index, t in enumerate(X_test)]

# writePickle('Data/ADDataset/ADSplit1CandidateTriplesV0.1.pb', split1)
# writePickle('Data/ADDataset/ADSplit2CandidateTriplesV0.1.pb', split2)


# generateTestDatasetQuads('Data/ADDataset/ADSplit1CandidateTriplesV0.1.pb', 'AD1')
# evaluateTestDatasetPerformance('Data/ADDataset/ADSplit1CandidateTriplesV0.1.pb', 'AD1')
# evaluateMultimodelPerformance('Data/ADDataset/ADSplit1CandidateTriplesV0.1.pb', 'AD1', 1)



# generateTestDatasetQuads('Data/ADDataset/ADSplit2CandidateTriplesV0.1.pb', 'AD2')
# evaluateTestDatasetPerformance('Data/ADDataset/ADSplit2CandidateTriplesV0.1.pb', 'AD2')
# evaluateMultimodelPerformance('Data/ADDataset/ADSplit2CandidateTriplesV0.1.pb', 'AD2', 1)


# =================== Iteration 1 =============

# predictedCausalTriples = getPredictedCausalTriples('Data/ADDataset/ADSplit1CandidateTriplesV0.1.pb', 'AD1', 1)
# print('predictedCausalTriples: ', len(predictedCausalTriples))
# #print(predictedCausalTriples) # 342

# #blockList = 28
# blockList = ['smoke monitored hypertension', 'cancer associated heart attack', 'cancer mellitus hypertension', 'smoke cigarette hypertension', 'stroke signaling systolic blood pressure', 'smoke drink hypertension', 'cancer observed heart rate', 'cancer die smoke', 'stroke attenuates systolic blood pressure', 'cancer detected hypertension', 'cancer include hypertension', 'diabetes eating depression', 'stroke calculate heart rate', 'stroke evaluate heart rate', 'smoke living hypertension', 'alcohol consumed systolic blood pressure', 'alcohol consumed heart rate', 'stroke rv heart rate', 'stroke diabetes hypertension', 'stroke diabetes systolic blood pressure', 'diabetes and/or heart attack', 'heart rate resting heart attack', 'cancer lagged alcohol', 'smoke diabetes stroke', 'cancer ] alcohol', 'diabetes = depression', 'cancer diabetes alcohol', 'cancer = alcohol']
# print('blockList: ', len(blockList))
# positiveCasualTriples = [t['triple'] for t in predictedCausalTriples if t['triple'] not in blockList]
# print('result: ', len(positiveCasualTriples))
# writePickle('BlockLists/blocklist-V0.1.pb', blockList)
# writePickle('PositiveCasual/PositiveCasual-V0.1.pb', positiveCasualTriples)

# cleanTrainedEmbeddings(1)
# generateFeedbackTestDatasetQuads('Data/ADDataset/ADSplit1CandidateTriplesV0.1.pb', 'AD1', 1)
# evaluateTestDatasetPerformance('Data/ADDataset/ADSplit1CandidateTriplesV0.1.pb', 'AD1', 1)
# evaluateTestDatasetPerformance('Data/ADDataset/ADSplit2CandidateTriplesV0.1.pb', 'AD2', 1)
# evaluateMultimodelPerformance('Data/ADDataset/ADSplit1CandidateTriplesV0.1.pb', 'AD1', 1, 1)


# generateFeedbackTestDatasetQuads('Data/ADDataset/ADSplit2CandidateTriplesV0.1.pb', 'AD2', 1)
# evaluateTestDatasetPerformance('Data/ADDataset/ADSplit2CandidateTriplesV0.1.pb', 'AD2', 1)
# evaluateMultimodelPerformance('Data/ADDataset/ADSplit2CandidateTriplesV0.1.pb', 'AD2', 1, 1)


# =================== Iteration 2 =============

# predictedCausalTriples = getPredictedCausalTriples('Data/ADDataset/ADSplit2CandidateTriplesV0.1.pb', 'AD2', 1, 1)
# print('predictedCausalTriples: ', len(predictedCausalTriples))
# print(predictedCausalTriples) # 417


# # #blockList = 49
# blockList = ['alcohol smoking diabetes', 'heart attack measured systolic blood pressure', 'stroke fasting heart rate', 'stroke known heart rate', 'smoke diabetes hypertension', 'diabetes focusing heart attack', 'diabetes adjusting depression', 'weight including heart disease', 'heart rate ] systolic blood pressure', 'stroke attenuated systolic blood pressure', 'alcohol diabetes systolic blood pressure', 'stroke translate hypertension', 'weight classifying heart disease', 'weight drinking heart disease', 'cancer working alcohol', 'hypertension maintained heart rate', 'stroke smoking systolic blood pressure', 'cancer suggesting smoke', 'cancer limited alcohol', 'cancer suggests alcohol', 'alcohol diabetes heart disease', 'hypertension diabetes heart rate', 'hypertension Is heart rate', 'weight according heart disease', 'cancer diabetes hypertension', 'alcohol [ depression', 'stroke outcome heart rate', 'stroke including systolic blood pressure', 'alcohol according heart disease', 'stroke based systolic blood pressure', 'stroke known hypertension', 'weight breed heart disease', 'weight provide heart disease', 'diabetes including heart attack', 'diabetes ascvd heart attack', 'cancer compared stroke', 'hypertension ratio heart rate', 'cancer treated hypertension', 'alcohol left systolic blood pressure', 'weight and/or heart disease', 'cancer aged smoke', 'cancer diabetes smoke', 'stroke < systolic blood pressure', 'stroke = diabetes', 'alcohol diabetes stroke', 'cancer known smoke', 'diabetes odds depression', 'alcohol mellitus diabetes', 'cancer diabetes stroke']
# print('blockList: ', len(blockList))


# positiveCasualTriples = [t['triple'] for t in predictedCausalTriples if t['triple'] not in blockList]
# print('result: ', len(positiveCasualTriples))
# writePickle('BlockLists/blocklist-V0.2.pb', blockList)
# writePickle('PositiveCasual/PositiveCasual-V0.2.pb', positiveCasualTriples)

# cleanTrainedEmbeddings(2)
# generateFeedbackTestDatasetQuads('Data/AsianNet/AsianNetCandidateTriplesV0.1.pb', 'AsianNet', 2)
# evaluateTestDatasetPerformance('Data/AsianNet/AsianNetCandidateTriplesV0.1.pb', 'AsianNet', 2)
# evaluateMultimodelPerformance('Data/AsianNet/AsianNetCandidateTriplesV0.1.pb', 'AsianNet', 1, 2)

# =================== Iteration 3 =============

# predictedCausalTriples = getPredictedCausalTriples('Data/AsianNet/AsianNetCandidateTriplesV0.1.pb', 'AsianNet', 1, 2)
# print('predictedCausalTriples: ', len(predictedCausalTriples))
# print(predictedCausalTriples) 

# # # #blockList = 49
# blockList = ['bronchitis smoke smoking', 'bronchitis analysed tuberculosis', 'bronchitis compared tuberculosis', 'bronchitis followed tuberculosis', 'dyspnoea predicted smoking', 'lung cancer smoke smoking', 'lung cancer stratified smoking', 'lung cancer assuming smoking', 'lung cancer testing smoking', 'lung cancer secondhand smoking', 'lung cancer misdiagnosed tuberculosis', 'smoking smoked tuberculosis']
# print('blockList: ', len(blockList))


# positiveCasualTriples = [t['triple'] for t in predictedCausalTriples if t['triple'] not in blockList]
# print('result: ', len(positiveCasualTriples))
# writePickle('BlockLists/blocklist-V0.3.pb', blockList)
# writePickle('PositiveCasual/PositiveCasual-V0.3.pb', positiveCasualTriples)

# cleanTrainedEmbeddings(3)


