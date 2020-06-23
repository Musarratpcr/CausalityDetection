# Causality Mining
Natural language Processing (NLP) has provided a key cohesive ingredient for pushing the boundaries of technological advances beyond individuals 
to the 4th industrial revolution. In clinical domain, the researches apply state-of-the-art NLP technieques to process patient information and 
get deeper insights from availabe online textual resources. Cause-effect is one of the essential relations which provdies ample suport
for the reasong and decision making.

# Causality Mining Approaches
Generaly there are two categories inclucing pattern base appraoch and machine learing approach for causality mining. In the pattern abase appraochs, a hand crafted rules are used to classifiy a pair of terms as causal or non-causal. However, this approach suffer genralization, and it rely too much on human experts. While the sencond approach use machine learning techniques for achieving the goal. 

The advancement of machine leanring algorithms have proved remakeable achivements in various domain including clinical domain and have producued a well established models called pre-trianed models. These models can be utalized by other researchers to perfomre their target of interest task. 

In this work we have applied an Active Transfer Learning approach to extrqact causal entities for clinical text. The steps required for this process is as follows.

# Causal Quad Generation
Initial from an annotated data set (SemEval) we extract a list of initial causal  triples in the form of <Subject, Verb, Object>. The extracted cuasual tripeles are extanded by using Google News, FastText, ConceptNet Numberbatch Models pre-trained models. The extanded list of triples are transform to quads <subject, Verb, Object, Similarity> where the foruth element represented the similarity of the expanded terms with terms extracted from training data considered as orignal causal terms. All quads having similarity less than 0.5 are dicaraded and we achieve a list of final causal quads. 

# Word Embeddings Generation
We transfomr the causal quads to word vector by using, Word2Vec, and BERT models independently. The generated verctors are used to extracted the cuaslity relation from unseen clinical text

# Causal Relation Identificiation 
The unseen clinica text is process to generate a list of candidate causal triples. The triples are transform to vectors using the same methodlogy used in training phase. While the vectors similarity is derterminde with the causal triple vectors. If the similarity is greater than a threshold of 0.85, the test triples is considred as a casual triples. 

# Activel Learning Loop
The system identified causal quads (the fourht element represent the similarity with causal triple) are verified by domain expert and is feed back to the the trianing vectors. 
