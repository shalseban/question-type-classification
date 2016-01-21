from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score,accuracy_score
import numpy as np
import pandas as pd
from custom_lists import *
import re
from sklearn.feature_extraction import DictVectorizer
import pickle
from scipy import sparse
from sklearn.externals import joblib
import pdb
import nltk
import time

'''
Class that contains all the transformations and classification functions needed for the question
'''
class QuestionsData():
	
	'''
	Read csv that contains data and initialize the object
	'''
	def __init__(self,csv_path='test_data/test_data_formatted.csv',question_column_name = 'Question',output_column_name='Org_tag'):
		self.quest_df = pd.DataFrame.from_csv(csv_path).get([question_column_name,output_column_name])
		print '|=                                 | Loaded questions into memory'
		
	'''
	Extract Syntatic features from the question
	Syntatic Features:
	- First Word Token : Which is the first word in the sentence ?
	- First Word Type : Which Part-of-Speech does the first word belong to?
	- First Noun : What is the first Noun present in the sentence ?
	- First Question Token - What is the first question token ('Wh' word) present in the sentence ?
	- First Question Noun - What is the first noun succeeding the first question token ?
	- Is Affirmation - Is this question an affirmation question or not based on specific pattern rules ?
	'''
	def extract_syntatic_features(self):
		quest_df = self.quest_df
		syntatic_features = []
		for idx,val in enumerate(quest_df[quest_df.columns[0]]):
			first_question_token_idx = 0
			first_question_noun =first_question_token ='n/a'
			tagged_set = nltk.pos_tag(nltk.word_tokenize(val.decode('utf-8')))
			first_word_token = tagged_set[0][0].lower()
			first_word_type = tagged_set[0][1]
			first_noun = next((k[0] for k in tagged_set if k[1]=='NN'),'N/A').lower()
			first_question_token_idx,first_question_token = next(((v,k[0]) for v,k in enumerate(tagged_set) if k[0] in mid_sentence_wh_questions),(0,'N/A'))
			first_question_token = first_question_token.lower()
			if first_question_token_idx!=0:
				first_question_noun = next((k[0] for k in tagged_set[first_question_token_idx:] if k[1]=='NN'),'N/A').lower()
			is_affirmation = self.determine_affirmation(val)
			syntatic_features.append([first_word_token,first_word_type,first_noun,first_question_token,first_question_noun,is_affirmation])
		syntatic_features = pd.DataFrame(syntatic_features,columns=['first_word_token','first_word_type','first_noun','first_question_token','first_question_noun','is_affirmation'])
		print '|====                              | Syntatic Features Extracted'
		self.dict_vectorize_syntatic_features(syntatic_features)
		
	'''
	Determine if a given question is an 'affirmation' question or not
	Returns 0 if the question is not an 'affirmation' question
	Returns 1 if the question is an 'affirmation' question
	'''			
	def determine_affirmation(self,text):
		init_string = '|'.join(be_words+mod_words)
		be_string = '|'.join(be_words)
		mid_wh_string = '|'.join(mid_sentence_wh_questions)
		ind_quest = '|'.join(indirect_quest)
		if re.match('('+init_string+') .+ ?',text.lower()) is None:
			return 0
		if re.match('('+ind_quest+') [a-z]*.* ('+mid_wh_string+') [a-z]*.* ?',text.lower()) is not None or re.match('('+be_string+') [a-z]*.* or [a-z]*.* ?',text.lower()) is not None or re.match('[a-z]* (anyone|anybody).*[a-z]*.*(tell|know).*[a-z]*.* ?',text.lower()) is not None:
			return 0
		return 1
	

	'''
	Load vectorizer from pickeled model and vectorize the syntatic features
	'''
	def dict_vectorize_syntatic_features(self,syntatic_features):
		dct_vect = pickle.load(open("pickled_objects/DictVectorizer/dct_vect.pkl",'rb'))
		self.syntatic_features = dct_vect.transform(syntatic_features.T.to_dict().values())
		print '|========                          | Syntatic Features Vectorized '
		
	'''
	Extract uni-gram and bi-gram features from the question
	'''	
	def extract_ngram_features(self):
		quest_df = self.quest_df
		tfidf_vect = pickle.load(open("pickled_objects/TfIdfVectorizer/tfidf_vect.pkl",'rb'))
		self.ngram_features = tfidf_vect.transform(quest_df[quest_df.columns[0]])
		print '|============                      | Ngram Features Extracted'
		print '|=============                     | Ngram Features Vectorized'
		
		
	'''
	Merge Syntatic and N-gram feature set
	'''
	def merge_feature_set(self):
		self.X = sparse.hstack((self.ngram_features,self.syntatic_features))
		del self.syntatic_features
		del self.ngram_features
		print '|===================               | Final Feature Set Assembled'
		
	'''
	Load pickled modeled and run on feature set
	'''
	def classify_questions(self):
		X = self.X
		clf = joblib.load('pickled_objects/Models/LogistRegressionModels/logreg_complete1.pkl')
		print '|=====================             | Model loaded'
		self.Y = clf.predict(X)
		del self.X
		print '|==============================    | Questions Classified'
		
		
	'''
	Decode encoded labels to text
	'''
	def decode_label(self):
		le = pickle.load(open("pickled_objects/LabelEncoder/LE.txt",'rb'))
		self.Y_label = le.inverse_transform(self.Y)
		print '|================================  | Label decoded'
	
	
	'''
	Get accuracy and f1 score for the data set
	'''
	def get_classification_staistics(self):
		le = pickle.load(open("pickled_objects/LabelEncoder/LE.txt",'rb'))
		quest_df =self.quest_df
		Y=self.Y
		Y_actual_labels = quest_df.get([quest_df.columns[1]])
		#pdb.set_trace()
		Y_actual_labels = Y_actual_labels.values.reshape((1,Y_actual_labels.shape[0]))[0]
		Y_actual = le.transform(Y_actual_labels)
		print '*****************************************************'
		print '                   STATISTICS'
		print 'F1 score:          ' + str(f1_score(Y_actual,Y))
		print 'Precision:         ' + str(precision_score(Y_actual,Y))
		print 'Recall:            ' + str(recall_score(Y_actual,Y))
		print 'Confusion Matrix:  '
		print confusion_matrix(Y_actual_labels,self.Y_label)

		
	'''
	Save output as a csv in 'classified_questions' folder
	'''
	def save_output(self):
		Y_label = self.Y_label
		Y_label_df = pd.DataFrame(Y_label.reshape((Y_label.shape[0],1)),columns=['Predicted_Question_Type'])
		uuid = str(time.time())[:-4]
		pd.concat((self.quest_df,Y_label_df),axis=1).to_csv('classified_questions/Question_Type_'+uuid+'.csv')
		print '|==================================| Output Saved'

'''
Class for getting result for individual questions
'''
class IndividualQuestion(QuestionsData):
	'''
	Initialize individual question
	'''
	def __init__(self,quest_string):
		self.quest_df = pd.DataFrame([[quest_string]],columns=['Question'])