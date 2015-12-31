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
from pandas_confusion import ConfusionMatrix
from sklearn.linear_model import LogisticRegression
from sklearn import grid_search

class QuestionsData():

	def __init__(self,csv_path='data/modeling_data/Dataset_edited.csv',question_column_name = 'Question',output_column_name='Org_tag'):
		self.quest_df = pd.DataFrame.from_csv(csv_path).get([question_column_name,output_column_name])
		print '|=                                 | Loaded questions into memory'
		self.config = self._load_config('train-config.py')
		self.model = self.config['model']
		self.n_gram_range = self.config['n_gram_range']
		self.model_param = self.config[self.model+"_param"]
		
		
	def _load_config(self, filename):
		return eval(open(filename).read())
	
	def label_encode(self):
		Y_actual_label = self.quest_df.get([self.quest_df.columns[1]])
		Y_actual_label=Y_actual_label.values.reshape((1,Y_actual_label.shape[0]))[0]
		le = preprocessing.LabelEncoder()
		le.fit(Y_actual_label)
		pickle.dump(le,open(self.config['label_encoder_folder']+"lab_enc.pkl",'wb'))
		self.Y_actual_label = Y_actual_label
		self.Y_actual = le.transform(Y_actual_label)
		
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
	
	def dict_vectorize_syntatic_features(self,syntatic_features):
		dct_vect =DictVectorizer()
		dct_vect.fit(syntatic_features.T.to_dict().values())
		pickle.dump(dct_vect,open(self.config['dict_vect_folder']+"dct_vect.pkl",'wb'))
		self.syntatic_features = dct_vect.transform(syntatic_features.T.to_dict().values())
		print '|========                          | Syntatic Features Vectorized and Pickled'
		
		
		
	def extract_ngram_features(self):
		quest_df = self.quest_df
		questions = quest_df[quest_df.columns[0]]
		tfidf_vect = TfidfVectorizer(min_df=1,stop_words=custom_stop_words,ngram_range=self.n_gram_range)
		tfidf_vect.fit(questions)
		pickle.dump(tfidf_vect,open(self.config['count_vect_folder']+"tfidf_vect.pkl",'wb'))
		self.ngram_features = tfidf_vect.transform(questions)
		print '|============                      | Ngram Features Extracted'
		print '|=============                     | Ngram Features Vectorized and Pickled'
		
	def merge_feature_set(self):
		self.X = sparse.hstack((self.ngram_features,self.syntatic_features))
		del self.syntatic_features
		del self.ngram_features
		print '|===================               | Final Feature Set Assembled'
		
		
	def grid_search(self):
		X = self.X
		Y= self.Y_actual
		clf = LogisticRegression()
		optimital_model = grid_search.GridSearchCV(clf, self.model_param['param'],n_jobs=-1,scoring='f1')
		print '|=====================             | Training Started'
		optimital_model.fit(X,Y)
		joblib.dump(optimital_model,self.model_param['model_folder']+"logreg_complete1.pkl")
		print '|=====================             | Training Complete, Model fit and pickled'
		print '*****************************************************'
		print '                   METRICS'
		print 'Best score:          ' + str(optimital_model.best_score_)
		print 'Best estimator       ' +  str(optimital_model.best_estimator_)
		
