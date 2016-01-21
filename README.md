# Question-type-classification

Classifies each question into one of the following categories:

-What
-When
-Where
-Who
-Affirmation
-Unknown

##Training Data Set:
Trained on close to 6000 questions obtained majorly from UIUC dataset - http://cogcomp.cs.illinois.edu/Data/QA/QC/train_5500.label

##Features Used:
###1. Syntatic Features:
####- First Word Token : Which is the first word in the sentence ?
####- First Word Type : Which Part-of-Speech does the first word belong to?
####- First Noun : What is the first Noun present in the sentence ?
####- First Question Token - What is the first question token ('Wh' word) present in the sentence ?
####- First Question Noun - What is the first noun succeeding the first question token ?
####- Is Affirmation - Is this question an affirmation question or not based on specific pattern rules ?

###2. Ngram Features:
####- Unigram
####- Bigram

###Is Affirmation rule set :

####It is an affirmation question if - 

Does it start with: 
1. Be-verbs: = { am, is, are, been, being, was, were }.
2. Modal-verbs: = {can, could, shall, should, will, would, may, might }.
3. Auxiliary-verbs: = { do, did, does, have, had, has }.
eg : “Is Obama the president of the U.S. now?” 

####It is not an affirmation question if - 

Does it start with : {'can', 'could','will', 'would','may'} and contain a 'wh' question later
eg : "Can you tell me what is the time now?"

Does it start with : Be-verbs and contain an 'or' later
eg:“Is he married or not?” ,“Will the concern be on May 23rd or June 1st?”

Does it contain the following pattern : 'some text' 'anybody'|'anyone' 'some text' 'tell'|'know' 'some text'
eg:“Can anybody tell me who is the president of the U.S.?”, “Does anyone know how much Bill Gates earns a year?”


Model used - Multi-class Logistic Regression

Run the question_classifier.py file to for classification

To train a new model, the train-config file can be modified and run the 'training_man.py' file.

Major dependencies are listed down in requirements.txt

