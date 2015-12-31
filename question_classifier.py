import question

print 'Choose one of the following options:'
print '1.Test model on labeled data'
print '2.Test model on unabled data'
print '3.Test model on individual questions'
choice = int(raw_input())
if choice !=3:
	print 'Enter path of Question csv:'
	csv_path = raw_input()
	if choice ==1:
		print 'Enter column name for questions and output(, seperated):'
		quest_name,out_name = raw_input().split(',')
		quest = question.QuestionsData()
	else:
		print 'Enter column name for questions:'
		quest_name = raw_input()
		quest = question.QuestionsData(csv_path,quest_name)
	quest.extract_syntatic_features()
	quest.extract_ngram_features()
	quest.merge_feature_set()
	quest.classify_questions()
	quest.decode_label()
	quest.save_output()
	if choice ==1 :
		quest.get_classification_staistics()
else:
	while 1:
		print 'Enter the question:'
		quest_string = raw_input()
		quest_str = question.IndividualQuestion(quest_string)
		quest_str.extract_syntatic_features()
		quest_str.extract_ngram_features()
		quest_str.merge_feature_set()
		quest_str.classify_questions()
		quest_str.decode_label()
		print 'Question Type : ' + quest_str.Y_label[0]
		