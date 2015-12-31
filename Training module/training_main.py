import question
import pdb
print 'Enter column name for questions and output(, seperated):'
quest_name,out_name = raw_input().split(',')
quest = question.QuestionsData()
quest.label_encode()
quest.extract_syntatic_features()
quest.extract_ngram_features()
quest.merge_feature_set()
quest.grid_search()
pdb.set_trace()