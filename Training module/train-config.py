{
    'n_gram_range': (1,2),
    'model' : 'logreg',
    'logreg_param': {
        'model_folder':'../pickled_objects/Models/LogistRegressionModels/',
        'param':{'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],'class_weight':['auto',None] }
    },
    'mnnb_param':{
        'model_folder':'../pickled_objects/Models/MultiNomNBayesModels/',
        'param':{'alpha' : [0.1,0.2,0.5,0.7,0.9,1.0]}
    },
    'svm_param':{
        'model_folder':'../pickled_objects/Models/SVMModels/',
        'param':{'kernel':('linear', 'rbf'), 'C':[1, 10]}
    },
    'count_vect_folder':'../pickled_objects/TfIdfVectorizer/',
    'label_encoder_folder':'../pickled_objects/LabelEncoder/',
    'dict_vect_folder':'../pickled_objects/DictVectorizer/',

}
