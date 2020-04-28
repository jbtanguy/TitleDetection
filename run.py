import pickle
import json
import io
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.model_selection import StratifiedKFold
from vectorize import vectorize
from rstr import *

def open_json(path):
	f = open(path)
	data = json.load(f)
	f.close()
	return data

def get_score(y_ref, y_pred):
	score = [0, 0]
	for i in range(len(y_ref)):
		if(y_ref[i]==y_pred[i]):
			score[0]+=1
		else:
			score[1]+=1
	return score[0]/(score[0]+score[1])


def prepare_data(corpus_train, method):
	dic_json = open_json(corpus_train)
	if 'rstr' in method: # rstr, different because we have to calculate the desc matrix beforehand
		texts = [infos["text_line"] for x, infos in dic_json.items()]
		desc, matrix = get_matrix_rstr(method, dic_json, desc_arg=None)
		v = DictVectorizer()
		X = v.fit_transform(matrix)
		y, L_IDS = [], []
		for ID, infos in dic_json.items():
			y.append(infos["label"])
			L_IDS.append(ID)
		y.append('0')
		L_IDS.append('-1')
	else:
		X, y, L_IDS = [], [], []
		desc = {}
		for ID, infos in dic_json.items():
			if 'POS' in method:
				res_vectorize = vectorize(infos, method, desc=desc)
			else:
				res_vectorize = vectorize(infos, method, desc=desc)
			X.append(res_vectorize[0])
			desc = res_vectorize[1]
		y = [infos['label'] for ID, infos in dic_json.items()]
		L_IDS = dic_json.keys()
	return X, y, L_IDS, desc

if __name__ == "__main__":
	# 1. CONFIG: corpora, methods and classifiers
	# CONFIG ENGLISH
	config = {
	'train_corpora': ['./corpora/english/fintoc2019_en_train.json', './corpora/english/fintoc2019_en_train_test.json', './corpora/english/fintoc2020_en_train.json', './corpora/english/fintoc2019_en_train_test_fintoc2020_en_train.json'],
	'dev_corpus': './corpora/english/fintoc2020_en_dev.json',
	'test_corpus': './corpora/english/fintoc2020_en_test.json',
	'methods': ['ngramChar_'+freq+'_'+str(i)+','+str(j) for i in range(2, 11) for j in range(2, 11) for freq in ['abs', 'rel'] if i <= j],
	'classifiers': [('RF50', RandomForestClassifier(max_depth=50))],
	'models_dir': './models/english/',
	'models_annots_dir': './models_annots/english/'
	}
	""" CONFIG FRENCH
	config = {
	'train_corpora': ['./corpora/french/deft2011_train.json', './corpora/french/deft2011_train_test.json', './corpora/french/deft2011_train_test_fintoc2020_fr_train.json', './corpora/french/fintoc2020_fr_train.json'],
	'dev_corpus': './corpora/french/fintoc2020_fr_dev.json',
	'test_corpus': './corpora/french/fintoc2020_fr_test.json',
	'methods': ['ngramChar_abs_n,N', 'ngramChar_rel_n,N'],
	'classifiers': [('RF50', RandomForestClassifier(max_depth=50))],
	'models_dir': './models/french/',
	'models_annots_dir': './models_annots/french/'
	}"""

	for train_corpus in config['train_corpora']:
		dev_corpus = config['dev_corpus']
		test_corpus = config['test_corpus']
		for method in config['methods']:
			for classif in config['classifiers']:
				classif_name = classif[0]
				classif_obj = classif[1]
				print('-'*50)
				print(train_corpus.split('/')[-1].replace('.json', '') + '_' + method + '_' + classif_name)
				print('-'*50)
				# 1. Model training
				# a. data preparation
				print('Learning...')
				X_train, y_train, IDs_train, desc = prepare_data(train_corpus, method)
				if 'ngramChar' in method:
					dictvectorizer = DictVectorizer()
					X_train = dictvectorizer.fit_transform(X_train)
				# b. learning
				model = classif_obj.fit(X_train, y_train)
				# c. save the model
				model_name = config['models_dir'] + train_corpus.split('/')[-1].replace('.json', '') + '_' + method + '_' + classif_name + '.model'
				outFile = open(model_name, 'wb')
				pickle.dump([model, desc], outFile)
				print('Model saved.')
				# 2. Annotation dev and test corpus
				print('Labeling...')
				for corpus_to_test in [dev_corpus, test_corpus]:
					corpus = open_json(corpus_to_test)
					if 'rstr' in method:
						_, matrix = get_matrix_rstr(method, corpus, desc_arg=desc)
						v = DictVectorizer()
						X = v.fit_transform(matrix)
						y_pred = model.predict(X)[:-1]
					else:
						vectors = [vectorize(infos, method, desc, test=True)[0] for instance_id, infos in corpus.items()]
						if 'ngramChar' in method:
							for num_desc in desc.values():
								if num_desc not in vectors[0]:
									vectors[0][num_desc] = 0
							dictvectorizer = DictVectorizer() # ii. Transformation en sparse matrix
							vectors = dictvectorizer.fit_transform(vectors)
						y_pred = model.predict(vectors)
					ids = [id for id in list(corpus.keys())]
					outFile_path = config['models_annots_dir'] + corpus_to_test.split('/')[-1].replace('.json', '') + '__' + train_corpus.split('/')[-1].replace('.json', '') + '_' + method + '_' + classif_name + '.model' + '.txt'
					outFile = io.open(outFile_path, mode='w', encoding='utf-8')
					for i in range(len(corpus.keys())):
						outFile.write(ids[i] + '\t' + y_pred[i] + '\n')
					outFile.close()
					print(corpus_to_test.split('/')[-1].replace('.json', '') + ' : done.')
