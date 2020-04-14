import sys
import yaml
import pickle
import json
import io
import spacy
import optparse
import statistics
from sklearn.metrics import precision_recall_fscore_support
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


def prepare_data(config):
	dic_json = open_json(config['train_params']['corpus_path'])
	if 'rstr' in config['train_params']['mode']: # rstr, different because we have to calculate the desc matrix beforehand
		texts = [infos["text_line"] for x, infos in dic_json.items()]
		desc, matrix = get_matrix_rstr(config['train_params']['mode'], dic_json, desc_arg=None)
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
			if 'POS' in config['train_params']['mode']:
				res_vectorize = vectorize(infos, config['train_params']['mode'], desc=desc)
			else:
				res_vectorize = vectorize(infos, config['train_params']['mode'], desc=desc)
			X.append(res_vectorize[0])
			desc = res_vectorize[1]
		y = [infos['label'] for ID, infos in dic_json.items()]
		L_IDS = dic_json.keys()
	return X, y, L_IDS, desc

if __name__ == "__main__":
	# 1. CONFIG: Read the configs and store them into a dictionary 
	parser = optparse.OptionParser()
	parser.add_option("-c", "--config", dest="config",
		default = "config.yml",
		help="path to the config file")
	(options, args) = parser.parse_args()

	try:
		with open(options.config, 'r') as ymlfile:
			config = yaml.safe_load(ymlfile)
	except (IOError, OSError):
		print(options.config)
		print('Impossible to read the configutations file. This program needs a config file named \'config.yml\'.')

	# 2. Launch processings
	if config['processus']['train_model'] == True:
		# a. Classifiers selection and creation
		mode_syn = config['train_params']['mode']
		model_path_syn = config['train_params']['model_path']
		for freq in ['abs', 'rel']:
			for i in range(1, 11):
				for j in range(1, 11):
					if j >= i:
						config['train_params']['mode'] = mode_syn.replace('_n,', '_'+str(i)+',').replace(',N', ','+str(j)).replace('freq', freq)
						config['train_params']['model_path'] = model_path_syn.replace('_n,', '_'+str(i)+',').replace(',N', ','+str(j)).replace('freq', freq)
						classifiers = []
						if 'MNB' in config['train_params']['algos']:
							classifiers.append(['MNB', MultinomialNB()])
						if 'DT10' in config['train_params']['algos']:
							classifiers.append(['DT10', DecisionTreeClassifier(max_depth=10)])
						if 'RF50' in config['train_params']['algos']:
							classifiers.append(['RF50', RandomForestClassifier(max_depth=50)]) #nb_estimator = nb arbres
						if 'RF100' in config['train_params']['algos']:
							classifiers.append(['RF100', RandomForestClassifier(max_depth=100)]) #nb_estimator = nb arbres
						if 'SVM' in config['train_params']['algos']:
							classifiers.append(['SVM', svm.SVC()])
						# b. Data preparation
						X_train, y_train, IDs_train, desc = prepare_data(config)
						if 'ngramChar' in config['train_params']['mode']:
							dictvectorizer = DictVectorizer()
							X_train = dictvectorizer.fit_transform(X_train)
						# c. Learning
						for name_classif, OBJ in classifiers:
							print('Learning with ' + name_classif)
							if config['train_params']['CV'] == True: # Cross-validation
								X = X_train
								y = y_train
								skf = StratifiedKFold(n_splits=10, shuffle=True)
								cpt = 1
								for train_index, test_index in skf.split(X, y):
									X_train, X_test = X[train_index], X[test_index]
									y_train, y_test = y[train_index], y[test_index]
									model = OBJ.fit(X_train, y_train)
									model_name = config['train_params']['model_path'].replace('.model', '_'+name_classif+'_' + str(cpt) +'.model')
									pickle.dump([model, desc], outFile)
									print('Done. Model saved here: ' + model_name)
									cpt += 1
							else: # No cross-validation
								model = OBJ.fit(X_train, y_train)
								model_name = config['train_params']['model_path'].replace('.model', '_'+name_classif+'.model')
								outFile = open(model_name, 'wb')
								pickle.dump([model, desc], outFile)
								print('Done. Model saved here: ' + model_name)

	if config['processus']['test_model'] == True:
		

		algo_type = 'RF100'



		for model in config['test_params']['models']:
			if algo_type not in model:
				continue

			config['test_params']['mode'] = model.replace('./models/deft2011_train_', '').replace('_'+algo_type+'.model', '')
			loaded_obj = pickle.load(open(model, 'rb'))
			loaded_model, desc = loaded_obj[0], loaded_obj[1]
			corpus = open_json(config['test_params']['corpus_ref_path'])
			if 'rstr' in config['test_params']['mode']:
				_, matrix = get_matrix_rstr(config['test_params']['mode'], corpus, desc_arg=desc)
				v = DictVectorizer()
				X = v.fit_transform(matrix)
				y_ref = [info['label'] for instance_id, info in corpus.items()]
				y_pred = loaded_model.predict(X)[:-1]
			else:
				vectors = [vectorize(infos, config['test_params']['mode'], desc, test=True)[0] for instance_id, infos in corpus.items()]
				if 'ngramChar' in config['test_params']['mode']:
					for num_desc in desc.values():
						if num_desc not in vectors[0]:
							vectors[0][num_desc] = 0
					dictvectorizer = DictVectorizer() # ii. Transformation en sparse matrix
					vectors = dictvectorizer.fit_transform(vectors)
				y_ref = [info['label'] for instance_id, info in corpus.items()]
				y_pred = loaded_model.predict(vectors)
			labels = ['0', '1']
			percent_true = get_score(y_ref, y_pred)
			acc = str(round(percent_true, 4))
			res = precision_recall_fscore_support(y_ref, y_pred, average=None, labels=labels)
			name = model.split('/')[-1]
			for i in range(len(labels)):
				l = labels[i]
				r = str(round(res[1][i], 4))
				p = str(round(res[0][i], 4))
				f = str(round(res[2][i], 4))
				nb = str(round(res[3][i], 4))
				if i == 0:
					ligne = name + '\tfdeft2011 test\t' + acc + '\t' + l + '\t' + p + '\t' + r + '\t' + f + '\t' + nb
				else:
					ligne = '\tdeft2011 test\t\t' + l + '\t' + p + '\t' + r + '\t' + f + '\t' + nb
				ligne = ligne.replace('.', ',')
				print(ligne)
			# Macro moyenne
			p, r, f = precision_recall_fscore_support(y_ref, y_pred, average='macro')[0],precision_recall_fscore_support(y_ref, y_pred, average='macro')[1], precision_recall_fscore_support(y_ref, y_pred, average='macro')[2]
			p, r, f = str(round(p, 4)), str(round(r, 4)), str(round(f, 4))
			ligne = '\tdeft2011 test\t\tMacromoyenne\t' + p + '\t' + r + '\t' + f + '\t'
			print(ligne.replace('.', ','))

	if config['processus']['use_model'] == True:
		loaded_obj = pickle.load(open(config['use_model_params']['model'], 'rb'))
		loaded_model, desc = loaded_obj[0], loaded_obj[1]
		corpus = open_json(config['use_model_params']['corpus_unLabelled_path'])
		if 'rstr' in config['use_model_params']['mode']:
			_, matrix = get_matrix_rstr(config['train_params']['mode'], corpus, desc_arg=desc)
			v = DictVectorizer()
			X = v.fit_transform(matrix)
			y_pred = loaded_model.predict(X)[:-1]
		else:
			vectors = [vectorize(infos, config['use_model_params']['mode'], desc, test=True)[0] for instance_id, infos in corpus.items()]
			if 'ngramChar' in config['use_model_params']['mode']:
				for num_desc in desc.values():
					if num_desc not in vectors[0]:
						vectors[0][num_desc] = 0
				dictvectorizer = DictVectorizer() # ii. Transformation en sparse matrix
				vectors = dictvectorizer.fit_transform(vectors)
			y_pred = loaded_model.predict(vectors)
		ids = [id for id in list(corpus.keys())]
		outFile = io.open(config['use_model_params']['corpus_labelled_path'], mode='w', encoding='utf-8')
		for i in range(len(corpus.keys())):
			outFile.write(ids[i] + '\t' + y_pred[i] + '\n')
		outFile.close()

	if config['processus']['use_model_only_proba'] == True:
		# That means we only want to give a proba that the instance is a Title.
		loaded_obj = pickle.load(open(config['use_model_only_proba_params']['model'], 'rb'))
		loaded_model, desc = loaded_obj[0], loaded_obj[1]
		corpus = open_json(config['use_model_only_proba_params']['corpus_unLabelled_path'])
		if 'rstr' in config['use_model_only_proba_params']['mode']:
			_, matrix = get_matrix_rstr(config['use_model_only_proba_params']['mode'], corpus, desc_arg=desc)
			v = DictVectorizer()
			X = v.fit_transform(matrix)
			y_pred = loaded_model.predict(X)[:-1]
		else:
			vectors = [vectorize(infos, config['use_model_only_proba_params']['mode'], desc, test=True)[0] for instance_id, infos in corpus.items()]
			if 'ngramChar' in config['use_model_only_proba_params']['mode']:
				for num_desc in desc.values():
					if num_desc not in vectors[0]:
						vectors[0][num_desc] = 0
				dictvectorizer = DictVectorizer() # ii. Transformation en sparse matrix
				vectors = dictvectorizer.fit_transform(vectors)
			probas = loaded_model.predict_proba(vectors)
	
		ids = [id for id in list(corpus.keys())]
		outFile = io.open(config['use_model_only_proba_params']['corpus_labelled_path'], mode='w', encoding='utf-8')
		for i in range(len(corpus.keys())):
			outFile.write(ids[i] + '\t' + str(probas[i][1]) + '\n')
		outFile.close()
		print()
		print(statistics.mean([probas[i][1] for i in range(len(probas))]))