import json
import os 
import io
from sklearn.metrics import precision_recall_fscore_support

def get_id_labels(file_path):
	id_labels = {}
	file = io.open(file_path, mode='r', encoding='utf-8')
	line = file.readline()
	while line != '':
		id_, label = line.replace('\n', '').split('\t')
		id_labels[str(id_)] = str(label)
		line = file.readline()
	return id_labels

def get_aligned_labels(ref_id_labels, test_id_labels):
	ref_labels, test_labels = [], []
	for id_ in ref_id_labels.keys():
		ref_labels.append(ref_id_labels[id_])
		test_labels.append(test_id_labels[id_])
	return (ref_labels, test_labels)

dir_path = './models_annots/english/'
ref_path = './corpora/english/fintoc2020_en_dev.json'
res_outFile_path = './evaluation/fintoc2020_en_dev__res_models.txt'
res_outFile = io.open(res_outFile_path, mode='w', encoding='utf-8')
ref_json = json.load(open(ref_path))
ref_id_labels = {str(id_): str(infos['label']) for id_, infos in ref_json.items()}

res_outFile.write('P, R et F-m calculés sur les corpus de développement de fintoc2020, comptant 1 322 titres et 35 912 non-titres pour l\'anglais et 1 076 titres et 12 544 non-titres pour le français.\n\n')
res_outFile.write('Modèle\tP (macro-moyenne)\tR (macro-moyenne)\tF-m (macro-moyenne)\tP (titre)\tR (titre)\tF-m (titre)\tP (non-titre)\tR (non-titre)\tF-m (non-titre)\n')

for test_file_name in os.listdir(dir_path):
	if 'dev' in test_file_name:
		test_id_labels = get_id_labels(dir_path+test_file_name)
		ref_labels, test_labels = get_aligned_labels(ref_id_labels, test_id_labels)
		macro_res = precision_recall_fscore_support(y_true=ref_labels, y_pred=test_labels, average='macro')
		label_res = precision_recall_fscore_support(y_true=ref_labels, y_pred=test_labels, average=None, labels=['0', '1'])
		res_outFile.write(test_file_name+'\t'+str(macro_res[0])+'\t'+str(macro_res[1])+'\t'+str(macro_res[2])+'\t'+str(label_res[0][1])+'\t'+str(label_res[1][1])+'\t'+str(label_res[2][1])+'\t'+str(label_res[0][0])+'\t'+str(label_res[1][0])+'\t'+str(label_res[2][0])+'\n')
