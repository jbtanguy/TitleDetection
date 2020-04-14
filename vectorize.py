import re

def get_stylo(s):
	patterns = [",", ";", "_", "-", "\(", "\)", "\.", "[A-Z]","[0-9]"]
	L = [len(re.findall(patt, s)) for patt in patterns]
	return L 

def vector_ngrams_pos(instance_info, labels):
	vec = []
	for pos in labels:
		if pos in instance_info['pos']:
			vec.append(instance_info['pos'].count(pos))
		else:
			vec.append(0)
	return vec

def vectorize_baseline(instance_info, mode):
	vec = []
	#B1:feat base, B2:base + longueur
	#B3:stylo, B4: stylo + base, B5:stylo+longueur 
	#B6:stylo+base+longueur
	#B7:stylo+base+longueur+ngramsPOS
	features_base = ["begins_with_numbering", "is_italic","is_all_caps", "begins_with_cap", "page_nb"]
	if mode != 'B3' and mode != 'B5':
		for feat in features_base:
			vec.append(int(instance_info[feat]))
	if mode == 'B2' or mode == 'B5' or mode == 'B6':
		vec.append(len(instance_info['text_line']))
	texte = instance_info['text_line']
	if mode != 'B1' and mode != 'B2':
		stylos = get_stylo(texte)
		for feat_style in stylos:
			vec.append(feat_style)
	if mode == 'B7':
		pos_labels = ['SYM', 'PUNCT', 'ADJ', 'CCONJ', 'NUM', 'DET', 'PRON', 'ADP', 'VERB', 'NOUN', 'PROPN', 'PART', 'ADV', 'INTJ', 'X', 'SPACE']
		vec_pos = vector_ngrams_pos(instance_info, pos_labels)
		vec = vec + vec_pos
	return vec

def vectorize_ngramChar(instance_info, mode, desc, test=False):
	s = '$$%s^^'%instance_info['text_line']
	m, M = [int(x) for x in re.split(',', mode.split('_')[-1])]
	freq = mode.split('_')[1] # 'abs' or 'rel'
	occs = {}
	for deb in range(len(s)):
		for i in range(m, M+1):
			d = s[deb:deb+i]
			if test == True:
				if d not in desc:
					continue
			else:
				desc.setdefault(d, len(desc))
			occs.setdefault(desc[d], 0)
			if freq == 'abs':
				occs[desc[d]] += 1
			else: # rel
				occs[desc[d]] += 1/float(len(s))
	return occs, desc

def vectorize_ngramPOS(instance_info, mode, desc):
	vec = []
	pos_labels = ['SYM', 'PUNCT', 'ADJ', 'CCONJ', 'NUM', 'DET', 'PRON', 'ADP', 'VERB', 'NOUN', 'PROPN', 'PART', 'ADV', 'INTJ', 'X', 'SPACE']
	for pos in pos_labels:
		if pos in instance_info['pos']:
			if mode.split('_')[1] == 'rel':
				vec.append(instance_info['pos'].count(pos)/len(pos_labels))
			else: # freq = abs
				vec.append(instance_info['pos'].count(pos))
		else:
			vec.append(0)
	return vec

def vectorize(instance_info, mode, desc={}, test=False, texts=[]):
	# Les modes possibles sont :
	# B1, B2, B3, B4, B5, B6, B7, ngramChar_freq_m,M, ngramPOS_freq, rstr --> freq(abs ou rel), m et M = des entiers
	if 'B' in mode: # Baseline
		vec = vectorize_baseline(instance_info, mode)
	elif 'ngramChar' in mode:
		vec, desc = vectorize_ngramChar(instance_info, mode, desc, test)
	elif 'ngramPOS' in mode:
		vec = vectorize_ngramPOS(instance_info, mode, desc)
	else:
		vec = []
		print('/!\\ Impossible to vectorize the instance !')
	return vec, desc



