import sys
sys.path.append('./rstr_max/')
from rstr_max import Rstr_max

def get_total_counts(obs):
	tot = 0
	for ss_id, count in obs.items():
		tot += count
	return tot

def relative_transformation(X):
	X_rel = []
	for obs in X:
		obs_rel = {}
		tot = get_total_counts(obs)
		for ss_id, count in obs.items():
			obs_rel[ss_id] = count / tot
		X_rel.append(obs_rel)
	return X_rel

def get_matrix_rstr(mode, dic_json, desc_arg=None):
	texts = [infos["text_line"] for x, infos in dic_json.items()]
	lenmax = mode.split('_')[-1].split(',')[0]
	supportmax = mode.split('_')[-1].split(',')[1]
	rstr = Rstr_max()
	X = [] 
	for s in texts:
  		rstr.add_str(s)
  		X.append({})
	r = rstr.go()

	cpt_str = 0
	desc = []
	for (offset_end, nb), (l, start_plage) in r.items():
		ss = rstr.global_suffix[offset_end-l:offset_end]
		list_occur = []
		for o in range(start_plage, start_plage+nb):
			id_text = rstr.idxString[rstr.res[o]]
			list_occur.append(id_text)
		set_occur = set(list_occur)
		# Ici, il y a un souci dans les dimensions puisque tous les descripteurs ne sont pas
		# forcément présents. Donc dans certains cas on n'a rien pour une instance donnée. 
		if desc_arg is not None and ss not in desc_arg:  # Test
			continue
		if len(set_occur) > 1:
			if len(ss) < int(lenmax) and len(set_occur) < float(supportmax)*len(texts):
				for id_text in list_occur:
					X[id_text].setdefault(cpt_str, 0)
					X[id_text][cpt_str] += 1
				if desc_arg is None: # corpus train = on ajoute le descripteur
					desc.append(ss)
				cpt_str += 1

	# Ajout d'une instance virtuelle pour garantir l'homogénéite dans les dimensions des matrices de train et test
	if desc_arg is None: # Train
		descriptors = desc
	else: # test
		descriptors = desc_arg
	dic = {}
	for d in descriptors:
		dic[descriptors.index(d)] = 1
	X.append(dic)

	if mode.split('_')[1] == 'rel':
		X = relative_transformation(X)

	return desc, X