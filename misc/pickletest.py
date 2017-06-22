import pickle

cbow_train = {}
for i in range(1, 6):
	f = open("../misc/f{}_test-fold-{}-train-cbow-512-probs.pickle".format(i,i), "rb")
	fold_dict = pickle.load(f)
	cbow_train.update(fold_dict)

with open("train_cbow.pickle", "wb") as g:
	pickle.dump(cbow_train, g)

# probys = [[probs[k][j] for j in probs[k].keys()] for k in probs.keys()]
