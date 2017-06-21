import pickle

f = open("f5_test-fold-5-train-cbow-512-probs.pickle", "rb")
probs = pickle.load(f)

for k in probs.keys():
	print(k)

probys = [[probs[k][j] for j in probs[k].keys()] for k in probs.keys()]
