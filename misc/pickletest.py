import pickle

with open("train_cbow.pickle", "rb") as f:
	train_probs = pickle.load(f)

with open("dev-train-cbow-512-bs50-e20-traineradam-dropout0.1-s113-probs.pickle", "rb") as f:
	dev_probs = pickle.load(f)

train_probs.update(dev_probs)
with open("train_dev_cbow.pickle", "wb") as f:
	pickle.dump(train_probs, f)
# probys = [[probs[k][j] for j in probs[k].keys()] for k in probs.keys()]