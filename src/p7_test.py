import os, enchant, csv, json

DIR = "../data/essays/train/omit_p7/"

for file in os.listdir(DIR):
	sansp7 = ""
	with open("keywords_P7.json", "r") as f:
		keywords = json.load(f)
	with open(DIR+file, "r") as g:
		raw = g.read().split(" ")
		for tok in raw:
			if tok.lower() in keywords:
				print(file)