import os, enchant, csv, json

DIR = "../data/essays/dev/omit_p7/"

for file in os.listdir(DIR):
	sansp7 = ""
	with open("keywords_P7.json", "r") as f:
		keywords = json.load(f)
	with open(DIR+file, "r") as g:
		raw = g.read().split(" ")
		for tok in raw:
			if tok.lower() not in keywords:
				sansp7 = sansp7+tok+" "
	with open("../data/essays/dev/omit_p7/"+file, "w") as h:
		h.write(sansp7)