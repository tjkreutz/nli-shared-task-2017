import os, enchant, csv, json

DIR = "sage_results/"
d = enchant.Dict("en_US")
prompt_words = []

for file in os.listdir(DIR):
	with open(DIR+file, 'r') as csvfile:
		reader = csv.reader(csvfile, delimiter=',', quotechar='|')
		count = 0
		for row in reader:
			if count == 100: break
			elif row:
				if d.check(row[0].strip()) == True:
					try:
						prompt_words.append(row[0].strip())
						count+=1
					except: TypeError

with open("keywords.json", "w") as out:
	json.dump(list(set(prompt_words)), out)

