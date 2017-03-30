import math
import re
import nltk

import pandas as pd

from bs4 import BeautifulSoup
from nltk.tokenize import RegexpTokenizer, sent_tokenize, \
word_tokenize 
from textstat.textstat import textstat

combined_dataset_file = "combined.csv"

df = pd.read_csv(combined_dataset_file, header =0)
df.drop(['Unnamed: 0'], inplace = True, axis = 1, errors='ignore')

print df.columns.values
print df.shape

def body_word_ari_gunning():
	"""
	Body metrics

	1)generating a body word count column
	2)Automated reading index-
		4.71*(characters/words)+0.5*(words/sentences)-21.43
	3)Gunning Fox Index
	"""
	body_word_length = []
	sentence_token = []
	valid_grammar_flag = []
	automated_reading_index = []
	gunning_fog = []
	tokenizer = RegexpTokenizer(r'\w+')
	for index, row in df.iterrows():
		"""
		use BeautifulSoup to remove tags if required.
		However Body word count should contain the whole body 
		including any tags such as <p> and <code>
		"""
		body_only = re.sub('<code>[^>]+</code>', '', row['Body'])
		soup = BeautifulSoup(body_only,"lxml")
		word_tokens = tokenizer.tokenize(soup.text)
		word_count = len(word_tokens)
		body_word_length.append(word_count)
		tag_removed_text = soup.text
		tag_removed_text = tag_removed_text.replace("\n","")
		character_count = len(tag_removed_text)
		valid_sentence = re.findall('[\w][.?!]\s[A-Z0-9]',tag_removed_text)
		sentence_token = sent_tokenize(tag_removed_text)
		sentences_count = len(sentence_token)
		if((len(valid_sentence)+1)==len(sentence_token)):
			valid_grammar_flag.append(1)
		else:
			valid_grammar_flag.append(0)
		if sentences_count!=0 and word_count!=0:
			ari = 4.71 * (character_count/word_count) + 0.5 * (word_count/sentences_count) - 21.43
			gfi = textstat.gunning_fog(tag_removed_text)
		else:
			ari = 14
			gfi = 17
		automated_reading_index.append(ari)
		gunning_fog.append(gfi)

	df['BodyWordCount'] = body_word_length
	df['ValidGrammar'] = valid_grammar_flag
	df['AutomatedReadingIndex'] = automated_reading_index
	df['GunningFogIndex'] = gunning_fog
	df.to_csv('combined.csv')

# def average_term_entropy():
	#average term entropy in body
	#DONT KNOW WHAT THE EFF IT IS but it works :D
	# for index, row in df.iterrows():
	# 	freqdist = nltk.FreqDist(row['Body'])
	# 	probs = [freqdist.freq(l) for l in freqdist]	
	# 	entropy = -sum(p * math.log(p,2) for p in probs)
	# 	print entropy
	# 	break

if __name__ == '__main__':
	body_word_ari_gunning()
	print df.columns.values
	print df.shape


