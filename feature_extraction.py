import math
import re
import nltk
import pandas as pd
import math

from bs4 import BeautifulSoup
from nltk.tokenize import RegexpTokenizer, sent_tokenize, \
word_tokenize
from nltk.corpus import cmudict
from textstat.textstat import textstat
from collections import defaultdict, deque, Counter


# Reading Combined Dataset
combined_dataset_file = "combined.csv"
df = pd.read_csv(combined_dataset_file, header =0)
df.drop(['Unnamed: 0'], inplace = True, axis = 1, errors='ignore')

#Initializing Carneige Mellon's Dictionary
word_dict = cmudict.dict()

def body_word_ari_gunning():
	"""
	Body metrics

	1)generating a body word count column
	2)Automated reading index-
		4.71*(characters/words)+0.5*(words/sentences)-21.43
	3)Gunning Fox Index
	"""
	body_word_length = []
	body_sentences_length = []
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
		body_sentences_length.append(sentences_count)
		
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
	df['BodySentencesCount'] = body_sentences_length
	df['ValidGrammar'] = valid_grammar_flag
	df['AutomatedReadingIndex'] = automated_reading_index
	df['GunningFogIndex'] = gunning_fog
	df.to_csv('combined.csv')


#Checking for the occurence of specific tokens given a prefix
def markov_model(stream, model_order):
	model, stats = defaultdict(Counter), Counter()
	circular_buffer = deque(maxlen = model_order)

	for token in stream:
		prefix = tuple(circular_buffer)
		circular_buffer.append(token)
		if(len(prefix) == model_order):
			stats[prefix] += 1
			model[prefix][token] += 1

	return model,stats


#Formula to calculate entropy for the question body
def entropy(stats, normalization_factor):
	ans = 0
	for proba in stats.values():
		if(proba / float(normalization_factor)) > 0:
			ans += -1 * ((proba / float(normalization_factor)) * math.log(proba / float(normalization_factor), 2))
	return (ans)
 

#Formula to calculate entropy rate for the question body
def entropy_rate(model, stats):
	ans = 0
	for prefix in stats:
		ans += (stats[prefix] * entropy(model[prefix], stats[prefix]))
	if(float(sum(stats.values())) > 0):
		ans = ans / float(sum(stats.values()))
	else:
		ans = 0
	return ans

#Function to calculate metric entropy
def metric_entropy():
	tokenizer = RegexpTokenizer(r'\w+')
	c = 0
	randomness_info = []
	for index, row in df.iterrows():

		body_only = re.sub('<code>[^>]+</code>', '', row['Body'])
		soup = BeautifulSoup(body_only,"lxml")
		tag_removed_text = soup.text
		tag_removed_text = tag_removed_text.replace("\n","")
		char_in_body = list(tag_removed_text)
		character_count = len(tag_removed_text)
		model, stats = markov_model(char_in_body, 2)
		if float(character_count) > 0:
			randomness_info.append(entropy_rate(model, stats) / float(character_count))
		else:
			c += 1
			randomness_info.append(0)

	print c
	df['BodyMetricEntropy'] = randomness_info
	df.to_csv('combined.csv')

# lambda function to check if the word is valid and is not a punctuation
not_punctuation = lambda w: not (len(w)==1 and (not w.isalpha()))

# Counting number of syllabels in each word using Carnegie Mellon's Pronouncing Dictionary
def get_syllabels_count(valid_words):
	syllabels_count = 0
	for word in valid_words:
		try:
			max_value = 0
			pronounciations = word_dict[word]
			for i in pronounciations:
				value = 0
				for j in i:
					if j[-1].isdigit():
						value+=int(j[-1])
				if value > max_value:
					max_value = value
			syllabels_count += max_value
		except KeyError:
			continue
	return syllabels_count

#Function to calculate Flesch Formula
def flesch_formula(word_count, sent_count, syllable_count):
	if sent_count > 0 and word_count > 0:
		return 206.835 - 1.015*word_count/sent_count - 84.6*syllable_count/word_count
	else:
		return 0		

#Calculating flesch score for readability
def flesch_score():
	tokenizer = RegexpTokenizer(r'\w+')
	final_flesch_score = []
	valid_words = []
	for index, row in df.iterrows():
		body_only = re.sub('<code>[^>]+</code>', '', row['Body'])
		soup = BeautifulSoup(body_only,"lxml")
		word_tokens = tokenizer.tokenize(soup.text)
		for word in word_tokens:
			if not_punctuation(word):
				valid_words.append(word)
		word_count = len(valid_words)
		tag_removed_text = soup.text
		tag_removed_text = tag_removed_text.replace("\n","")
		syllabels_count = get_syllabels_count(valid_words)
		sentence_token = sent_tokenize(tag_removed_text)
		sentences_count = len(sentence_token)
		final_flesch_score.append(flesch_formula(word_count, sentences_count, syllabels_count))

	df['BodyFleschKinkaidGradeLevel'] = final_flesch_score
	df.to_csv("combined.csv")


def assign_0_or_1():

	open_or_closed = []
	for index, row in df.iterrows():
		if row['Reason'] == 'open':
			open_or_closed.append(1)
		else:
			open_or_closed.append(0)
	df['OpenOrClosed'] = open_or_closed

	df.to_csv("combined.csv")





if __name__ == '__main__':
	#body_word_ari_gunning()
	#metric_entropy()
	#flesch_score()
	#assign_0_or_1()
	print df.head()
	print df.columns.values
	print df.shape


