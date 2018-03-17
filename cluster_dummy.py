from __future__ import division, print_function
from elasticsearch import Elasticsearch
import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
import json
from sklearn.feature_extraction.text import CountVectorizer

es=Elasticsearch()                         #elasticsearch object created
i=0
element = []
#global features,x_coor,y_coor
with open('wiki_references_dummy.json') as json_data:
	for each_line in json_data:
		if i >5 :
			break           
		d=json.loads(each_line)                                                            #converts json to python dict
		getsource=es.get(index='ref_dummy',doc_type='wiki_references',id=i)
		s=""
		for word in getsource["_source"]["combined_topicsequence"]:
			s=s+" "+word
		element.append(s)
		i=i+1
#for sentence in element:
#		print(sentence)
	
vectorize=CountVectorizer()
#vectorize=TfidfTransformer(smooth_idf=False)
x=vectorize.fit_transform(element)
print(x.toarray())


