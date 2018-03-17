from __future__ import division, print_function
from elasticsearch import Elasticsearch
import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from scipy.spatial import distance
from pprint import pprint
from sklearn.mixture import GaussianMixture
from sklearn.mixture import BayesianGaussianMixture
from sklearn.decomposition import LatentDirichletAllocation




es=Elasticsearch()                         #elasticsearch object created
i=0
element = []
cluster_list=[]
cluster_list1=[]
cluster_list2=[]
cluster_list3=[]
cluster_list4=[]
cluster_list5=[]
with open('wiki_references_dummy.json') as json_data:
	for each_line in json_data: 
		if i>20:
			break
		d=json.loads(each_line)                                                            #converts json to python dict
		getsource=es.get(index='ref_dummy',doc_type='wiki_references',id=i)
		s=""
		for word in getsource["_source"]["combined_topicsequence"]:
			s=s+" "+word
		element.append(s)
		i=i+1

#for sentence in element:
#	print (sentence)
		
length_element=len(element)
#print (length_element)	
vectorize=CountVectorizer()
x=vectorize.fit_transform(element)
matrix=x.toarray()


#gmm=GaussianMixture(covariance_type='full', max_iter=100, n_components=6)
#gmm.fit(tfid_matrix)
#print(gmm.predict_proba(tfid_matrix))
#pprint(gmm.predict_proba(tfid_matrix).tolist())


gmm=GaussianMixture(covariance_type='full', max_iter=100, n_components=6)
gmm.fit(matrix)
index=0
for ary in gmm.predict_proba(matrix).tolist():
	#print(str(index)+" "+str(ary))
	index=index+1

count=0
count1=0
count2=0
count3=0
count4=0
count5=0
index1=0
index2=[]
for arr in gmm.predict_proba(matrix).tolist():
	if arr[0]!=0:
		count=count+1
		cluster_list.append(arr)
		index2.append(index1)
	index1=index1+1
print ("cluster no 1"+"- "+"total no of clusters:"+str(count)+"- "+"cluster index:"+str(index2))
print (cluster_list)


index2=0
index3=[]
for arr in gmm.predict_proba(matrix).tolist():
	if arr[1]!=0:
		count1=count1+1
		cluster_list1.append(arr)
		index3.append(index2)
	index2=index2+1
print("cluster no 2"+"- "+"total no of clusters:"+str(count1)+"- "+"cluster index:"+str(index3))
print (cluster_list1)

index3=0
index4=[]
for arr in gmm.predict_proba(matrix).tolist():		
	if arr[2]!=0:
		count2=count2+1
		cluster_list2.append(arr)
		index4.append(index3)
	index3=index3+1
print("cluster no 3"+"- "+"total no of clusters:"+str(count2)+"- "+"cluster index:"+str(index4))
print (cluster_list2)

index4=0
index5=[]
for arr in gmm.predict_proba(matrix).tolist():	
	if arr[3]!=0:
		count3=count3+1
		cluster_list3.append(arr)
		index5.append(index4)
	index4=index4+1
print("cluster no 4"+"- "+"total no of clusters:"+str(count3)+"- "+"cluster index:"+str(index5))
print (cluster_list3)

index5=0
index6=[]
for arr in gmm.predict_proba(matrix).tolist():		
	if arr[4]!=0:
		count4=count4+1
		cluster_list4.append(arr)
		index6.append(index5)
	index5=index5+1
print("cluster no 5"+"- "+"total no of clusters:"+str(count4)+"- "+"cluster index:"+str(index6))
print (cluster_list4)

index6=0
index7=[]
for arr in gmm.predict_proba(matrix).tolist():
	if arr[5]!=0:
		count5=count5+1
		cluster_list5.append(arr)
		index7.append(index6)
	index6=index6+1
print("cluster no 6"+"- "+"total no of clusters:"+str(count5)+"- "+"cluster index:"+str(index7))
print(cluster_list5)
		












