from __future__ import division, print_function
from elasticsearch import Elasticsearch
import json
from sklearn.feature_extraction.text import CountVectorizer
#from pprint import pprint
from sklearn.mixture import GaussianMixture
from math import log
import numpy as np
from sklearn.metrics.cluster import entropy

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
		if i>10:
			break
		d=json.loads(each_line)                                                            #converts json to python dict
		getsource=es.get(index='ref_dummy',doc_type='wiki_references',id=i)
		s=""
		for word in getsource["_source"]["combined_topicsequence"]:
			s=s+" "+word
		element.append(s)
		i=i+1
		
length_element=len(element)
vectorize=CountVectorizer()
x=vectorize.fit_transform(element)
matrix=x.toarray()
print(str(len(matrix[0])))
np.savetxt("dataset_new.csv", matrix, delimiter=",")

gmm=GaussianMixture(covariance_type='full', max_iter=100, n_components=6)
gmm.fit(matrix)

index=0
for ary in gmm.predict_proba(matrix).tolist():
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
prob=-(count/(index-1)*log(count/(index-1), 2))
print ("entropy for cluster 1:"+str(prob))


index2=0
index3=[]
for arr in gmm.predict_proba(matrix).tolist():
	if arr[1]!=0:
		count1=count1+1
		cluster_list1.append(arr)
		index3.append(index2)
	index2=index2+1

prob1=-(count1/(index-1)*log(count1/(index-1), 2))
print ("entropy for cluster 2:"+str(prob1))

index3=0
index4=[]
for arr in gmm.predict_proba(matrix).tolist():		
	if arr[2]!=0:
		count2=count2+1
		cluster_list2.append(arr)
		index4.append(index3)
	index3=index3+1
prob2=-(count2/(index-1)*log(count2/(index-1), 2))
print ("entropy for cluster 3:"+str(prob2))

index4=0
index5=[]
for arr in gmm.predict_proba(matrix).tolist():	
	if arr[3]!=0:
		count3=count3+1
		cluster_list3.append(arr)
		index5.append(index4)
	index4=index4+1
prob3=-(count3/(index-1)*log(count3/(index-1), 2))
print ("entropy for cluster 4:"+str(prob3))

index5=0
index6=[]
for arr in gmm.predict_proba(matrix).tolist():		
	if arr[4]!=0:
		count4=count4+1
		cluster_list4.append(arr)
		index6.append(index5)
	index5=index5+1

prob4=-(count4/(index-1)*log(count4/(index-1), 2))
print ("entropy for cluster 5:"+str(prob4))

index6=0
index7=[]
for arr in gmm.predict_proba(matrix).tolist():
	if arr[5]!=0:
		count5=count5+1
		cluster_list5.append(arr)
		index7.append(index6)
	index6=index6+1

prob5=-(count5/(index-1)*log(count5/(index-1), 2))
print ("entropy for cluster 6:"+str(prob5))

sum=-(prob5*log(prob5)+prob4*log(prob4)+prob3*log(prob3)+prob2*log(prob2)+prob1*log(prob1)+prob*log(prob))
print ("entropy:"+str(sum))











