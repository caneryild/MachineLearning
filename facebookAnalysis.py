#Facebook Dataset is used to find out popularities via PageRank(with teleportation) , similarities
#(jacckardt distance similarty and cosine distance similarity) and clusers detection(with kmeans and hierarchical clustering)
#Caner Yildirim
#caneryild163@gmail.com
#https://snap.stanford.edu/data/egonets-Facebook.html
import numpy as np
import networkx as nx
#import itertools as it
from math import sqrt
from numpy import vstack,array
from numpy.random import rand
from scipy.cluster.vq import kmeans,vq
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import cophenet
import operator
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
#sudo apt-get instal python-scipy
#sudo apt-get instal python-numpy
#sudo apt-get instal python-networkx
#sudo apt-get instal python-matplotlib
#max_iter=10 and eps 1e-4 is the best for SIMRANK--------------------------------
'''def simrank(G, r=0.8, max_iter=10, eps=1e-4):

    nodes = G.nodes()
    nodes_i = {k: v for(k, v) in [(nodes[i], i) for i in range(0, len(nodes))]}

    sim_prev = np.zeros(len(nodes))
    sim = np.identity(len(nodes))

    for i in range(max_iter):
        if np.allclose(sim, sim_prev, atol=eps):
            break
        sim_prev = np.copy(sim)
        for u, v in it.product(nodes, nodes):
            if u is v:
                continue
            u_ns, v_ns = G.predecessors(u), G.predecessors(v)

            # evaluating the similarity of current iteration nodes pair
            if len(u_ns) == 0 or len(v_ns) == 0: 
                # if a node has no predecessors then setting similarity to zero
                sim[nodes_i[u]][nodes_i[v]] = 0
            else:                    
                s_uv = sum([sim_prev[nodes_i[u_n]][nodes_i[v_n]] for u_n, v_n in it.product(u_ns, v_ns)])
                sim[nodes_i[u]][nodes_i[v]] = (r * s_uv) / (len(u_ns) * len(v_ns))


    return sim
'''
#cosine similarity-------------------------	
def cosdist( features,featlineno,featsize):
	resultCosDist = [[0 for size1 in range(featlineno)] for size2 in range(featlineno)] 
	for index1 in range(0,featlineno):
		for index2 in range(0,featlineno):
			cosDist=0
			leftsize=0
			rightsize=0
			if(index1<index2):				
				for index3 in range(1,featsize):
					cosDist+=features[index1][index3]*features[index2][index3]
					leftsize+=features[index1][index3]
					rightsize+=features[index2][index3]
				leftsize=sqrt(leftsize)
				rightsize=sqrt(rightsize)
				if leftsize*rightsize !=0 : 
					cosDist=(float(cosDist))/(leftsize*rightsize)
				else:
					cosDist=0
				resultCosDist[index1][index2]=cosDist
				resultCosDist[index2][index1]=cosDist
	return resultCosDist
	
#Jaccard Sim----------------------------------
def jaccardSim( features,featlineno,featsize):
	resultCosDist = [[0 for size1 in range(featlineno)] for size2 in range(featlineno)] 
	for index1 in range(0,featlineno):
		for index2 in range(0,featlineno):
			totsize=0
			jaccRet=0
			if(index1<index2):				
				for index3 in range(1,featsize):
					jaccRet+=features[index1][index3]*features[index2][index3]
					totsize+= features[index1][index3] or features[index2][index3] 
				if totsize !=0:
					jaccRet=float(jaccRet)/totsize
				else:
					jaccRet=0
				resultCosDist[index1][index2]=jaccRet
				resultCosDist[index2][index1]=jaccRet
	return resultCosDist
	
size1=4039
y=[]
ys=[[0 for size in range(size1)] for size in range(size1)] 
dataset=open('fb/fbFriendshipEdges.txt','r')
AllPageRankMatrix = [[0 for size in range(size1)] for size in range(size1)] 
AllPageRank= [0 for size in range(size1)] 
FriendNos = [0 for size in range(size1)]

#Get number of friends for each user
for line in dataset:
	x,y=line.split()
	x=int(x)
	y=int(y)
	ys[x][y]=1 #adjecency list to matrix, dataset now meaningful for the app
	FriendNos[x]=FriendNos[x]+1
	
#The Pagerank Vector is created	
for index1 in range(size1) :
	AllPageRank[index1]=1/size1
	#print(str(index)+" has "+str(FriendNos[index1])+" friends\n")
	#not print , visualized in next

	
#The Pagerank Matrice is created
for index1 in range(size1) :
	for index2 in range(size1) :
		if(FriendNos[index1]==0):
			pass
		else:
			AllPageRankMatrix[index1][index2]=ys[index1][index2]/FriendNos[index1]
		

npAllPageRankMatrix = np.matrix(np.array(AllPageRankMatrix))
npAllPageRank = np.transpose(np.matrix(np.array(AllPageRank)))
diGraphAllAdjMatrix = nx.DiGraph(npAllPageRankMatrix)

#POPULAR PEOPLE---------
#Reiterate X times matrix vector multipication, %15 s value can be used
'''for iterNo in range(5):
	beta=0.85
	npAllPageRank=npAllPageRankMatrix*npAllPageRank*0.85+0.15/size1
	'''
#print(npAllPageRank)
#OR------------------------------------------------
plt.plot(FriendNos)#show ids vs. friend nos
plt.show()

D=nx.pagerank(diGraphAllAdjMatrix,alpha=0.85, max_iter=100, tol=1e-06)
plt.bar(range(len(D)), D.values(), align='center')
plt.xticks(range(len(D)), D.keys())

plt.plot(D.values())#show ids vs. popularity,pagerank
plt.show()
#for sorting and showing the most popular one 
#for key, value in sorted(D.iteritems(), key=lambda (k,v): (v,k),reverse=True):
   # print "%s: %s" % (key, value) 

#SIMRANK------Too much time 1 and half hour------------
#print(simrank(diGraphAllAdjMatrix))


#FINISH	PART 1------------------------------------------------------------
#START	PART 2------------------------------------------------------------
dataset2=open('fb/part2/3980.edges','r')
size2=4039-3979
first=3980
ys2=[[0 for size in range(size2)] for size1 in range(size2)] 
PartPageRankMatrix = [[0 for size in range(size2)] for size in range(size2)] 
PartPageRank= [0 for size in range(size2)] 
FriendNos2 = [0 for size in range(size2)]

#Get number of collegurs for each person
for line in dataset2:
	x,y=line.split()
	x=int(x)
	y=int(y)
	
	if x >= first & y>=first :
		ys2[x-first][y-first]=1 #adjecency list to matrix, dataset now meaningful for the app
		FriendNos2[x-first]=FriendNos2[x-first]+1

#The Pagerank Vector init in local spaca
for index1 in range(size2) :
	PartPageRank[index1]=1/size2
	#print(str(index1)+" has "+str(FriendNos2[index])+" friends\n")

	
#Adj. Matrix with weights
for index1 in range(size2) :
	for index2 in range(size2) :
		if(FriendNos2[index1]==0):
			pass
		else:
			PartPageRankMatrix[index1][index2]=ys2[index1][index2]/FriendNos2[index1]
		

npPartPageRankMatrix = np.matrix(np.array(PartPageRankMatrix))
npPartPageRank = np.transpose(np.matrix(np.array(PartPageRank)))
diGraphPartAdjMatrix = nx.DiGraph(npPartPageRankMatrix)
'''for iterNo in range(5):
	beta=0.85
	npPartPageRank=npPartPageRankMatrix*npPartPageRank*0.85+0.15/size1
	
	ORR
	'''
#in local part, popular people
#prpart=nx.pagerank(diGraphPartAdjMatrix,alpha=0.85, max_iter=100, tol=1e-06)
#for key, value in sorted(prpart.iteritems(), key=lambda (k,v): (v,k),reverse=True):
#    print "%s: %s" % (key, value)

#SIMRANK--------------------------------------------
#print(simrank(diGraphPartAdjMatrix))
	
circlesize=0
featsize=0

#--------------CIRCLES-----------------

circles=[]
circlepeople=[[0 for size in range(22)] for size in range(size2-1)] 
with open('fb/part2/3980.circles') as f:
	for line in f:
		line = line.split() # to deal with blank 
		if line:            # lines (ie skip them)
			line = [int(i) for i in line]
		circles.append(line[1:])
		circlesize=circlesize+1		

#print(circles)
for index1 in range(circlesize):
	for index2 in range(len(circles[index1])):
		if circles[index1][index2] >= 3980:
		
			circlepeople[circles[index1][index2]-3980][index1]=1
		
cosDistsCirc=cosdist(circlepeople,size2-1,22)
#print(cosDistsCirc)
jsCirc=jaccardSim(circlepeople,size2-1,22)
#print(jsCirc)
#----------------FEATURES------------
feats=[]

with open('fb/part2/3980.feat') as f:
	for line in f:
		line = line.split() # to deal with blank 
		if line:            # lines (ie skip them)
			line = [int(i) for i in line]
			feats.append(line)
		featsize=featsize+1

cosDistsFeatures=cosdist(feats,featsize,42)
#print(cosDistsFeatures)
jsfea=jaccardSim(feats,featsize,42)
#print(jsfea)

#Lastly check whether 2 people have similar friends
#cosDistsFriendship=cosdist(PartPageRankMatrix[:][:],4039-3980,4039-3980)
#print(cosDistsFriendship)
#Compare People
#jsfri=jaccardSim(PartPageRankMatrix[:][:],4039-3980,4039-3980)
#print(jsfri)

#output overall similarity
print("Similarity matrix according to cos distances")
print((np.matrix(cosDistsFeatures)+np.matrix(cosDistsCirc[:][:]))/2)

print("Similarity matrix according to jaccard distances")
print((np.matrix(jsfea)+np.matrix(jsCirc))/2)
#Kmeands CLUSTER------------------------
concatted=np.concatenate((np.array(circlepeople), np.array(feats)), axis=1)
for num in range(2,10):
	print("k=")
	print(num)
	
	codebook, distortion = kmeans(concatted, num)
	code, dist = vq(concatted, codebook)
	print(code)

#centroids, labels = kmeans([ys2,circles,feats], 3)
#Hiearchical clustering

Z = linkage(concatted, 'ward')
c, coph_dists = cophenet(Z, pdist(concatted))
plt.figure(figsize=(25, 10))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(
    Z,
    leaf_rotation=90.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()
