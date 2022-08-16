# -*- coding: utf-8 -*-
import codecs;
import os, sys
import sys, re
import re


Titulo=sys.argv[1]
docto=codecs.open('Documentos/NWH_'+Titulo+'.txt','r')
Voc={}
for x in docto.readlines():
	if len(x)>3:
		l=x.split("\t")
		Text=l[1].split(" ")
		for y in Text:
			if len(y)>0:
				if y in Voc:
					Voc[y]+=1
				else:
					Voc[y]=1
docto.close()


DocVoc=codecs.open('Vocabulario/'+Titulo+'.csv','w')
#al final, imprime cuantos elementos tienen frecuencia 1
Frec1=0
for x in Voc:
	DocVoc.write(x+","+str(Voc[x])+",\n")
	if Voc[x]==1:
		Frec1+=1
DocVoc.close()
print(Titulo+"--> "+str(Frec1))

