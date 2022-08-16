# -*- coding: utf-8 -*-
import codecs
import os, sys
import re
from nltk.stem import PorterStemmer

def Stemming(doc):
	docto=codecs.open('Documentos/'+doc+'-Lemas.txt','r')
	Salida=codecs.open('Documentos/'+doc+'-Sttemming.txt','w')
	for x in docto.readlines():
		if len(x)>1:
			datos=x.split('\t')
			Texto=datos[1].lower().split(' ')
			CadFinal=""
			for w in Texto:
				print(w, ": ", ps.stem(w))
				CadFinal=CadFinal+ps.stem(w)+' '
			Salida.write(datos[0]+'\t'+CadFinal[1:]+'\t'+datos[2])
	docto.close()
	Salida.close()	
##############################################################	
ps = PorterStemmer()
Stemming('NWH_Ingles')
Stemming('NWH_Espanol')

