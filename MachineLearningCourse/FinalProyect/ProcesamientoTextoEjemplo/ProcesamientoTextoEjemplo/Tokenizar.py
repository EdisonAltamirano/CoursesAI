# -*- coding: utf-8 -*-
import codecs
import os, sys
import re

from nltk.tokenize import word_tokenize as wt

def Tokeniza(doc):
	docto=codecs.open('Iniciales/'+doc+'.txt','r')
	Salida=codecs.open('Documentos/'+doc+'-Tokens.txt','w')
	for x in docto.readlines():
		if len(x)>1:
			datos=x.split('\t')
			Texto=wt(datos[1].lower())
			CadFinal = " ".join(Texto)
			Salida.write(datos[0]+'\t'+CadFinal+'\t'+datos[2])
	docto.close()
	Salida.close()

##############################################################	
#Titulo=sys.argv[1]


Tokeniza('NWH_Ingles')

Tokeniza('NWH_Espanol')

