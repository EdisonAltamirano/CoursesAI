# -*- coding: utf-8 -*-
import codecs
import os, sys
import re
import spacy
import nltk
import string 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize as wt
from nltk.stem import WordNetLemmatizer

 
lemma = WordNetLemmatizer()

def LematizaEspanol(doc):
	docto=codecs.open('Iniciales/'+doc+'.txt','r')
	Salida=codecs.open('Documentos/'+doc+'-Lemas.txt','w')
	Salida2=codecs.open('Documentos/'+doc+'-Categorias.txt','w')
	for x in docto.readlines():
		if len(x)>1:
			datos=x.split('\t')
			Texto=datos[1].lower()
			doc=nlp(Texto)
			CadLemas=""
			CadCategorias=""
			for token in doc:
				print(token.text+'/'+token.lemma_+'/'+token.pos_)
				CadCategorias=CadCategorias+token.pos_+' '
				CadLemas=CadLemas+token.lemma_+' '
			Salida.write(datos[0]+'\t'+CadLemas[1:]+'\t'+datos[2])
			Salida2.write(datos[0]+'\t'+CadCategorias[1:]+'\t'+datos[2])
	docto.close()
	Salida.close()
	Salida2.close()
##############################################################	
def LematizaIngles(doc):
	docto=codecs.open('Iniciales/'+doc+'.txt','r')
	Salida=codecs.open('Documentos/'+doc+'-Lemas.txt','w')
	Salida2=codecs.open('Documentos/'+doc+'-Categorias.txt','w')	
	cerradas=stopwords.words('english')
	Signos = string.punctuation
	print(Signos)
	for x in docto.readlines():
		if len(x)>1:
			datos=x.split('\t')
			Texto=wt(datos[1])
			print(datos)
			CadLemas=""
			for token in Texto:
				Lema=lemma.lemmatize(token)
				if not Lema in cerradas and not Lema[0] in Signos:
				#print(token+'-'+Lema)
					CadLemas=CadLemas+Lema.lower()+' '
			print("Original--->"+datos[1])
			print("Limpio-->"+CadLemas)
			Salida.write(datos[0]+'\t'+CadLemas[1:]+'\t'+datos[2])
			Tag=nltk.pos_tag(Texto)
			CadCategorias=""
			for y in Tag:
				#print(y[1])
				CadCategorias=CadCategorias+y[1]+' '
			Salida2.write(datos[0]+'\t'+CadCategorias[1:]+'\t'+datos[2])
			
	docto.close()
	Salida.close()
	Salida2.close()
#####################################################
nlp = spacy.load("es_core_news_sm")
LematizaIngles('NWH_Ingles')


# nlp = spacy.load("en_core_web_sm")
# LematizaEspanol('NWH_Espanol')

