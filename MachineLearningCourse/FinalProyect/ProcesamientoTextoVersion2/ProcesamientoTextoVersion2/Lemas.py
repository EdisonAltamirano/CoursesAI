# -*- coding: utf-8 -*-
import codecs;
import os, sys
import re


from nltk.stem import WordNetLemmatizer

Lematizar=WordNetLemmatizer()
docto=codecs.open('Documentos/NWH_Ingles-Tokens.txt','r')
for x in docto.readlines():	
	datos=x.split('\t')
	Text=datos[2].split(' ')
	for y in range(len(Text)):
		print(Lematizar.lemmatize(Text[y]))

docto.close()
		
"""         
from pattern.es import parse, split 
reload(sys)
sys.setdefaultencoding("utf-8")
docto=codecs.open('Doctos/NWH_Espanol-Tokens.txt','r')
for x in docto.readlines():	
	datos=x.split('\t')
	s = parse(datos[2], lemmata=True)
	print(s)
docto.close()
"""
#ey/NN/B-NP/O/ey sorry/VB/B-VP/O/sorry i/NN/B-NP/O/i am/VBP/B-VP/O/be late/RB/B-ADVP/O/late I/PRP/B-NP/O/i printed/VBP/B-VP/O/print directions/NNS/B-NP/O/direction
#somos/VB/B-VP/O/ser esa/DT/B-NP/O/esa clase/NN/I-NP/O/clase de/IN/B-PP/B-PNP/de gente/NN/B-NP/I-PNP/gente con/IN/B-PP/O/con la/DT/O/O/el que/WP$/O/O/que nuestros/PRP$/B-NP/O/nuestros padres,no/NN/I-NP/O/padres,no quieren/VB/B-VP/O/querer
