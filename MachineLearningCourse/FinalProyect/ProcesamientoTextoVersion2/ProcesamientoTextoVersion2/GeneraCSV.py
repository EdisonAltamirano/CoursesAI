# -*- coding: utf-8 -*-
import codecs;
import os, sys
import sys, re
import re
import math, cmath

"""    
reload(sys)
sys.setdefaultencoding("utf-8")
"""

def destringifyTupleData(d):
    return [tuple(destringifyList(l)) for l in d]

def destringifyList(l):
    return map(float, l)

#############################################################################
def ObtenListaAtrib(Atributos):
	a=[]
	cad=""
	for l in Atributos.readlines():
		t=l.split(',')
		if len(t[0])>0:
			cad=cad+','+t[0]
			a.append(t[0])
	#print(len(a))
	return a,cad[1:]

#################################################################
def ObtenDocto(docto):
    salida=codecs.open("CSV/"+Entrada+'_Conteo.csv','w')
    salidaBi=codecs.open("CSV/"+Entrada+'_Binario.csv','w')
    salida.write("Id,"+CadAtrib+",Clase\n")
    salidaBi.write("Id,"+CadAtrib+",Clase\n")
    for l in docto.readlines():
        if len(l)>5:
            x=l.split('\t')
            FinClas=x[2].replace('\n','')
            texto=x[1].split(' ')
            cad=str(x[0])+','#Frecuencia
            cad3=str(x[0])+','#Ocurrencia
            Id=x[0]
            vector=InicializaVector()
            for i in texto:
                if i in vector:
                    vector[i]+=1
            for i in range(len(Atrib)):
                cad=cad+str(vector[Atrib[i]])+','
                if vector[Atrib[i]]>=1:
                    cad3=cad3+'1,'
                else:
                    cad3=cad3+'0,'
            cad=cad+FinClas
            cad3=cad3+FinClas
            salida.write(cad+'\n')
            salidaBi.write(cad3+'\n')
    salida.close()
    salidaBi.close()
##########################################################################
def InicializaVector():
    tmp={}
    for i in Atrib:
        tmp[i]=0
    return tmp

#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&	

Entrada=sys.argv[1]
Atrib=[]
CadAtrib=''
#Leer el vocabulario para obtener los atributos
DocCar=codecs.open('Vocabulario/'+Entrada+'.csv','r')
Atrib,CadAtrib=ObtenListaAtrib(DocCar)
print(str(len(Atrib)))
DocCar.close()

DoctoTrain=codecs.open('Documentos/NWH_'+Entrada+'.txt','r')
ObtenDocto(DoctoTrain,)
DoctoTrain.close()
