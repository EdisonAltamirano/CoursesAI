orden en que se ejecutaron los programas.
IMRTANTE: Conservar la estructura de carpetas para que el código no genere error

#Tokenizar texto (para los dos idiomas)
python Tokenizar.py

#Lematizar "Quitar cerradas y signos" (para los dos idiomas)
python3 Lemas.py

#Stemming (para los dos idiomas)
python3 Stemming.py

#Este código genera una lista de términos y su frecuencia en el corpus (carpeta Vocabulario)
python3 ObtenVoc.py Espanol-Categorias
python3 ObtenVoc.py Espanol-Tokens
python3 ObtenVoc.py Ingles-Sttemming
python3 ObtenVoc.py Espanol-Lemas
python3 ObtenVoc.py Ingles-Categorias
python3 ObtenVoc.py Ingles-Tokens
python3 ObtenVoc.py Espanol-Sttemming
python3 ObtenVoc.py Ingles-Lemas

#Genera el formato CSV para cada conjunto de categorías e idioma. Se genera utilizando la frecuencia binaria y el conteo de frecuencia
python3 GeneraCSV.py Espanol-Categorias
python3 GeneraCSV.py Espanol-Tokens
python3 GeneraCSV.py Ingles-Sttemming
python3 GeneraCSV.py Espanol-Lemas
python3 GeneraCSV.py Ingles-Categorias
python3 GeneraCSV.py Ingles-Tokens
python3 GeneraCSV.py Espanol-Sttemming
python3 GeneraCSV.py Ingles-Lemas


