# -*- coding: utf-8 -*-
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize as wt
from nltk.stem import WordNetLemmatizer, PorterStemmer
import string 
from collections import Counter
from sklearn.model_selection import train_test_split
import math
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
from sklearn.feature_selection import chi2
import numpy as np
from io import StringIO
lemma = WordNetLemmatizer()
# Function to convert  
def listToString(s): 
    # initialize an empty string
    str1 = "" 
    # traverse in the string  
    for ele in s: 
        str1 += ele  
    # return string  
    return str1 
def listToString1(s): 
    # initialize an empty string
    str1 = "" 
    # traverse in the string  
    for ele in s: 
        str1 += ele +', ' 
    # return string  
    return str1 
gold_csv="Gold-Ingles.csv"
text_csv="English.txt"
gold=pd.read_csv(gold_csv, header=None)
text=pd.read_csv(text_csv, sep="	", header=None)
text.drop(labels=1,axis=1,inplace=True)
gold.columns=["ID_TXT","AGE","GENRE"]
data=text.join(gold["AGE"]).join(gold["GENRE"])
data.columns=["ID_TXT","TXT","AGE","GENRE"]
print(data.head())
count=0
print(data.shape[0])
data.drop(data.index[[int (i) for i in range(200, data.shape[0])]], inplace=True)
progress_divisor = data.shape[0] // 10
words_df={"original":[],"relevant":[]}
cerradas=stopwords.words('english')
Signos = string.punctuation
#LEMAS/STOP WORDS
# for i in data.iterrows():
#     datos=i[1]['TXT']
#     doc=wt(datos)
#     CadLemas=""
#     for token in doc:
#         Lema=lemma.lemmatize(token)
#         if not Lema in cerradas and not Lema[0] in Signos:
#         #print(token+'-'+Lema)
#             CadLemas=CadLemas+Lema.lower()+' '
#     words_df["original"].append(datos.split())
#     words_df["relevant"].append(CadLemas.split())

#Stemming/StopWords
# ps=PorterStemmer()
# for i in data.iterrows():
#     datos=i[1]['TXT']
#     doc=wt(datos)
#     CadLemas=""
#     for token in doc:
#         New_word=ps.stem(token)
#         if not New_word in cerradas and not New_word[0] in Signos:
#         #print(token+'-'+Lema)
#             CadLemas=CadLemas+New_word.lower()+' '
#     words_df["original"].append(datos.split())
#     words_df["relevant"].append(CadLemas.split())
#POS-TAGGING
import spacy
mydict = {"relevant": [],"original":[]}
nlp = spacy.load("en_core_web_sm")
for i in data.iterrows():
    relevante = []
    doc = nlp(i[1]["TXT"])
    for entity in doc.ents:
        relevante.append(entity.text)
    for token in doc:
        interes_word = token.pos_ in ["PROPN","NOUN","ADJ"]
        if token.ent_iob == 2 and interes_word:
            relevante.append(token.lemma_)
    mydict["relevant"].append(relevante)
    mydict["original"].append(entity.text.split())    
words_df=pd.DataFrame.from_dict(mydict)
data=data.join(words_df)
classes = Counter()
clase3 = 0
clase2 = 0
clase1 = 0
total_text = 0
counter_words = Counter()
for i in data.iterrows():
    classes.update([i[1]["AGE"]])
    counter_words.update(i[1]["relevant"])
    total_text += len(i[1]["TXT"])
    if i[1]["AGE"] == "30s":
        clase1 += len(i[1]["original"])
    elif i[1]["AGE"] == "20s":
        clase2 += len(i[1]["original"])
    elif i[1]["AGE"] == "10s":
        clase2 += len(i[1]["original"])
avg_words = (clase1 + clase2 + clase3) / len(data["original"])
total_text /= len(data["TXT"])
print(classes)
print(f"Average chars in text: {total_text}")
print(f"Average words: {avg_words}")
print("Common words")
common_words=counter_words.most_common(20)
print(common_words)
print("Non-common words")
noncommon_words=counter_words.most_common()[-20:]
print(noncommon_words)

split_data=0.3
for i in data.iterrows():
    i[1]['relevant']=', '.join(map(str, i[1]['relevant']))
#Train Data
(train_data, test_data, train_classes, test_classes) = train_test_split(
    data["relevant"],
    data["AGE"],
    test_size=split_data,
    random_state=1,
    stratify=data["AGE"]
)
matrix_threshold={}
analyze_set=set()
word_threshold=50
for i in counter_words:
    if counter_words[i]>word_threshold:
        matrix_threshold[i]=[]
        analyze_set.add(i)  
i = 0
#Frecuencia de terminos 
for row in train_data:
    for word in analyze_set:
        matrix_threshold[word].append(0)
    for i in row.split():
        if i in analyze_set:
            matrix_threshold[i][i] = 1
    i += 1
train_frequency = pd.DataFrame.from_dict(matrix_threshold)
#Frecuencia inversa del documento
N=1
count=0
freq_inv= pd.DataFrame(columns=["WORD","DFT","IDFT"])
for column in train_frequency:
    if train_frequency[column].sum()>0:
        freq_inv=freq_inv.append({"WORD":column,"DFT":train_frequency[column].sum(),"IDFT":math.log(N/train_frequency[column].sum())},ignore_index = True)
#Ponderacion tf-idf
freq_inv['PONDERACION']=freq_inv["DFT"]*freq_inv["IDFT"]
print(freq_inv.head())
fig = plt.figure(figsize=(8,6))
data.groupby('AGE').TXT.count().plot.bar(ylim=0)
plt.show()

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
features = tfidf.fit_transform(data['relevant']).toarray()
labels = data['AGE']

N = 2
data['category_id'] = data['AGE'].factorize()[0]
category_id_df = data[['AGE', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'AGE']].values)
for AGE, category_id in sorted(category_to_id.items()):
  features_chi2 = chi2(features, labels == category_id)
  indices = np.argsort(features_chi2[0])
  feature_names = np.array(tfidf.get_feature_names())[indices]
  unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
  bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
  print("# '{}':".format(AGE))
  print("  . Most correlated unigrams:\n       . {}".format('\n       . '.join(unigrams[-N:])))
  print("  . Most correlated bigrams:\n       . {}".format('\n       . '.join(bigrams[-N:])))

count_vect = CountVectorizer()
print(train_data.head())
X_train_counts = count_vect.fit_transform(train_data)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
clf = MultinomialNB().fit(X_train_tfidf, train_classes)
print(clf.predict(count_vect.transform(["Top quality is the hardest thing to appraise from the description of a electric power inverter. You can go by status or by price tag (i.e. you generally get what you shell out for). Cheap electrical power inverters can be very good worth for cash, but be ready to acquire yet another one at small interval. ; ;Power Inverters change DC power stored in batteries into AC electrical power to operate regular electronic equipments. A electrical power inverter enables you to operate computers, amusement devices, energy resources and kitchen appliances. ; ;Electricity inverters vary in price tag, electricity ranking, effectiveness, waveform and reliability. They generate three sorts of waveforms, square wave, modified sine or sq. wave, and genuine sine wave. The easiest and most affordable is a square wave inverter, but nowadays it is uncommon. Quite a few products will not perform on a sq. wave sign. Genuine sine wave inverters get the job done well, but are high-priced. It is critical to bear in mind that the inexpensive electricity inverters might not be of excellent good quality. ; ;Inexpensive inverters usually generate a square wave sign that has the very same frequency of a modified sq. wave inverter, but sharp edges in spot of sleek curves. They bring about noises in numerous appliances, but they operate fairly great most of the time. Transformer operated appliances really do not like this sort of sign. There is truly a danger of harmful the electricity provide running it from a low-cost electricity inverter. ; ;Cheap inverters are great for mild bulbs, but appliances like variable speed motors (e.g., electric powered drills) and vulnerable electronics (e.g., computer system electric power supplies) usually get harmed. Heaps of inexpensive inverters are offered in the industry these times. Just one way to find these inverters is to study the commercials and particular deals posted by just about every of the important businesses. Then only review the prices from business to firm and select just one. The World wide web is also a place in which you can do all your low-cost inverter research and getting. ; ;REFERENCES: ; ; ; ;http://www.markarticles.com/14281/24/Go-Out-With-Your-Dad-on-Valentine-s-Day.html ; ; ; ; ; ;http://freearticleservice.com/articledetail.php?artid=18813 catid=24 title=Valentine%27s-Date-with-Your-Dad ; ; ; ; ; ;http://articles.onlineweblibrary.com/Art/246818/24/Valentine-s-Date-with-Your-Dad.html ; ; For unexpected emergency electrical power, these kinds of as throughout an outage, electrical power inverters can be employed with an extension cord to plug in electric powered appliances. Electric power inverters are an digital unit that converts direct existing (DC) of 12 volts to alternating current (AC) of one hundred twenty volts for operation of house appliances. ; ;These are commonly utilised for electricity for domestic appliances like electric lights, kitchen appliances, electrical applications, microwaves, radios, and pcs. ; ;With DC AC electrical power inverters, electrical power stored in a battery is transformed to AC electrical energy for utilization in extensive wide variety of apps. Even though switching from immediate latest to large voltage alternating current, many of these transform with a ninety% effectiveness. ; ;These are accessible in different designs different in watts and sizes. The distinct capacity choices need to your major consideration when selecting 1. ; ;As these mechanisms generally presume that the supply battery is in excellent situation and absolutely charged, it is necessary that the engine inside the resource car or truck be started off at minimum the moment an hour for ten-15 minutes to steer clear of discharge. ; ;There are three simple forms: ; ;one. Square wave have become obsolete and are not advised due to their inefficiency, noise and harshness. ; ;2. Modified sine wave are most well-liked and are normally sufficient to operate most units. ; ;3. Correct-sine wave make from the public utility grid technique and are best for delicate electronics. The output sample is equivalent to 60Hz grid electric power. ; ;Figuring out the best to fit and meet your needs appropriately demands a mindful consideration on these factors: ; ;* Would you use it day-to-day or on an occasional foundation? ; ;* How many electrical outlets do you want to plug in? Pick the dimensions that best fits your wants. ; ;* Does the total load exceed 150 watts? If sure, a immediate link (DC) to the mobile requirements to be set up. ; ;* Are you preparing to use hefty-draw gear? If yes, a large-capability device and direct connection is necessary. ; ;- Correct Dimension for Your Needs ; ;Choosing the best or suited dimensions of energy inverters requires addition of the wattage need of all the appliances that will perform at the same time. It is often proposed to think about the greater figure as this ensures that the inverter will supply for all appliances safely and securely. ; ;Relying on the amp-hour ability readily available, the size of time an inverter capabilities can be estimated. ; ;- Sort of Batteries Suggested ; ;Real time for recharging depends on the age and issue of the battery and the need staying positioned on the devices staying operated by the inverter. ; ;For five hundred watts and larger inverters we suggest use of deep cycle (marine or RV) batteries that offer many hundred finish fee/discharge cycles. Commence the engine every single thirty to sixty min's and let it run for 10 min's to recharge. ; ;When DC to AC electrical power inverters are running appliances with substantial constant load ratings for extended periods, it can be a good idea to connect to a stand on your own battery - not the a single in your automobile. ; ;If automobile or truck battery is excessively applied, it's possible that it will drain to a stage when it has inadequate reserve, so use an added deep cycle just one. Deep cycle batteries are able of withstanding repeated drains and have the best reserve rankings. ; ;REFERENCES: ; ; ; ;http://articles.onlineweblibrary.com/Art/287326/24/Go-Out-With-Your-Dad-on-Valentine-s-Day.html ; ; ; ; ; ;http://freearticleservice.com/articledetail.php?artid=19521 catid=24 title=Spending+A+Great+Time+With+Your+Dad+on+Valentine%27s+Day ; ; ; ; ; ;http://www.bestarticledirectoryservice.com/articledetail.php?artid=18813 catid=24 title=Valentine%27s+Date+with+Your+Dad ; ; Inverters are the essential move in between a battery's DC electric power and the AC electrical power needed by regular family electrical methods. In a grid linked household, an inverter connected to a battery bank can offer an uninterruptible supply of backup electric power in the celebration of electrical power failures, or can be employed to sell additional choice electricity electrical power back to the utility company. ; ;A DC to AC power inverter, also termed DC to AC converter, electronically converts DC power from a battery to 60 hertz AC electric power at one hundred twenty volts like in homes. Batteries generate electric power in immediate present (DC) form, which run at incredibly very low voltages but are not able to be employed to operate most modern residence appliances. Inverters consider the DC power supplied by a storage battery bank and electronically transform it to AC power. An inverter used for backup electric power in a grid related residence will use grid electrical power to retain the batteries charged, and when grid electricity fails, it will change to drawing electric power from the batteries and supplying it to the developing electrical system. ; ;Most modern inverters also involve more than voltage and beneath voltage defense, safeguarding delicate devices from unsafe electric power surges as nicely. All the DC to AC energy inverters calls for a twelve-volt enter, but there is a broad array of versions readily available in the industry dependent on the output wattage that they offer. A few of the most commonly employed designs are one hundred fifty watts, 325 watts, 600 watts, 1500 watts and 3000 watts. The decrease wattage, versions can be directly connected to a cars cigarette lighter socket, even though the bigger kinds should be straight wired to greater batteries. ; ;Common purposes for a DC-AC energy inverter involve microwave ovens, televisions, video clip recorders, personal computer and energy instruments and monitoring/communications tools. ; ;REFERENCES: ; ; ; ;http://www.bestarticledirectoryservice.com/articledetail.php?artid=19521 catid=24 title=Spending+A+Great+Time+With+Your+Dad+on+Valentine%27s+Day ; ; ; ; ; ;http://www.markarticles.com/12624/24/Spending-A-Great-Time-With-Your-Dad-on-Valentine-s-Day.html ; ; ; ; ; ;http://www.bestarticledirectoryservice.com/articledetail.php?artid=23223 catid=24 title=Date+Your+Dad+This+Valentine%27s+Day ; ; "])))
models = [
    MultinomialNB(),
    DecisionTreeClassifier(),
    LinearSVC(),
]
CV = 5
cv_df = pd.DataFrame(index=range(CV * len(models)))
entries = []
for model in models:
  model_name = model.__class__.__name__
  accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
  for fold_idx, accuracy in enumerate(accuracies):
    entries.append((model_name, fold_idx, accuracy))
cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])
print(cv_df.groupby('model_name').accuracy.mean())
import seaborn as sns
sns.boxplot(x='model_name', y='accuracy', data=cv_df)
sns.stripplot(x='model_name', y='accuracy', data=cv_df, 
              size=8, jitter=True, edgecolor="gray", linewidth=2)
plt.show()
from sklearn.model_selection import train_test_split
model = LinearSVC()
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, data.index, test_size=split_data, random_state=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
from sklearn.metrics import confusion_matrix

conf_mat = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(8,6))
sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=category_id_df.AGE.values, yticklabels=category_id_df.AGE.values)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
from IPython.display import display

for predicted in category_id_df.category_id:
  for actual in category_id_df.category_id:
    if predicted != actual and conf_mat[actual, predicted] >= 2:
      print("'{}' predicted as '{}' : {} examples.".format(id_to_category[actual], id_to_category[predicted], conf_mat[actual, predicted]))
      print('')

model.fit(features, labels)
from sklearn.feature_selection import chi2
N = 2
for AGE, TXT in sorted(category_to_id.items()):
  indices = np.argsort(model.coef_[category_id])
  feature_names = np.array(tfidf.get_feature_names())[indices]
  unigrams = [v for v in reversed(feature_names) if len(v.split(' ')) == 1][:N]
  bigrams = [v for v in reversed(feature_names) if len(v.split(' ')) == 2][:N]
  print("# '{}':".format(AGE))
  print("  . Top unigrams:\n       . {}".format('\n       . '.join(unigrams)))
  print("  . Top bigrams:\n       . {}".format('\n       . '.join(bigrams)))

texts = ["oamt.org Chemistry lab machines incorporates an assortment of devices this kind of as centrifuges, chemistry analyzers, glassware, hematology analyzers, distillation machines, thermometers, sterilizers, blood fuel analyzers, pipettes, electrolyte analyzers, microscopes, coagulation analyzers, autoclaves, spectrometers, and far more. This machines desires to be of a higher normal and ought to be dependable and earlier mentioned all risk-free. When deciding upon lab devices you need to seek out an accredited maker with know-how in this field so you can be confident that you are acquiring solutions which are analyzed and permitted to undertake the processes that you prepare to undertake. Buying the finest in phrases of high quality does not essentially suggest expensive. There are a range of respected lab equipment producers, importers and stockists presenting a large alternative at aggressive prices. Common consumers of lab equipment contain: Pharmaceutical manufacturing and screening organisations Chemical making organisations Automotive Marketplace Defence and aerospace Marketplace Petroleum sectors Bio-manufacturing These organisations and industries depend upon their gear to provide higher ranges of precision all the way through the testing procedure. Blunders can be costly and in some circumstances catastrophic. With this in thoughts below are the main concerns when purchasing lab machines: Get New Whilst used lab equipment will be sold at an attractive selling price you are far more than very likely unaware of its earlier use. It could have been used and tried to the limits of its capability and whilst it may search in good problem to the naked eye you might find it to be of very low high quality. This is not a purchase you ought to lower corners with, so buy new and buy from permitted resources. Appear for Pace and Precision The whole function of lab testing is to get exact, trusted benefits so look for out equipment that offers the newest technologies and have a established track document for precision. You may possibly fork out additional for newer improvements, but the most up-to-date enhancements in technology will in all probability aid you do the career quicker. Merchandise Assures If you invest in lab devices from an authorized supply you must be coated by a guarantee. But some gear is not necessarily covered for state-of-the-art testing. Make sure you know what you will be utilizing your tools for and examine the warranty handles these types of checks. As a standard rule, when purchasing lab gear you really should usually purchase from a trustworthy resource and exactly where achievable choose a manufacturer with a verified observe report in laboratory testing. You should also make positive that the equipment meets and exceeds EU polices. High quality chemistry lab tools is required to get exact analytical final results in chemistry laboratories. To meet the program requirements of chemistry laboratories in instructional organizations, medical amenities and universities, a lot of well-regarded medical laboratory tools dealers in the US give new as well as recertified products from popular brands at inexpensive prices. New Devices Capabilities Impressive Know-how and Technical specs Chemistry lab gear developed in accordance to prescribed codes and specifications, and showcasing modern technological technical specs would absolutely increase the efficiency of tiny and big chemistry laboratories. These devices let the scientists to easily complete several tests and experiments and obtain analytical final results with bigger accuracy. A completely purposeful chemistry laboratory would require a variety of units this kind of as chemistry analyzers, microscopes, centrifuges, balances and scales, measuring cylinders, blood fuel analyzers, urinalysis analyzers, stage of treatment analyzers, funnels, beakers, hot plates, pipettes, plastic pitchers, and far more. As most health laboratory devices suppliers offer recertified units along with new appliances, exploration facilities now have the comfort to purchase the proper equipment according to their budget. Brand new gadgets are included with innovative technologies to make sure consistency, velocity, accuracy and sturdiness. However these gadgets are expensive, they are expected to carry out several diagnostic purposes flawlessly. bull Assure exceptional performing precision: All new laboratory gear is presented for sale immediately after stringent performance analysis and excellent checks. Consequently, they would be no cost from operational inaccuracies and technical problems. bull Sufficient warranty: Most new chemistry laboratory units come with proper company warranty of 1 to two several years. ",
        "about a year ago i met a girl at a party,i went to this party with a friend of mine and didn't know anyone there.but i saw this attractive,chubby girl who was friends with my friend,and i decided to be bold and get chatty with her.we exchanged numbers and everything,but i never called her.i thought she wouldn't be in to a weird guy like me anyway.fast foward to last week,at a job activity i see a cute chubby girl who i think i recognise,eventually she seemed to recognise me and we were reaquainted,which was nice.the thing is after that,i decided to call her and talk with her,well to make a long story short,she is also a weird girl,much like i am a weird guy.she has a twisted sense of humor like i do,she's smart,speaks perfect englishmchubby,really beautiful face.i'm smitten by her.this is the type of girl that i've always been looking for.and thankfully she's single.hopefully i can change that status.if she's into me as much as i'm into her(and not to sound cocky,but i think she does)then i'l have a story to tell next time.i'm not wasting any time and i sure as hell don't want to be in the freind zone.wish me luck.she seems interested in me,which is more important than you'd think.she makes me feel important,something i havem't felt in a long time.", 
        "Most gorgeous scenery in your everyday living is to try to block your own mountain / hill highest from the few moments, the gorgeous wide ranging check out. Working life is the most wonderful surroundings Nancy Millen Garments This year to wear presently. In an effort to check this out boundless very good landscape, karen millen individuals will continue to climb up. Our life is including cards, they transported to his or her hands less than option. Is a useful one or maybe terrible to be broken. Together with every day life is definitely not no matter whether you have a excellent hands, but try to sit they with their hands.Every day life is to be a pine. ; ;Life is much like high altitude. Sapling possesses its own sounds, as well as quick along with generous, and / or deeply along with slim; pile possesses its own off-road, or simply extra tall imposing, darkish as well as Qingji it's karen millen CM006 a unique destiny, in the event the clockwise undo, your skin with future, we should always know mountain / hill, Best Pistol; knowing woods, conviction. our life is just like a significant place, our company is upon step to relax and play his or her's yan20120213ping function. Sometimes prompted by way of genuine sense of person, seasoned accomplishment as well as failure, fulfillment and even disappointment as soon as i reminisce, to discover every day life is being a Nancy Millen Taffeta Trimmed Trench Topcoat Overcast. Our lives, including short-lived, can come not to mention departed, short lived. Epiphyllum, even though people are right now, karen millen WM002 nonetheless still left to individuals is definitely beautiful. The people reside an eternity, in the event the out of date, but probably viewed in to decide if there is not any beauty allowed to remain the planet earth. Every day life is simply 2 forms: decaying or simply losing. ; ;After school from college or university, as a result of their own individual endeavours to locate a increased amounts of satisfaction function, and also this cardiovascular system can feel calm lots, naturally, a person in the neighborhood.But when Simply put i typed in your house the time, all the tiredness has vanished, karen millen 2011 the full heart and soul of your companion considerably, given that the inhale with corrupted loved ones called my family, toasty all of us, now let my very own cardiovascular system even more of feeling of full satisfaction, greater numbers of feelings of happiness. became popular his coat, cleanse hands and wrists, in the dining table, the mother has become the new daily meals, prior to my family, purchased given us a tumbler water earlier than, says: really don't run to have, for you to , the best tumbler for domestic hot water, hot karen millen black dresses the actual abs, while dining, We procured this type of water, and also slowly sipped the application. sometimes have on his own treasured karen millen apparel New.Mother will not forget about me when necessary claimed, to choose their Karen millen Clothes,the mother will almost allways be with supporting one. Each personal center possesses the Karen Millen Dresses Year 2011, what amount spirit, the stage is usually as great, it could be, you only by natural means enjoy parent caution, though certainly not clumsy focus its ever more graying temples mane; [url=] ;karen millen outlet online ;[/url] probably, you benefit from its unusual information, the actual nearby mall was in fact over the breach enemies uncover any, quite possibly under, at which you might express that this really is competitors, yet or even dreamed about, which would get far more quot;being a tiger. quot; perhaps you are merely to a few of their individual small fraud in addition to fraudulence, naturally deserted, however , gladly bask in painstakingly gained recover the cash ; ;karen millen DL062 ; ; on, potentially, in their your head within the point, you've been some sort of soloist. ; ;The market with a little compound crank away from, since there is an infinite attractive stars concerning the soil, human beings seemed, reproduction, with the nascent years in slowly grown to create a contemporary culture together with the world. Human instinct of the person, can help determine stuffed his or her head he / she were not able to ; ;karen millen DL252 ; ; primarily check out others, he will have wants. On the other hand, yan20120213ping issues will often have a level, individuals continually are now living a bunch simply being, only if their own Karen millen an individual bare, essentially who don't have overnight accommodation in lieu of allowing this to help you many others, the real key will are living indefinitely among the many personal, before die-off on your own. Personal straight to the traditional imperial collection in just one, but if he / she could not become understanding to help choice, ; ;karen millen coats ; ; threshold is a heart, or else running in the particular spirits of your companion, it is likely that this fishing boat the actual overlying liquid. We tend to modern most people a whole lot more as a result. ; ;After virtually all, unfortunately we cannot have the strength connected with pre-loaded, people generally prefer to have a home in the center of the gang, many of us usually pray which will some others good care, then simply do not do this within the soloist, and in addition see their consumers to look at it. Reply on a railing researching back, many of us placed Exactly what have, everything that acquire, what precisely gift does not matter so long as your head though others, you possibly can no regrets connected with; ; ;karen millen online shop ; ; angling Lu looks over, perform go, the style the view look at just what is invisible inside spirits in isn't important, individuals discover them selves this some is usually warrant the heart associated with personal core provides a level above the boogie, and so they should certainly examine just how the place; activity is the amount, varies according to the amount of your own Betty Millen Classcal Dark Primarily Various colored Outfit. "]
text_features = tfidf.transform(texts)
predictions = model.predict(text_features)
for text, predicted in zip(texts, predictions):
  print("  - Predicted as: '{}'".format(predicted))
  print("")
from sklearn import metrics
print(metrics.classification_report(y_test, y_pred, 
                                    target_names=data['AGE'].unique()))

