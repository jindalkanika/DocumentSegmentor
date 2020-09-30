import pandas as pd
import glob
import numpy as np
import shutil

file_path = r"..\bbc_train\bbc"

######################TRAINING DATASET PREPARATION#############################

for files in glob.glob(file_path + "/*", recursive=True):

    print(files)

dataset = pd.DataFrame()
list_file = []
type_file = []
content_file = []

for f in glob.glob(file_path + "/*", recursive=True):

    for files in glob.glob(f + "/*" + "." + "txt", recursive=True):
        file = open(files, 'r')
        content = file.read()

        list_file.append(files)
        type_file.append(f.split('\\')[-1])
        content_file.append(content)

dataset['File Names'] = list_file
dataset['Category'] = type_file
dataset['Content'] = content_file
dataset['category_id'] = dataset['Category'].factorize()[0]

category_id_dataset = dataset[['Category', 'category_id']].drop_duplicates().sort_values('category_id')

dataset['Category'].value_counts()

#####################TF-IDF, FEATURE EXTRACTION AND LABEL######################

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(sublinear_tf=True, min_df = 2, norm='l2', encoding='latin-1', 
                        ngram_range=(1, 3), stop_words='english')
features = tfidf.fit_transform(dataset.Content).toarray()
labels = dataset.category_id
features.shape

########################N-GRAMS################################################

from sklearn.feature_selection import chi2

category_to_id = dict(category_id_dataset.values)
id_to_category = dict(category_id_dataset[['category_id', 'Category']].values)
N = 5
for category, category_id in sorted(category_to_id.items()):
  features_chi2 = chi2(features, labels == category_id)
  indices = np.argsort(features_chi2[0])
  feature_names = np.array(tfidf.get_feature_names())[indices]
  unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
  bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
  trigrams = [v for v in feature_names if len(v.split(' ')) == 3]
  print("# '{}':".format(category))
  print("  . Most correlated unigrams:\n       . {}".format('\n       . '.join(unigrams[-N:])))
  print("  . Most correlated bigrams:\n       . {}".format('\n       . '.join(bigrams[-N:])))
  print("  . Most correlated trigrams:\n       . {}".format('\n       . '.join(trigrams[-N:])))
  
###########################MULTIONOMIAL NAIVE BAYES MODEL TRAINING#############

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()

X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, 
                                                                                 dataset.index, 
                                                                                 test_size=2, 
                                                                                 random_state=1)
model.fit(X_train, y_train)



#################################RANDOM FOREST CLASSIFIER TRAINING#############

#from sklearn.model_selection import train_test_split
#from sklearn.ensemble import RandomForestClassifier

#model = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=1)

#X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, 
                                                                                 dataset.index, 
                                                                                 test_size=2, 
                                                                                 random_state=1)
#model.fit(X_train, y_train)

##############################TEST DATASET PREPARATION#########################
  
test_file = r"..\bbc_test"
#destination_path = r"..\bbc_sorted"

dataset_test = pd.DataFrame()
list_file_test = []
content_file_test = []

for files in glob.glob(test_file + "/*" + "." + "txt", recursive=True):
    file = open(files, 'r')
    content = file.read()

    list_file_test.append(files)
    content_file_test.append(content)

dataset_test['File Names'] = list_file_test
dataset_test['Content'] = content_file_test

#category_id_dataset = dataset[['Category', 'category_id']].drop_duplicates().sort_values('category_id')

test_features = tfidf.transform(dataset_test.Content).toarray()
predictions = model.predict(test_features)
    
prediction_list = []
for predicted in (predictions):
  type_file = id_to_category[predicted]
  prediction_list.append(type_file)

dataset_test['PredictedType'] = prediction_list

####################FOLDER CREATION AND DOCUMENT SEGMENTATION##################

required_folders = dataset_test['PredictedType'].unique().tolist()
import os
for i in required_folders:
# define the name of the directory to be created
    path = "..\\bbc_sorted_2\\" + i
    try:
        os.makedirs(path)
    except OSError:
        print ("Creation of the directory %s failed" % path)
    else:
        print ("Successfully created the directory %s" % path)

for i in range(len(dataset_test)):
    print (i)
    source = dataset_test['File Names'][i]
    destination =  "..\\bbc_sorted_2\\" + "\\" + dataset_test['PredictedType'][i] 
    shutil.copy(source, destination)
    
##############################FINAL DATASET EXTRACTION#########################
    
OriginalType = []
for i in dataset_test['File Names']:
    print(i)
    file_name = i.split('\\')[-1]
    file_type = file_name.split('(')[0]
    OriginalType.append(file_type)

dataset_test['OriginalType'] = OriginalType

dataset_test.to_excel("output_document_segmentation_multinomialNB.xlsx")
