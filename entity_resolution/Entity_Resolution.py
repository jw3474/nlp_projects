import json
import csv
import math
import sys
import numpy as np
from sklearn.ensemble import RandomForestClassifier

"""
Below function loads two datasets that describe the same entities, 
and identifies which entity in one dataset is the same as an entity in the other dataset.

Datasets were provided by Foursquare and Locu, 
and contain descriptive information about various venues such as venue names and phone numbers.

I used random forest classifier to match these entities and evaluated the model with F-score, precision, and recall.
"""

def get_matches(locu_train_path, foursquare_train_path, matches_train_path, locu_test_path, foursquare_test_path):

    """
    This function receive a file path of a json file and load it as a dictionary.
    """
    def load_json(file_path):
        json_objects = json.load(open(file_path))
        result = {}
        for i in json_objects:
            result[i["id"]] = i
        return result

    """
    This function receive a dictionary file and clean the "phone" part of this dictionary.
    It will remove special symbols and spaces and make sure all the phone numbers are ten digits numbers.
    """
    def clean_phone_number(dict_file):
        for key, value in dict_file.items():
            if dict_file[key]['phone'] == None:
                dict_file[key]['phone'] = ""
            if "(" in dict_file[key]['phone']:
                dict_file[key]['phone'] = dict_file[key]['phone'].replace("(", "")
            if ")" in dict_file[key]['phone']:
                dict_file[key]['phone'] = dict_file[key]['phone'].replace(")", "")
            if "-" in dict_file[key]['phone']:
                dict_file[key]['phone'] = dict_file[key]['phone'].replace("-", "")
            dict_file[key]['phone'] = dict_file[key]['phone'].replace(" ", "")
            if dict_file[key]['phone'] == "":
                dict_file[key]['phone'] = None
        return dict_file
    
    """
    This function receive a csv file path and load the matches as list/dictionary.
    """
    def read_matches(file_path):
        matches = {}
        with open(file_path) as csvfile:
            f = csv.reader(csvfile)
            next(f)
            for line in f:
                (locu_id, four_id) = (line[0], line[1])
                matches[(locu_id, four_id)] = 1 
        return matches
    
    """
    This function receives a string and return all possible n-gram representations of that string in a set.
    """
    def n_gram_split(string, n_gram):
        result = set()
        if len(string) < n_gram:
            result.add(string)
            return result
        for i in range(len(string) - n_gram + 1):
            result.add(string[i:(i+n_gram)])
        return result
    
    
    """
    This function calcuate the Jaccard distance of the shared four-grams characters in a field between two entities.
    The field can be 'name' or 'street_address'.
    """
    def n_gram_jaccard_score(entity1,entity2,field, n_gram = 4):
        name1 = " ".join([entity1[x] for x in field])
        name2 = " ".join([entity2[x] for x in field])

        if name1 == "":
            set1 = set()
        else:
            set1 = set.union(*[n_gram_split(x, n_gram) for x in name1.lower().split()])

        if name2 == "":
            set2 = set()
        else:
            set2 = set.union(*[n_gram_split(x, n_gram) for x in name2.lower().split()])

        c = set1.intersection(set2)    
        denominator = (len(set1) + len(set2) - len(c))

        if denominator == 0:
            return 0
        else:
            return float(len(c))/denominator
    
    """
    This function calculates the geometric distance between two entities 
    measured by Equirectangular approximation of their latitudes and longitudes.
    """
    def geo_distance(entity1, entity2):
        latitude1 = entity1["latitude"]
        longitude1 = entity1["longitude"]
        latitude2 = entity2["latitude"]
        longitude2 = entity2["longitude"]

        if latitude1 is None or longitude1 is None:
            return 10

        var_x = (longitude2 - longitude1) * math.cos((latitude1 + latitude2)/2)
        var_y = latitude2 - latitude1
        var_d = math.sqrt(var_x*var_x + var_y*var_y)
        return var_d
    
    """
    This function calculates the Jaccard distance of the words in a field between two entities.
    The field can be 'name' or 'street_address'.
    """
    def jaccard_score(entity1,entity2,field):
        name1 = " ".join([entity1[x] for x in field])
        name2 = " ".join([entity2[x] for x in field])
        if name1 == "":
            set1 = set()
        else:
            set1 = set(name1.lower().split())
        if name2 == "":
            set2 = set()
        else:
            set2 = set(name2.lower().split())
        c = set1.intersection(set2)
        denominator = (len(set1) + len(set2) - len(c))
        return 0 if denominator == 0 else float(len(c)) / denominator
    
    """
    This function returns binary indicates of whether an attribute is None.
    """
    def is_none(entry):
        if entry is None or entry == "":
            return 1
        else:
            return 0
    
    """
    This function returns binary indicates of whether two attributes are equal.
    """
    def is_equal(entry1, entry2):
        if entry1 is None or entry2 is None or entry1 == "" or entry2 == "":
            return 0
        return 1 if entry1 == entry2 else 0

    """
    This function a pair of entities and return their features in a list.
    """
    def features(entity1, entity2):
        return [n_gram_jaccard_score(entity1, entity2, ["name"]), \
                n_gram_jaccard_score(entity1, entity2, ["street_address"]), \
                jaccard_score(entity1, entity2, ["name"]), \
                jaccard_score(entity1, entity2, ["street_address"]), \
                geo_distance(entity1, entity2), \
                is_none(entity1["street_address"]), \
                is_none(entity2["street_address"]), \
                is_none(entity1["phone"]), \
                is_none(entity2["phone"]), \
                is_equal(entity1["postal_code"], entity2["postal_code"]), \
                is_equal(entity1["locality"], entity2["locality"]), \
                is_equal(entity1['phone'], entity2['phone']), \
                is_equal(entity1['name'], entity2['name'])]
    
    """
    This function returns labels of each pair of entities.
    True means the pair is a match. False means the pair is not a match.
    """
    def get_y(indexes, matches):
        y = []
        for (key1, key2) in indexes:
            if (key1, key2) in list(matches_train.keys()):
                y.append(True)
            else:
                y.append(False)
        return y
    
    """
    This function returns the precision, recall, and f-score after prediciton.
    """
    def score_matches(y_pred, matches, indexes):
        fp = 0
        tp = 0

        for i in range(len(indexes)):
            for k in set(matches.keys()):     
                if k == indexes[i]:
                    if y_pred[i] == True:
                        tp += 1

        total_positive = 0
        for i in range(len(y_pred)):
            if y_pred[i] == True:
                total_positive += 1

        fp = total_positive - tp

        precision = tp / float(tp + fp)
        recall = tp / float(len(matches))
        f_score = (2.0 * precision * recall) / (precision + recall)
        return tp, fp, precision, recall, f_score
    
    """
    This function return the matches_test.csv file after prediction.
    """
    def write_matches(y_pred_test, indexes_test, file_name = "matches_test.csv"):
        with open(file_name, 'w') as out:
            out.write("locu_id,foursquare_id\n")
            for i in range(len(y_pred_test)):
                if y_pred_test[i] == True: 
                    out.write("%s,%s\n" % indexes_test[i]) 
        return None
    
    #sys.stderr.write( "Loading files..." )
    
    print("Loading files...")
    locu_train = load_json(locu_train_path)
    foursquare_train = load_json(foursquare_train_path)
    locu_test = load_json(locu_test_path)
    foursquare_test = load_json(foursquare_test_path)
    matches_train = read_matches(matches_train_path)
    print("Done...")
    
    print("\n")
    
    print("Cleaning data...")
    locu_train = clean_phone_number(locu_train)
    foursquare_train = clean_phone_number(foursquare_train)
    locu_test = clean_phone_number(locu_test)
    foursquare_test = clean_phone_number(foursquare_test)
    print("Done...")
    
    print("\n")
    
    print("Building training data: X_train...")
    indexes_train = []
    X_train = []
    for k1, v1 in locu_train.items():
        for k2, v2 in foursquare_train.items():
            indexes_train.append((k1, k2))
            X_train.append(features(v1, v2))
    print("Done...")
    
    print("\n")
    
    print("Building testing data: X_test...")
    indexes_test = []
    X_test = []
    for k1, v1 in locu_test.items():
        for k2, v2 in foursquare_test.items():
            indexes_test.append((k1, k2))
            X_test.append(features(v1, v2))
    print("Done...")
    
    print("\n")
    
    print("Building training data: y_train...")
    y_train = get_y(indexes_train, matches_train)
    print("Done...")
    
    print("\n")
    
    print("Fitting Random Forest Classifier with X_train and y_train...")
    classifier = RandomForestClassifier(n_estimators = 100).fit(X_train, y_train)
    print("Done...")
    
    print("\n")
    
    print("Predicting...")
    y_pred_train = classifier.predict(X_train)
    y_pred_test = classifier.predict(X_test)
    
    y_prob_train = classifier.predict_proba(X_train)
    y_prob_test = classifier.predict_proba(X_test)
    print("Done...")
    
    print("\n")
    
    print("Calculating scores...")
    tp, fp, precision, recall, f_score = score_matches(y_pred_train, matches_train, indexes_train)
    print("Done...")
    
    print("\n")
    
    print("Writing matches_test.csv file...")
    write_matches(y_pred_test, indexes_test, file_name = "matches_test.csv")
    print("Done...")
    
    print("\n")
    
#     feature_importances = classifier.feature_importances_
#     print(feature_importances)

    print("True Positive =",tp,"False Positive =",fp,"Precision =",precision,"Recall =",recall,"F-score =",f_score)
    return tp, fp, precision, recall, f_score

locu_train_path = './train/locu_train.json'
foursquare_train_path = './train/foursquare_train.json'
locu_test_path = './test/locu_test.json'
foursquare_test_path = './test/foursquare_test.json'
matches_train_path = './train/matches_train.csv'

tp, fp, precision, recall, f_score = get_matches(locu_train_path, foursquare_train_path, matches_train_path, locu_test_path, foursquare_test_path)

