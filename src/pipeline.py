#!/usr/bin/env python3

"""
This script requires Python 3 and the scikit-learn package. See the README file for more details.
Example invocations:
    Generate the features from the tokenized essays:
        $ python pipeline.py [--train ] [--test] [--preprocessor]

    Run with precomputed features:
        $ python pipeline.py [--train] [--test dev] [--preprocessor] --training_features path/to/train/featurefile --test_features /path/to/test/featurefile
"""
import os
import csv
import argparse
import numpy as np
import pickle
from time import strftime
from features import *
from sklearn import metrics
from sklearn.datasets import dump_svmlight_file, load_svmlight_file
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import Normalizer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC, SVC
from sklearn.pipeline import FeatureUnion
from sklearn.model_selection import LeaveOneOut, KFold
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier


SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
CLASS_LABELS = ['ARA', 'CHI', 'FRE', 'GER', 'HIN', 'ITA', 'JPN', 'KOR', 'SPA', 'TEL', 'TUR', 'X']  # valid labels
RECLASSIFY_LABELS = [('HIN', 'TEL')]  # groups of labels we want to reclassify
PROMPTS = ["P0", "P1", "P2", "P3", "P4", "P5", "P6", "P7"]


def load_features_and_labels(train_partition, test_partition, training_feature_file,
                             test_feature_file, preprocessor='tokenized', vectorizer=None, 
                             feature_outfile_name=None):
    """
    If no feature files are provided, generates feature matrices for training and test data. By default, it assumes the
    use of the text in the 'tokenized' directory (vs 'original'). This is also the directory pointed to in the
    labels files. To use your own processing pipeline, write the processed essays to a new directory under
    "../data/essays/`train_dir`/" and "../data/essays/`test_dir`/", and modify the corresponding labels files with the
    correct "essay_path" column.

    If precomputed feature files are provided, this function reads features and labels for the training and test sets
    instead of creating new ones.

    NOTE: `training_feature_file` and `test_feature_file` must be provided together. If only one is provided, all
          features will be recomputed to avoid dimension mismatch.

    Parameters
    -----------
    train_partition: str
        String indicating the name of the training directory (e.g. 'train'). This directory should
        exist in "../data/essays/" and "../data/labels/". It will also be created in "../data/features/"
        to output the training features

    test_partition: str
        String indicating the name of the testing directory (e.g. 'dev' or 'test'). This directory should
        exist in "../data/essays/" and "../data/labels/". It will also be created in "../data/features/" 
        to output the test features.

    training_feature_file: str
        Path to saved training feature file (must be svm_light format).

    test_feature_file: str
        Path to saved test feature file (must be svm_light format).

    preprocessor: str, 'tokenized' by default
        Name of directory under '../data/essays/<partition_name>/ where the processed essay text is stored.
        Options:
            'original': raw text
            'tokenized': segmented on sentence boundaries (one sentence per line) and word-tokenized (tokens
                                   surrounded by white space).
        HINT: You can use a custom preprocessing pipeline by saving the processed data in
              "../data/essays/<partition>/<custom_preprocessor_name>/" for train and test partitions.

    vectorizer: Vectorizer object or NoneType, None by default
        Object to convert a collection of text documents to a matrix. Must implement fit, transform, and fit_transform
        methods. If no vectorizer is provided, this function will use sklearn's CountVectorizer as default.
    
    feature_outfile_name: str
        Custom name for feature files. The train and test feature files from a given run will be given the same name
        but saved in their respective directories. If no custom name is provided, the file names will be the date and time
        they were generated.

    Returns
    -------
    tuple (length 2)
        -list of [training matrix, training labels as ints, training labels as strings]
        -list of [test matrix, test labels as ints, test labels as strings]

    """
    train_labels_path = "{script_dir}/../data/labels/{train}/labels.{train}.csv".format(train=train_partition, script_dir=SCRIPT_DIR)
    train_data_path = "{script_dir}/../data/essays/{}/tokenized/".format(train_partition, script_dir=SCRIPT_DIR)
    test_labels_path = "{script_dir}/../data/labels/{test}/labels.{test}.csv".format(test=test_partition, script_dir=SCRIPT_DIR)
    test_data_path = "{script_dir}/../data/essays/{}/tokenized".format(test_partition, script_dir=SCRIPT_DIR)

    path_and_descriptor_list = [(train_labels_path, "training labels file"),
                                (train_data_path, "training data directory"),
                                (test_labels_path, "testing labels file"),
                                (test_data_path, "testing data directory")]
    for path_, path_descriptor in path_and_descriptor_list:
        if not os.path.exists(path_):
            raise Exception("Could not find {desc}: {pth}".format(desc=path_descriptor, pth=path_))
    #
    #  Read labels files. If feature files provided, `training_files` and `test_files` below will be ignored
    # 
    with open(train_labels_path) as train_labels_f, open(test_labels_path) as test_labels_f:
        essay_path_train = '{script_dir}/../data/essays/{train}/{preproc}'.format(script_dir=SCRIPT_DIR, train=train_partition, preproc=preprocessor)
        essay_path_test = '{script_dir}/../data/essays/{test}/{preproc}'.format(script_dir=SCRIPT_DIR, test=test_partition, preproc=preprocessor)

        training_files, training_labels, training_prompts = zip(*[(os.path.join(essay_path_train, row['test_taker_id'] + '.txt'), row['L1'], row['essay_prompt'])
                                                for row in csv.DictReader(train_labels_f)])

        test_files, test_labels, test_prompts = zip(*[(os.path.join(essay_path_test, row['test_taker_id'] + '.txt'), row['L1'], row['essay_prompt'])
                                        for row in csv.DictReader(test_labels_f)])
    
    #
    #  Verify that either both or neither of training/test feature files are provided
    #
    if bool(training_feature_file) != bool(test_feature_file):
        print("Feature files were not provided for both test and train partitions. "
              "Generating default unigram features now.")
    
    #
    #  If feature files provided, get features and labels from them
    # 
    elif training_feature_file and test_feature_file:
        training_matrix, encoded_training_labels = load_svmlight_file(training_feature_file)
        original_training_labels = tuple([CLASS_LABELS[int(i)] for i in encoded_training_labels])
        
        if original_training_labels != training_labels:
            raise Exception("Training labels in feature file do not match those in the labels file.")

        test_matrix, encoded_test_labels = load_svmlight_file(test_feature_file)
        original_test_labels = tuple([CLASS_LABELS[int(i)] for i in encoded_test_labels])
        if original_test_labels != test_labels:
            raise Exception("Test labels in feature file do not match those in the labels file.")

        return [(training_matrix, encoded_training_labels, original_training_labels),
                (test_matrix, encoded_test_labels, original_test_labels)]
    
    # 
    #  If no feature files provided, create feature matrix from the data files
    #
    print("Found {} text files in {} and {} in {}"
          .format(len(training_files), train_partition, len(test_files), test_partition))
    print("Loading training and testing data from {} & {}".format(train_partition, test_partition))

    training_data, test_data = [],[]
    for f in training_files:
        with open(f) as doc:
            training_data.append(doc.read())

    for f in test_files:
        with open(f) as doc:
            test_data.append(doc.read())

    features = FeatureUnion([
        #('word_skipgrams', SkipgramVectorizer(n=2, k=2, base_analyzer='word', binary=True, min_df=5)),
        ('char_ngrams', TfidfVectorizer(ngram_range=(1,9), analyzer="char", binary=True))
        #('char_ngrams', TfidfVectorizer(analyzer="char", binary=True))
        #('char_ngrams', TfidfVectorizer(ngram_range=(1,9),analyzer="char", binary=True))
        #('prompt_ngrams', PromptWordVectorizer(ngram_range=(1, 9), analyzer="char", binary=True))
        #('char_ngrams', TfidfVectorizer(analyzer="word", binary=True))
        #('misspellings', MisspellingVectorizer(ngram_range=(1, 9), analyzer="char", binary=True))
        #('ipa_ngrams', IPAVectorizer(ngram_range=(1, 3), analyzer="word", binary=False)),
        #('pos_ngrams', POSVectorizer(ngram_range=(1, 4), analyzer="word")),
        #('average_word_length', AverageWordLength())
        #('final_letter', FinalLetter(analyzer="char")),
        
    ])

    features.fit(training_data)

    training_matrix, encoded_training_labels, vectorizer = transform_data(training_data, training_labels, features)
    test_matrix, encoded_test_labels,  _ = transform_data(test_data, test_labels, features)


    #
    # Write features to feature files
    # No need to have different names for train/test since they each have their own directory.
    outfile_name = (strftime("{}-%Y-%m-%d-%H.%M.%S.features".format(train_partition))
                    if feature_outfile_name is None 
                    else "{}-{}".format(train_partition, feature_outfile_name))

    outfile = strftime("{script_dir}/../data/features/essays/{train}/{outfile_name}"
                       .format(script_dir=SCRIPT_DIR, train=train_partition, outfile_name=outfile_name))
    dump_svmlight_file(training_matrix, encoded_training_labels, outfile)
    print("Wrote training features to", outfile.replace(SCRIPT_DIR, '')[1:])  # prints file path relative to script location
    
    outfile_name = (strftime("{}-%Y-%m-%d-%H.%M.%S.features".format(test_partition))
                    if feature_outfile_name is None
                    else "{}-{}".format(test_partition, feature_outfile_name))
    
    outfile = ("{script_dir}/../data/features/essays/{test}/{outfile_name}"
                .format(script_dir=SCRIPT_DIR, test=test_partition, outfile_name=outfile_name))
    dump_svmlight_file(test_matrix, encoded_test_labels, outfile)
    print("Wrote testing features to", outfile.replace(SCRIPT_DIR, '')[1:])  # prints file path relative to script location

    return [(training_matrix, encoded_training_labels, training_labels, training_prompts, training_files),
            (test_matrix, encoded_test_labels, test_labels, test_prompts, test_files)]


def transform_data(file_list, labels, features):
    """
    This function creates a document-term matrix, given a CSV index file listing the documents and a file dictionary of
    available files.

    If a feature vectorizer has been created, it can be passed in to be used. Otherwise one will be instantiated.
    Different versions of this function could be created to extract other feature types, such as word bigrams, or
    character n-grams, etc.

    Parameters
    ----------
    file_list: list of str
        File names to be used for creating matrix.
    
    labels: list of str
        Correct class labels corresponding with the essays in `file_list`. These will be encoded as integers for saving
        in svm_light format.

    vectorizer: Transformer object.
        Object to convert a collection of text documents to a matrix. Must have fit, transform, and fit_transform
        methods implemented.

    Returns
    -------
    tuple (length 4)
        -doc-term matrix (numpy array), 
        -list of correct labels encoded as ints, 
        -list of labels as strings,
        -transformer instance

    """
    # convert label strings to integers
    labels_encoded = [CLASS_LABELS.index(label) for label in labels]
    doc_term_matrix = features.transform(file_list)

    print("Created a document-term matrix with %d rows and %d columns." 
          % (doc_term_matrix.shape[0], doc_term_matrix.shape[1]))

    return doc_term_matrix.astype(float), labels_encoded, features


def pretty_print_cm(cm, class_labels):
    row_format = "{:>5}" * (len(class_labels) + 1)
    print(row_format.format("", *class_labels))
    for l1, row in zip(class_labels, cm):
        print(row_format.format(l1, *row))


def repredict_labels(clf, predicted, labels, training_matrix, test_matrix, encoded_training_labels):
    print("Repredicting labels between {0}".format(' and '.join(labels)))

    encoded_reclassify_labels = [CLASS_LABELS.index(label) for label in labels]

    retraining_indices = [i for i in range(len(encoded_training_labels)) if
                          encoded_training_labels[i] in encoded_reclassify_labels]
    retest_indices = [i for i in range(len(predicted)) if predicted[i] in encoded_reclassify_labels]

    retraining_docs = training_matrix[retraining_indices]
    retraining_labels = [encoded_training_labels[i] for i in retraining_indices]
    retraining_features = clf.predict_proba(retraining_docs)

    retest_docs = test_matrix[retest_indices]
    retest_features = clf.predict_proba(retest_docs)

    reclf = LinearSVC()
    reclf.fit(retraining_features, retraining_labels)
    repredicted = reclf.predict(retest_features)

    for i in range(len(retest_indices)):
        original_index = retest_indices[i]
        predicted[original_index] = repredicted[i]

    return predicted


def reclassify(clf, predicted, training_matrix, test_matrix, encoded_training_labels):
    print("Retraining the classifier...")

    for labels in RECLASSIFY_LABELS:
        predicted = repredict_labels(clf, predicted, labels, training_matrix, test_matrix, encoded_training_labels)

    return predicted

def train_cross_val(training_matrix, encoded_training_labels):
	
	probas = []

	svm = LinearSVC(multi_class='crammer_singer')
	clf = CalibratedClassifierCV(svm)

	kf = KFold(n_splits=5)
	for train_i, test_i in kf.split(training_matrix):
		print("Train partition: {}\tTest partition: {}".format(train_i, test_i))
		X_train, X_test = training_matrix[train_i], training_matrix[test_i]
		y_train, y_test = np.asarray(encoded_training_labels)[train_i], np.asarray(encoded_training_labels)[test_i]
		clf.fit(X_train, y_train)
		for sample in X_test:
			probas.append(clf.predict_proba(sample)[0])

	return probas

def stacker(train_probas, test_probas, encoded_training_labels):

	svm = LinearSVC(multi_class='crammer_singer')
	clf = CalibratedClassifierCV(svm)

	clf.fit(train_probas, encoded_training_labels)
	predicted = clf.predict(test_probas)

	return predicted


def leave_prompt_out(feature_matrix, labels, prompts, leave_out, include=False):

    X, y, Xout, yout = [], [], [], []

    for i, prompt in enumerate(prompts):
        if prompt != leave_out:
            X.append(feature_matrix[i])
            y.append(labels[i])
        else:
            Xout.append(feature_matrix[i])
            yout.append(labels[i])

    if include: 
        return X, y, Xout, yout
    else: 
        return X, y

def prompt_cross_val(training_features, test_features, training_labels, test_labels, training_prompts, test_prompts, keep_in_dev=True):

    scores = []
    training_features = training_features.toarray()
    test_features = test_features.toarray()

    loo = LeaveOneOut()
    for prompts_in, prompt_out in loo.split(PROMPTS):
        
        if keep_in_dev:
            
            X_train, y_train = leave_prompt_out(training_features, training_labels, training_prompts, PROMPTS[int(prompt_out)])
            X_test, y_test = leave_prompt_out(test_features, test_labels, test_prompts, PROMPTS[int(prompt_out)])

        else:

            features = np.concatenate((training_features, test_features), axis=0)
            labels = training_labels+test_labels
            prompts = training_prompts+test_prompts

            X_train, y_train, X_test, y_test = leave_prompt_out(features, labels, prompts, PROMPTS[int(prompt_out)], include=True)

        print("Performing cross-validation.\n")
        print("{} ommitted.".format(PROMPTS[int(prompt_out)]))

        clf = LinearSVC(multi_class='crammer_singer')
        clf.fit(X_train, y_train)
        predicted = clf.predict(X_test)

        if -1 not in y_test:
            print("\nConfusion Matrix:\n")
            cm = metrics.confusion_matrix(y_test, predicted).tolist()
            pretty_print_cm(cm, CLASS_LABELS)
            print("\nClassification Results:\n")
            print(metrics.classification_report(y_test, predicted, target_names=CLASS_LABELS))
        else:
            print("The test set labels aren't known, cannot print accuracy report.")

        scores.append(clf.score(X_test, y_test))

    print("Mean accuracy for prompt-out cross-validation...:\t {}".format(sum(scores)/len(scores)))

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    
    p.add_argument('--train',
            help='Name of training partition. "train" by default. This should be the name of a directory '
                        'in "../data/essays/" as well as "../data/features/"',
                   default='train')
    
    p.add_argument('--test',
                   help='Name of the testing partition. "dev" by default. This should be the name of a directory '
                        'in "../data/essays/" as well as "../data/features/"',
                   default='dev')
    
    p.add_argument('--preprocessor',
                   help='Name of directory with processed essay files. "tokenized" by default.',
                   default='tokenized')
    
    p.add_argument('--training_features',
                   help='Path to file containing precomputed training features. None by default. '
                        'Should be located in ../data/features/<train_partition_name>/')
    
    p.add_argument('--test_features',
                   help='Path to file containing precomputed test features. None by default.'
                        'Should be located in ../data/features/<test_partition_name>/')
    
    p.add_argument('--feature_outfile_name', 
                   help='Custom name, if desired, for output feature files to be written to '
                        '../data/features/essays/<train_partition_name>/ and '
                        '../data.features/essays/<test_partition_name>. '
                        'If none provided, feature files will be named using the date and time.'
                        'If precomputed feature files are provided, this argument will be ignored.')

    p.add_argument('--predictions_outfile_name', 
                   help='Custom name, if desired, for predictions file to be written to ../predictions/essays/.'
                        'If none provided, predictions file will be names using the date and time.')
    
    args = p.parse_args()

    training_partition_name = args.train
    test_partition_name = args.test
    preprocessor = args.preprocessor
    feature_file_train = args.training_features
    feature_file_test = args.test_features
    feature_outfile_name = args.feature_outfile_name
    predictions_outfile_name = args.predictions_outfile_name

    #
    # Load the training and test features and labels
    #
    training_and_test_data = load_features_and_labels(training_partition_name, test_partition_name, feature_file_train, 
                                                      feature_file_test, feature_outfile_name=feature_outfile_name)
    training_matrix, encoded_training_labels, original_training_labels, training_prompts, training_files = training_and_test_data[0]
    test_matrix, encoded_test_labels, original_test_labels, test_prompts, test_files = training_and_test_data[1]
    
    #
    # Run the classifier
    #

    # Normalize frequencies to unit length
    transformer = Normalizer()
    training_matrix = transformer.fit_transform(training_matrix)
    testing_matrix = transformer.fit_transform(test_matrix)

    # test_dev_X = np.concatenate((training_matrix, testing_matrix))
    # test_dev_y = np.concatenate((encoded_training_labels, encoded_test_labels))

    # with open("test_dev_features", "wb") as f:
    # 	np.save(f, test_dev_X)

    # with open("test_dev_labels", "wb") as j:
    # 	np.save(j, test_dev_y)

    # test_dev_X = np.load("test_dev_features")
    # test_dev_y = np.load("test_dev_labels")

    # Train the model
    # Check the scikit-learn documentation for other models
    print("Training the classifier...")

    #prompt_cross_val(training_matrix, testing_matrix, encoded_training_labels, encoded_test_labels, training_prompts, test_prompts)
    #params = [{'C': [1.0, 5.0, 10.0, 25.0, 50.0, 100.0]}]

    #clf = SVC(kernel='linear',probability=True)
    svm = LinearSVC(multi_class='crammer_singer')
    #clf = BaggingClassifier(LinearSVC(multi_class='crammer_singer'), max_samples=0.5, max_features=0.5)
    #clf = AdaBoostClassifier(LinearSVC(multi_class='crammer_singer'), n_estimators=100, algorithm="SAMME")
    clf = CalibratedClassifierCV(svm)
    #clf = GridSearchCV(estimator=svc, param_grid=params)

    clf.fit(training_matrix, encoded_training_labels)
    predicted = clf.predict(testing_matrix)

    #Reclassify given labels. This uses a stacking approach: a probability disctribution prediction for each label
    #is used as features. Reusing classify for labels that are often confused may be better than adding to
    #RECLASSIFY_LABELS.
    
    #predicted = reclassify(clf, predicted, training_matrix, testing_matrix, encoded_training_labels)
    
    #Write Predictions File
    

    labels_file_path = ('{script_dir}/../data/labels/{test}/labels.{test}.csv'
                        .format(script_dir=SCRIPT_DIR, test=test_partition_name))

    predictions_file_name = (strftime("predictions-%Y-%m-%d-%H.%M.%S.csv") 
                             if predictions_outfile_name is None 
                             else predictions_outfile_name)
    
    probs = {}

    for i, t in enumerate(testing_matrix):
        ps = {}
        preds = clf.predict_proba(t)
        for j, p in enumerate(preds[0]):
            ps[CLASS_LABELS[j]] = p

        probs[test_files[i][-9:-4]] = ps
    	
    with open("test_predictions.pkl", "wb") as f:
    	pickle.dump(probs,f)


    outfile = '{script_dir}/../predictions/essays/{pred_file}'.format(script_dir=SCRIPT_DIR, pred_file=predictions_file_name)
    with open(outfile, 'w+', newline='', encoding='utf8') as output_file:
        file_writer = csv.writer(output_file)
        with open(labels_file_path, encoding='utf-8') as labels_file:
            label_rows = [row for row in csv.reader(labels_file)]
            label_rows[0].append('prediction')
            for i, row in enumerate(label_rows[1:]):
                encoded_prediction = predicted[i]
                prediction = CLASS_LABELS[encoded_prediction]
                row.append(prediction)
        file_writer.writerows(label_rows)

    print("Predictions written to", outfile.replace(SCRIPT_DIR, '')[1:], "(%d lines)" % len(predicted))

    
    #Display classification results
    
    if -1 not in encoded_test_labels:
        print("\nConfusion Matrix:\n")
        cm = metrics.confusion_matrix(encoded_test_labels, predicted).tolist()
        pretty_print_cm(cm, CLASS_LABELS)
        print("\nClassification Results:\n")
        print(metrics.classification_report(encoded_test_labels, predicted, target_names=CLASS_LABELS))
    else:
        print("The test set labels aren't known, cannot print accuracy report.")


    # print("Doing cross-val on train...")
    # train_probas = train_cross_val(training_matrix, encoded_training_labels)
    # test_probas = [clf.predict_proba(x)[0] for x in testing_matrix]
    # print("Training a meta classifier..")
    # predicted = stacker(train_probas, test_probas, encoded_training_labels)

    # if -1 not in encoded_test_labels:
    #     print("\nConfusion Matrix:\n")
    #     cm = metrics.confusion_matrix(encoded_test_labels, predicted).tolist()
    #     pretty_print_cm(cm, CLASS_LABELS)
    #     print("\nClassification Results:\n")
    #     print(metrics.classification_report(encoded_test_labels, predicted, target_names=CLASS_LABELS))
    # else:
    #     print("The test set labels aren't known, cannot print accuracy report.")

