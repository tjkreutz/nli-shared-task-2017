#!/usr/bin/env python3

"""
This script requires Python 3 and the scikit-learn package. See the README file for more details.
Example invocations:
    Generate the features from the tokenized essays:
        $ python essay_baseline.py [--train ] [--test] [--preprocessor]

    Run with precomputed features:
        $ python essay_baseline.py [--train] [--test dev] [--preprocessor] --training_features path/to/train/featurefile --test_features /path/to/test/featurefile
"""
import argparse
import csv
import os
from time import strftime
from sklearn import metrics
from sklearn.datasets import dump_svmlight_file, load_svmlight_file
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import Normalizer
from sklearn.svm import LinearSVC

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
CLASS_LABELS = ['ARA', 'CHI', 'FRE', 'GER', 'HIN', 'ITA', 'JPN', 'KOR', 'SPA', 'TEL', 'TUR']  # valid labels


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

        training_files, training_labels = zip(*[(os.path.join(essay_path_train, row['test_taker_id'] + '.txt'), row['L1'])
                                                for row in csv.DictReader(train_labels_f)])

        test_files, test_labels = zip(*[(os.path.join(essay_path_test, row['test_taker_id'] + '.txt'), row['L1'])
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

    training_matrix, encoded_training_labels, vectorizer = load_unigrams(training_files,
                                                                         training_labels,
                                                                         vectorizer)
    test_matrix, encoded_test_labels,  _ = load_unigrams(test_files, test_labels, vectorizer)

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

    return [(training_matrix, encoded_training_labels, training_labels),
            (test_matrix, encoded_test_labels, test_labels)]


def load_unigrams(file_list, labels, vectorizer=None):
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

    vectorizer: Vectorizer object or NoneType, None by default.
        Object to convert a collection of text documents to a matrix. Must have fit, transform, and fit_transform
        methods implemented. If no vectorizer is provided, this function will use sklearn's CountVectorizer. The
        vectorizer that is fit on the training data should be re-used for the testing data.

    Returns
    -------
    tuple (length 4)
        -doc-term matrix (numpy array), 
        -list of correct labels encoded as ints, 
        -list of labels as strings,
        -vectorizer instance

    """
    # convert label strings to integers
    labels_encoded = [CLASS_LABELS.index(label) for label in labels]
    if vectorizer is None:
        vectorizer = CountVectorizer(input="filename")  # create a new one
        doc_term_matrix = vectorizer.fit_transform(file_list)
    else:
        doc_term_matrix = vectorizer.transform(file_list)

    print("Created a document-term matrix with %d rows and %d columns." 
          % (doc_term_matrix.shape[0], doc_term_matrix.shape[1]))

    return doc_term_matrix.astype(float), labels_encoded, vectorizer


def pretty_print_cm(cm, class_labels):
    row_format = "{:>5}" * (len(class_labels) + 1)
    print(row_format.format("", *class_labels))
    for l1, row in zip(class_labels, cm):
        print(row_format.format(l1, *row))


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
    training_matrix, encoded_training_labels, original_training_labels = training_and_test_data[0]
    test_matrix, encoded_test_labels, original_test_labels = training_and_test_data[1]
    
    #
    # Run the classifier
    #

    # Normalize frequencies to unit length
    transformer = Normalizer()
    training_matrix = transformer.fit_transform(training_matrix)
    testing_matrix = transformer.fit_transform(test_matrix)

    # Train the model
    # Check the scikit-learn documentation for other models
    print("Training the classifier...")
    clf = LinearSVC()
    clf.fit(training_matrix, encoded_training_labels)  # Linear kernel SVM
    predicted = clf.predict(testing_matrix)
    
    #
    # Write Predictions File
    #

    labels_file_path = ('{script_dir}/../data/labels/{test}/labels.{test}.csv'
                        .format(script_dir=SCRIPT_DIR, test=test_partition_name))

    predictions_file_name = (strftime("predictions-%Y-%m-%d-%H.%M.%S.csv") 
                             if predictions_outfile_name is None 
                             else predictions_outfile_name)

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

    #
    # Display classification results
    #
    if -1 not in encoded_test_labels:
        print("\nConfusion Matrix:\n")
        cm = metrics.confusion_matrix(encoded_test_labels, predicted).tolist()
        pretty_print_cm(cm, CLASS_LABELS)
        print("\nClassification Results:\n")
        print(metrics.classification_report(encoded_test_labels, predicted, target_names=CLASS_LABELS))
    else:
        print("The test set labels aren't known, cannot print accuracy report.")