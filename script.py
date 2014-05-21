#!/usr/bin/python
# -*- coding: utf-8 -*-

#########################################################################
############## Semeval - Aspect Based Sentiment Analysis ################
#########################################################################

#Author: Pedro Paulo Balage Filho
#Date: 19/03/2013
#Version: 1.0

# Python 3 compatibility
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import with_statement
from __future__ import unicode_literals

# imports
from libraries import baselines
from libraries.baselines import Corpus,Evaluate  # Distributed by the Semeval organizers (baselines and read corpora functions)
import xml.etree.ElementTree as ET, getopt, logging, sys, random, re, copy
from subprocess import Popen, PIPE
from pprint import pprint
import re
import os
import pickle

pwd = os.getcwd()
print('Loading modules and functions...')

#Semantic frames semafor parser
def run_semaphor(corpus,identifier):
    try:
        from libraries.semaphore import mysemaphore
    except:
        print('Semafor need to be installed in the machine in order to parse the semantic frames')
        print('Check directories.py and Semafor config file parameters')

    path = pwd + '/semaphor_files/semaphor_' + identifier + '.txt'
    sents = '\n'.join(corpus.texts)
    sents = sents.encode("ascii","ignore") # The current api only handle ascii text
    frames = mysemaphore(sents,path)
    # semaphore python library changes the chdir to /libraries. Revert into
    # previous one
    os.chdir(pwd)
    return frames

# Excecute Senna
def run_senna(sents,identifier):

    senna_cmd = ['senna', '-path', '/opt/senna/', '-usrtokens']

    sents = '\n'.join(sents) + '\n'
    sents = sents.encode("ascii","ignore") # The current api only handle ascii text
    # Run the tagger and get the output
    p = Popen(senna_cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE)
    (stdout, stderr) = p.communicate(input=sents)
    senna_output = stdout

    open('senna_files/senna_'+identifier+'.txt','w').write(senna_output)

    # Check the return code.
    if p.returncode != 0:
        print ('Senna command failed! Details: %s\n%s' % (stderr,senna_output))
        return None

    sentences = list()
    sent = list()

    lines = senna_output.split('\n')
    for line in lines:
        values = re.split(r'[ \t]',line)
        values = [t for t in values if len(t) != 0]
        if len(values) == 0:
            sentences.append(sent)
            sent = list()
            continue
        try:
            senna_conll = dict()
            senna_conll['word'] = values[0]
            senna_conll['pos'] = values[1]
            senna_conll['chunk'] = values[2]
            senna_conll['ne'] = values[3]
            senna_conll['srl'] = values[4:-1]
            senna_conll['tree'] = values[-1]
            sent.append(senna_conll)
        except:
            print ('Error reading senna output line: ' + line)

    return sentences

# load corpora from a pickle file or process everything in senna and semafor. #
# You dont need to chenge anything here if you already have the corpora.pkl
# file
def load_corpora():
    corpora = dict()

    if os.path.exists('corpora.pkl'):
        corpora = pickle.load(open('corpora.pkl','rb'))
    else:
        corpora['restaurants'] = dict()
        corpora['laptop'] = dict()

        # the trainset is composed by train and trial dataset
        train_filename = 'semeval_data/Restaurants_Train_v2.xml'
        trial_filename = 'semeval_data/restaurants-trial.xml'
        corpus = Corpus(ET.parse(train_filename).getroot().findall('sentence') + ET.parse(trial_filename).getroot().findall('sentence'))
        identifier = 'restaurants_train'
        corpora['restaurants']['trainset'] = dict()
        corpora['restaurants']['trainset']['corpus'] = corpus
        frames = run_semaphor(corpus,identifier)
        corpora['restaurants']['trainset']['semaphor'] = frames
        tokenized_sents = [frame['text'] for frame in frames]
        corpora['restaurants']['trainset']['senna'] = run_senna(tokenized_sents,identifier)

        # for testing and debuging I was using the trial dataset
        '''
        corpus_filename = 'semeval_data/restaurants-trial.xml'
        corpus = Corpus(ET.parse(corpus_filename).getroot().findall('sentence'))
        identifier = 'restaurants_trial'
        corpora['restaurants']['trialset'] = dict()
        corpora['restaurants']['trialset']['corpus'] = corpus
        frames = run_semaphor(corpus, identifier)
        corpora['restaurants']['trialset']['semaphor'] = frames
        tokenized_sents = [frame['text'] for frame in frames]
        corpora['restaurants']['trialset']['senna'] = run_senna(tokenized_sents,identifier)
        '''

        # Testset provided by SemEval
        corpus_filename = 'semeval_data/Restaurants_Test_Data_PhaseA.xml'
        corpus = Corpus(ET.parse(corpus_filename).getroot().findall('sentence'))
        identifier = 'restaurants_test'
        corpora['restaurants']['testset'] = dict()
        corpora['restaurants']['testset']['corpus'] = corpus
        frames = run_semaphor(corpus, identifier)
        corpora['restaurants']['testset']['semaphor'] = frames
        tokenized_sents = [frame['text'] for frame in frames]
        corpora['restaurants']['testset']['senna'] = run_senna(tokenized_sents,identifier)

        # the trainset is composed by train and trial dataset
        train_filename = 'semeval_data/Laptop_Train_v2.xml'
        trial_filename = 'semeval_data/laptops-trial.xml'
        corpus = Corpus(ET.parse(train_filename).getroot().findall('sentence') + ET.parse(trial_filename).getroot().findall('sentence'))
        identifier = 'laptop_train'
        corpora['laptop']['trainset'] = dict()
        corpora['laptop']['trainset']['corpus'] = corpus
        frames = run_semaphor(corpus, identifier)
        corpora['laptop']['trainset']['semaphor'] = frames
        tokenized_sents = [frame['text'] for frame in frames]
        corpora['laptop']['trainset']['senna'] = run_senna(tokenized_sents,identifier)

        # for testing and debuging I was using the trial dataset
        '''
        corpus_filename = 'semeval_data/laptops-trial.xml'
        identifier = 'laptop_trial'
        corpus = Corpus(ET.parse(corpus_filename).getroot().findall('sentence'))
        corpora['laptop']['trialset'] = dict()
        corpora['laptop']['trialset']['corpus'] = corpus
        frames = run_semaphor(corpus,identifier)
        corpora['laptop']['trialset']['semaphor'] = frames
        tokenized_sents = [frame['text'] for frame in frames]
        corpora['laptop']['trialset']['senna'] = run_senna(tokenized_sents,identifier)
        '''

        # Testset provided by SemEval
        corpus_filename = 'semeval_data/Laptops_Test_Data_PhaseA.xml'
        identifier = 'laptop_test'
        corpus = Corpus(ET.parse(corpus_filename).getroot().findall('sentence'))
        corpora['laptop']['testset'] = dict()
        corpora['laptop']['testset']['corpus'] = corpus
        frames = run_semaphor(corpus,identifier)
        corpora['laptop']['testset']['semaphor'] = frames
        tokenized_sents = [frame['text'] for frame in frames]
        corpora['laptop']['testset']['senna'] = run_senna(tokenized_sents,identifier)

        pickle.dump(corpora, open('corpora.pkl', 'wb'))

    return corpora

# Function to save the working dataset into a file in CONLL format
def save_conll(path, dataset, gold=True):

    corpus = dataset['corpus'].corpus
    senna = dataset['senna']
    semaphor = dataset['semaphor']

    fp = open(path,'w')

    # for each text in corpus
    for i in range(len(corpus)):
        line = ''

        # map the aspects tokens into tokenized senna text
        text = senna[i]
        tokens = [token['word'] for token in text]
        aspects = ['False' for token in text]
        aspect_terms = [t.split() for t in corpus[i].get_aspect_terms()]
        for term in aspect_terms:
            # Is it a unigram?
            if len(term) == 1:
                # In which position(s) this unigram can be found. Sometimes the
                # tokenizer joint the ' with the token. I am ignoring it
                for i in [i for i,token in enumerate(tokens) if token == term[0] or token.replace("'","") == term[0]]:
                    # tag the token as aspect
                    aspects[i]= 'True'
            else:
                # This is a n-gram
                # In which position(s) this n-gram start
                for i in [i for i,token in enumerate(tokens) if token.lower() == term[0]]:
                    # The text n-gram is the same as the aspect n-gram
                    if term == tokens[i:i+len(term)]:
                        # tag all the tokens in the ngram as aspect
                        for j in range(len(term)):
                            aspects[i+j] = 'True'


        srl = ['O'  for token in text]
        for index, senna_conll in enumerate(text):
            # Atribute the value for the first column in senna which has a
            # value
            for role in senna_conll['srl']:
                if role != 'O':
                    srl[index] = role

        # semantic frames
        key_concept_list = []
        frames = semaphor[i]['fn-labels']
        for concept in frames.keys():
            if isinstance(frames[concept],dict):
                for subconcept in frames[concept].keys():
                    if isinstance(frames[concept][subconcept],dict):
                        for subsubconcept in frames[concept][subconcept].keys():
                            if isinstance(frames[concept][subconcept][subsubconcept],str):
                                key_concept_list.append( (frames[concept][subconcept][subsubconcept],concept) )
                    elif isinstance(frames[concept][subconcept],str):
                        key_concept_list.append( (frames[concept][subconcept],concept) )
            elif isinstance(frames[concept],str):
                key_concept_list.append( (frames[concept][subconcept],concept) )

        # map the semantic frames into tokenized senna text
        tokens = [token['word'] for token in text]
        target_frames = ['O' for token in text]
        aspect_terms = [t.split() for t in corpus[i].get_aspect_terms()]
        for term,concept in key_concept_list:
            term = term.split()
            # Is it a unigram?
            if len(term) == 1:
                # In which position(s) this unigram can be found. Sometimes the
                # tokenizer joint the ' with the token. I am ignoring it
                for i in [i for i,token in enumerate(tokens) if token == term[0] or token.replace("'","") == term[0]]:
                    # tag the token as aspect
                    target_frames[i]= concept
            else:
                # This is a n-gram
                # In which position(s) this n-gram start
                for i in [i for i,token in enumerate(tokens) if token.lower() == term[0]]:
                    # The text n-gram is the same as the aspect n-gram
                    if term == tokens[i:i+len(term)]:
                        # tag all the tokens in the ngram as aspect
                        for j in range(len(term)):
                            target_frames[i+j] = concept

        # write in CONLL format (One feature per column)
        for index, senna_conll in enumerate(text):
            line += senna_conll['word'] + '\t'
            line += senna_conll['pos'] + '\t'
            line += senna_conll['chunk'] + '\t'
            line += senna_conll['ne'] + '\t'
            line += srl[index] + '\t'
            line += target_frames[index]
            if gold:
                line += '\t' + aspects[index] + '\n'
            else:
                line += '\n'

        line += '\n'
        fp.write(line)
    fp.close()

# Retrieve the aspects from CRF predictions. The last column in CRF output has
# the information if the word is an aspect (True) or not (FALSE).
def retrieve_aspects(predictions):
    # predictions in conll format
    lines = predictions.split('\n')
    sentences = list()
    aspects = list()
    last_aspect_line = -2
    for lineno, line in enumerate(lines):
        values = re.split(r'[ \t]',line)
        values = [t for t in values if len(t) != 0]
        # Empty line is a new text
        if len(values) == 0:
            sentences.append(aspects)
            aspects = list()
        else:
            if values[-1] == 'True':
                if last_aspect_line+1 == lineno:
                    aspects[-1] = aspects[-1] + ' ' + values[0]
                else:
                    aspects.append(values[0])
                last_aspect_line = lineno
    return sentences

# Funcition to train a CRF. It need the CRF++ installed
# http://crfpp.googlecode.com/
def train_crfpp(crf_learn_path, crf_params, template_path, model_path, trainset):

    train_file = 'crf/train.data'

    save_conll(train_file,trainset,gold=True)

    crf_cmd = [crf_learn_path, crf_params, template_path, train_file, model_path]

    # Run the tagger and get the output
    p = Popen(crf_cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE)
    (stdout, stderr) = p.communicate()

    #os.remove('train.data')

    # Check the return code.
    if p.returncode != 0:
        print ('crf_learn command failed! Details: %s\n%s' % (stderr,stdout))
        return None

# Funcition to use a CRF. It need the CRF++ installed
# http://crfpp.googlecode.com/
def test_crfpp(crf_test_path, model_path, testset):


    test_file_goldstandard = 'crf/test.data.gold'
    test_file = 'crf/test.data'
    predictions_file = 'crf/test.output'

    save_conll(test_file_goldstandard,testset,gold=True)
    save_conll(test_file,testset,gold=False)

    crf_cmd = [crf_test_path, '-m', model_path,test_file]

    # Run the tagger and get the output
    p = Popen(crf_cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE)
    (stdout, stderr) = p.communicate()

    predictions = stdout

    # Check the return code.
    if p.returncode != 0:
        print ('crf_learn command failed! Details: %s\n%s' % (stderr,stdout))
        return None

    # write predictions in the file
    open(predictions_file,'w').write(predictions)

    return retrieve_aspects(predictions)


# Function to generate the XML in the format asked for SemEval organizers
def tag(BaselineAspectExtractor,test_instances,aspect_list):

    clones = []

    for index,i in enumerate(test_instances):
        i_ = copy.deepcopy(i)
        i_.aspect_terms = []
        for c in aspect_list[index]:
            if c in i_.text:
                offsets = BaselineAspectExtractor.find_offsets(c, i.text)
                for start, end in offsets: i_.add_aspect_term(term=c,
                                                                offsets={'from': str(start), 'to': str(end)})
        clones.append(i_)
    return clones

# Function to call the aspect extraction task
def AspectExtraction(trainset,testset):
    print('Aspect Extraction:\n')

    b1 = baselines.BaselineAspectExtractor(trainset['corpus'])
    #predicted = b1.tag(testset['corpus'].corpus)
    #print('Baseline: P = %f -- R = %f -- F1 = %f (#correct: %d, #retrieved: %d, #relevant: %d)'% Evaluate(testset['corpus'].corpus,predicted).aspect_extraction())

    # paths for CRF++
    crf_learn_path = '/usr/local/bin/crf_learn'
    crf_test_path = '/usr/local/bin/crf_test'
    crf_params = '-c 4.0'
    template_path = 'crf/templates/template'
    model_path = 'crf/models/model'

    train_crfpp(crf_learn_path, crf_params, template_path, model_path, trainset)
    aspect_list = test_crfpp(crf_test_path, model_path, testset)

    predicted = tag(b1,testset['corpus'].corpus,aspect_list)
    #print('MySystem: P = %f -- R = %f -- F1 = %f (#correct: %d, #retrieved: %d, #relevant: %d)'% Evaluate(testset['corpus'].corpus,predicted).aspect_extraction())
    return predicted


######### Main Block ###########

# Load corpora from Pickle or processing everything
corpora = load_corpora()

# Run the aspect extraction for restaurants and laptop domains
for domain_name in ['restaurants','laptop']:
    trainset = corpora[domain_name]['trainset']
    testset = corpora[domain_name]['testset']
    predicted = AspectExtraction(trainset,testset)

    corpus = corpora[domain_name]['trainset']['corpus']
    corpus.write_out('%s--test.predicted-aspect.xml' % domain_name, predicted, short=False)
