#!/usr/bin/env python
#
# semaphore-python - Python interface to SEMAPHORE
# Copyright (c) 2013 Johannes Castner
# https://github.com/jac2130/semaphore-python
#
#SEMAFOR (a frame-semantic parser for English)
#Copyright (C) 2012
#Dipanjan Das, Andre Martins, Nathan Schneider, Desai Chen, & Noah A. Smith
#Language Technologies Institute, Carnegie Mellon University
#http://www.ark.cs.cmu.edu/SEMAFOR
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.

import re, sys, os


#getting the directories which are set by the user in the config file (located in the release directory).

from semaphore_directories import *

os.chdir(here)

def run_semaphore(release=release, command='fnParserDriver.sh', sample='../samples/sample.txt', output= '../samples/output.txt'):
    '''
    This function produces an xml file containing the frame-net style frames.
    "./fnParserDriver.sh" is an unwieldy bash script that I'm hoping to replace entirely with python code soon, in order to have fine
    control of the behavior of semaphore from python. But this is still relatively low on my list of priorities.
    '''

    try:
        os.chdir(release)

        os.system(command + ' ' + sample + " " + output)
    finally:
        #to make sure that we are in the directory in which semaphore.py is stored.
        os.chdir(here)

def import_semaphore(xml=output):

    '''
    Takes the xml output that results from running the Semaphore program and returns a python object.
    '''
    from collections import OrderedDict
    import xmltodict

    with open(xml, 'r') as f:
        raw_dict=xmltodict.parse(f.read())

    #cutting the initial layers of an unwieldy xml dictionary with far too many xml tags, as we are assuming a list of sentences:
    raw_list=raw_dict[u'corpus'][u'documents'][u'document'][u'paragraphs'][u'paragraph'][u'sentences'][u'sentence']

    try: raw_text=[str(raw_list[i][u'text']) for i in range(len(raw_list))]
    except: raw_text=[raw_list['text']]

    #sometimes there are annotationSets and sometimes not:

    def annotationSets(dictionary):
        try:
            return dictionary['annotationSets']['annotationSet']
        except:
            return []

    try: raw_list=[annotationSets(raw_list[i]) for i in range(len(raw_list))] #cleaning it up further
    except: raw_list=[annotationSets(raw_list)]

    def get_frames(list_or_dict, raw_text):
        #This is a function that deals with the uncertainty of whether we are getting a list or dictionary here (the xml output of semaphore is quite unwieldy)

        if not list_or_dict: #in the case that it is empty (no frames)
            return {}

        def de_nest(d, raw_text):
            try:
                if d['labels']!=None:
                    return [(str(d['labels']['label']['@name']), raw_text[eval(d['labels']['label']['@start']): eval(d['labels']['label']['@end']) + 1])]
                else:
                    return []
            except:
                return [(str(r['@name']), raw_text[eval(r['@start']): eval(r['@end']) +1]) for r in d['labels']['label']]

        try:
            return {str(list_or_dict[j]['@frameName']): dict([de_nest(d, raw_text)[l] for d in list_or_dict[j]['layers']['layer'] for l in range(len(de_nest(d, raw_text)))]) for j in range(len(list_or_dict))}

         #in case that it is a list (when there are j frames)
        except KeyError:

            return {str(list_or_dict['@frameName']): dict([de_nest(d, raw_text)[l] for d in list_or_dict['layers']['layer'] for l in range(len(de_nest(d, raw_text)))])}


    frame_dict=[{} for i in range(len(raw_list))]
    for i in range(len(raw_list)):
        try: frame_dict[i]['fn-labels'] =get_frames(raw_list[i], raw_text[i])
        except: frame_dict[i]['fn-labels']={}

        frame_dict[i]['text']=raw_text[i]

    return frame_dict

def clean_raw_text(text, file_name=''):

    '''
    cleans all text input and places the cleaned text in the 'samples' folder, one line at the time (as required by semaphore).
    '''

    import re
    import nltk, nltk.data

    sent_detector=nltk.data.load('tokenizers/punkt/english.pickle')

    raw_text=text
    clean_file=file_name if file_name else 'clean_text.txt'

    text=re.sub(r'-+(\n)\s*', '', raw_text)
    text=re.sub(r'(\n)+', '', text)

    text= '\n'.join([' '.join(nltk.word_tokenize(sent)) for sent in sent_detector.tokenize(text.strip())])
    open(clean_file, 'w').write(text)


def semaphore(text='', files='', semaphore=release):
    '''
    This function takes a string or a list of file path names and outputs a python data structure containing semantic frames for each
    sentence (the text can be completely raw).
    '''
    os.chdir(semaphore)

    if text:
        sample='../samples/cleaned.txt'
    if files:
        text=text+' '.join([open(f, 'r').read() for f in files])
        #I just name the newly cleaned file by the name of the first file in the file list + "_clean":
        sample='../samples/' + files[0].split('/')[-1][:-4] + '_clean.txt'

    if text:
        clean_raw_text(text, file_name=sample)

    else:
        sample='../samples/sample.txt'

    run_semaphore(release=semaphore, sample=sample)

    return import_semaphore()


def mysemaphore(text, path, semaphore=release):
    import shutil

    '''
    This function takes a string or a list of file path names and outputs a python data structure containing semantic frames for each
    sentence (the text can be completely raw).
    '''
    os.chdir(semaphore)

    sample='../samples/cleaned.txt'

    open(sample, 'w').write(text)

    run_semaphore(release=semaphore, sample=sample)

    shutil.copy (semaphore+'/../samples/output.txt',path)

    return import_semaphore()

