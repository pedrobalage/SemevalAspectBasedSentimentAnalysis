SemevalAspectBasedSentimentAnalysis
===================================

- Author: Pedro Paulo Balage 
- Date: 19/03/2013
- Version: 1.0


My system that participated in Semeval 2014 task 4: Aspect Based Sentiment Analysis ( http://alt.qcri.org/semeval2014/task4/ ). You may reproduce my results by executing:

    python script.py

In order to execute my script you must provide the CRF++ (Conditional Random Field) tool that I used to training and testing.

    https://code.google.com/p/crfpp/

I used CRF++-0.58.tar.gz in my experiments.

Together with this script I am providing the Train, Trial and Test datasets from SemEval Task 4 for the laptops and restaurants domains.

Please, note that these dataset were provided by the SemEval task4 organization and I am only including in my software because I didn't see any copyright information. However, for any purpose besides to reproduce my experiments, you should contact SemEval Task4 organization in order to use their datasets and correctly cite them.  The datasets were downloaded from:

http://alt.qcri.org/semeval2014/task4/index.php?id=data-and-tools

You don't need to reprocess the datasets again. I am including the corpora.pkl file which is a pickled version of corpora python dictionary structure with the datasets already processed. If you may wish to adapt my code for different datasets, you are required to install as well:

- Semaphor v2.1 (a frame-semantic parser for English) - https://www.ark.cs.cmu.edu/SEMAFOR/
- Senna v3 (a fast semantic role labeler) - http://ml.nec-labs.com/senna/

I am also using third party libraries, included in libraries folder. They are:

- semaphore-python - Python interface to SEMAPHORE - https://github.com/jac2130/semaphore-python
- Baseline methods for the 4th task of SemEval 2014 - http://alt.qcri.org/semeval2014/task4/

Any doubts or suggestions, please contact me at: pedrobalage (at) gmail (dot) com
