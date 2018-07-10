Anomalies in Texts of Natural Language
======================================

This directory contains code used to produce the results reported in section 4.3 of the paper.


Detecting Paragraphs in a Different Language
--------------------------------------------

The script `language_identification.py` samples a text from the Europarl corpus and
randomly replace `k` sequences of English sentences with their German counterparts.
The MDI algorithm will then be run to detect the top `2k` anomalies in the resulting
text and the detected paragraphs will be shown, including some context.

The following command will randomly replace 5 English paragraphs of length between 10 and 50 sentences:

    python language_identification.py 10 50 5


Detecting Stylistic Anomalies in the 1st Book of Moses
------------------------------------------------------

The script `genesis.py` runs the MDI algorithm on the 1st Book of Moses in the
King James Version of the bible.

The first argument to that script specifies the word2vec model to be used for
obtaining word embeddings.
The remaining arguments give the minimum and maximum length (in sentences)
and the total number of detections to be returned.

For the results reported in the paper, we used 50-D word embeddings computed
on the Brown corpus using the continuous skip-gram model:

    python genesis.py brown_50_sg.pickle 50 500 10