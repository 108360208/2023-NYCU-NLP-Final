#!/usr/bin/python
#coding: utf-8

################################################################################################################################
# Author: Jin Wang
# School of Information Science and Engineering, Yunnan University, China
# Department of Computer Science and Engineering, Yuan Ze University, Taiwan
#
# Description: checks format/scores ROCLING 2021 shared task
#
# Usage: python score_evaluation.py <file-prediction> <file-gold>
#
# The script checks the format of <file-prediction>.
# The format of <file-prediction> should be the same as the format of the training data file ""
#
# the script evluates the predictions against the gold ratings
# and outputs the following metrics:
# 1) Mean absolute error (MAE)
# 2) Pearson correlation coefficient
#
# To run the script, you need to install the packages sklearn and scipy
# 
# Last modified: July 26, 2021

from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import logging
import codecs

import numpy as np

from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr

# read an input file
# Format: "word_id, vallence_rating, arousal_rating"
def read_file(file_name):
    valence_dict = {}
    arousal_dict = {}
    count_lines = 0

    my_file = codecs.open(file_name, 'r', 'utf-8')
    try:
        for line in my_file.readlines():
            count_lines = count_lines + 1

            curline = line.replace(' ','').split(',')
            valence_dict[curline[0]] = float(curline[1])
            arousal_dict[curline[0]] = float(curline[2])
    except Exception as e:
        logging.error('Wrong format in line: %d. Expected format: "word_id, vallence_rating, arousal_rating"' % (count_lines))
        exit(1)

    return valence_dict, arousal_dict


if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info(r"running %s" % ' '.join(sys.argv))

    # check and process input arguments
    if len(sys.argv) != 3:
        logging.error('Usage: python score_evaluation.py <file-prediction> <file-gold>')
        sys.exit(1)
    file_prediction, file_gold = sys.argv[1:3]

    # read the input file with predicted valence arousal
    # the coding should be utf-8
    valence_pred_dict, arousal_pred_dict = read_file(file_prediction)

    # read the file with gold ratings
    valence_gold_dict, arousal_gold_dict = read_file(file_gold)

    # if a score is missing for a word in the file with predicted scores
    # assign the default score = 0.0
    valence_pred = []; arousal_pred = []
    valence_gold = []; arousal_gold = []
    for word_id in valence_gold_dict:
        valence_gold.append(valence_gold_dict[word_id])
        arousal_gold.append(arousal_gold_dict[word_id])

        if not word_id in valence_gold_dict:
            logging.error('word_id: %s was not found in %s; assigning the default score of 0.0' % (word_id, file_prediction))
            valence_pred.append(0.0)
            arousal_pred.append(0.0)
        else:
            valence_pred.append(valence_pred_dict[word_id])
            arousal_pred.append(arousal_pred_dict[word_id])

    # if all of the value in input file are the same, set the correlation to zero
    # also the mean absolute error set to eight
    same = True
    for i in range(1, len(valence_pred)):
        if valence_pred[i] != valence_pred[i-1]:
            same = False
            break
        if arousal_pred[i] != arousal_pred[i-1]:
            same = False
            break

    if same:
        logging.error('All values in %s are the same' % (file_prediction))
        logging.info('VALENCE Mean absolute error (MAE): 8.000')
        logging.info('VALENCE Pearson correlation coefficient (Pearsonr): 0.0')
        logging.info('AROUSAL Mean absolute error (MAE): 8.000')
        logging.info('AROUSAL Pearson correlation coefficient (Pearsonr): 0.0')
        sys.exit()

    # output the final score
    valence_pred = np.array(valence_pred)
    arousal_pred = np.array(arousal_pred)
    valence_gold = np.array(valence_gold)
    arousal_gold = np.array(arousal_gold)
    
    valence_mae = mean_absolute_error(valence_gold, valence_pred)
    valence_r = pearsonr(valence_gold, valence_pred)[0]
    arousal_mae = mean_absolute_error(arousal_gold, arousal_pred)
    arousal_r = pearsonr(arousal_gold, arousal_pred)[0]

    logging.info('VALENCE Mean absolute error (MAE): %.3f' % (valence_mae))
    logging.info('VALENCE Pearson correlation coefficient (Pearsonr): %.3f' % (valence_r))
    logging.info('AROUSAL Mean absolute error (MAE): %.3f' % (arousal_mae))
    logging.info('AROUSAL Pearson correlation coefficient (Pearsonr): %.3f' % (arousal_r))