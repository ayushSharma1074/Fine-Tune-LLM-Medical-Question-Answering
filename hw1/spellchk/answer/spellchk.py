from transformers import pipeline
from Levenshtein import distance
import logging, os, csv

# use "fill-mask" task and with an pre-trained DistilBERT model with uncased text
fill_mask = pipeline('fill-mask', model='distilbert-base-uncased')
mask = fill_mask.tokenizer.mask_token

def get_typo_locations(fh):
    tsv_f = csv.reader(fh, delimiter='\t')
    for line in tsv_f:
        yield (
            # line[0] contains the comma separated indices of typo words
            [int(i) for i in line[0].split(',')],
            # line[1] contains the space separated tokens of the sentence
            line[1].split()
        )

def select_correction(typo, predict):

    # Approach 1
    # recommended_words = [p['token_str'] for p in predict]
    # levenshtein_distances = [distance(typo, word) for word in recommended_words]
    
    # # find the word with smallest edit distance, if there are multiple same values
    # # return the one with the highest score i.e. the smallest predict indx
    
    # index = levenshtein_distances.index(min(levenshtein_distances))
    # return predict[index]['token_str']

    # Approach 2
    # calculate the edit distance between typo and token_str
    # predict = [{**p, 'ldis': distance(typo, p['token_str'])} for p in predict]
    # keep the predict if the distance is not 0
    # filter_predict = list(filter(filter_p, predict))
    # sort the predict and select the closer distance
    # if the distances are equal, then choose predict with the higher score
    # sort_predict = sorted(filter_predict, key=sorting_k)
    # return sort_predict[0]['token_str']

    
    #Approach 3
    recommended_words = [p['token_str'] for p in predict]
    # Setting inf as the distance value when edit distance is 0.
    levenshtein_distances = [float('inf') if distance(typo, word_tup[0])==0  else distance(typo, word_tup[0]) for word_tup in recommended_words]
    # find the word with smallest edit distance, if there are multiple same values
    # return the one with the highest score i.e. the smallest predict index
    index = levenshtein_distances.index(min(levenshtein_distances))
    return predict[index]['token_str']
    

#def filter_p(p):
    #return p['ldis'] != 0

#def sorting_k(p):
    #res = (p['ldis'], -p['score'])
    #return res

def spellchk(fh):
    for (locations, sent) in get_typo_locations(fh):
        spellchk_sent = sent
        for i in locations:
            # predict top_k replacements only for the typo word at index i
            predict = fill_mask(
                " ".join([ sent[j] if j != i else mask for j in range(len(sent)) ]), # replace typo with a mask
                top_k=3000
            )
            logging.info(predict)
        # Added a condition to change the case of the first character of the word when the index of the word being updated is 0 i.e it is the first word of the sentence.             
            correct_word = select_correction(sent[i], predict)
            if (i==0):
                correct_word= correct_word.capitalize()
            spellchk_sent[i]=correct_word  
        yield(locations, spellchk_sent)

if __name__ == '__main__':
    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-i", "--inputfile", 
                            dest="input", 
                            default=os.path.join('data', 'input', 'dev.tsv'), 
                            help="file to segment")
    argparser.add_argument("-l", "--logfile", 
                            dest="logfile", 
                            default=None, 
                            help="log file for debugging")
    opts = argparser.parse_args()

    if opts.logfile is not None:
        logging.basicConfig(filename=opts.logfile, filemode='w', level=logging.DEBUG)

    with open(opts.input) as f:
        for (locations, spellchk_sent) in spellchk(f):
            print("{locs}\t{sent}".format(
                locs=",".join([str(i) for i in locations]),
                sent=" ".join(spellchk_sent)
            ))
