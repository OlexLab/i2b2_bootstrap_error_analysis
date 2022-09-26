# ErrorAnalysis.py
# Amy Olex
# 3/27/22
## Trying to get an error analysis going on to match Chrono phrases to gold phrases.

#import Queue
import os

#import ChronoBERT.utils as utils
import utils
import argparse
from xml.etree.ElementTree import parse, tostring
from itertools import count, groupby
from operator import itemgetter
import pandas as pd
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
import numpy as np



def mergeLenient(gold, pred):
    """
        Merges predicted results and gold results using lenient span matching. Incoming dfs have the following columns: ['id', 'start', 'end', 'text', 'type']
        :param gold: the gold data frame
        :param pred: the predicted data frame of results.
        :return df: a pandas dataframe with matched predicted data.
    """

    ##['id', 'start', 'end', 'text', 'type']

    ## Input dataframe structure: ['id', 'start', 'end', 'text', 'type']
    #df = pd.DataFrame(columns=["gold_id", "gold_label", "gold_start", "gold_end", "gold_text", "gold_context", "gold_coords",
    #                           "pred_id", "pred_label", "pred_start", "pred_end", "pred_text", "pred_context", "pred_coords", "pred_used"])
    df = []
    ## Add flag column to predicted
    pred["used"] = 0
    used_idx = pred.columns.get_loc("used")
    blank_row_gold = ["na","na","na","na","na","na"]
    blank_row_pred = ["na","na","na","na","na","na","na"]

    ## Loop through gold one item at a time.  for each item, loop through all of predicted.
    # If a predicted is matched, then add gold and pred to new df, then mark pred as used.
    for g,grow in gold.iterrows():
        gused = False
        for p, prow in pred.iterrows():
            if prow.start <= grow.start <= prow.end:
                ## we have overlap
                ##print("Pred Start: " + str(prow.start) + "\nPred End: " + str(prow.end) + "\nGold Start: " +
                #     str(grow.start) + "\nPredicted Row: " + str(prow) + "\n Gold Row: " + str(grow))
                gused = True
                pred.iat[p, used_idx] = 1
                prow.used = 1
                df.append(grow.tolist() + prow.tolist())
            elif prow.start <= grow.end <= prow.end:
                ##print("Pred Start: " + str(prow.start) + "\nPred End: " + str(prow.end) + "\nGold End: " +
                #      str(grow.end) + "\nPredicted Row: " + str(prow) + "\n Gold Row: " + str(grow))
                ## we have overlap
                gused = True
                pred.iat[p, used_idx] = 1
                prow.used = 1
                df.append(grow.tolist() + prow.tolist())
            elif prow.start >= grow.start:
                if prow.end <= grow.end:
                    ##print("Pred Start: " + str(prow.start) + "\nPred End: " + str(prow.end) + "\nGold End: " +
                    #      str(grow.end) + "\nPredicted Row: " + str(prow) + "\n Gold Row: " + str(grow))
                    ## we have overlap
                    gused = True
                    pred.iat[p, used_idx] = 1
                    prow.used = 1
                    df.append(grow.tolist() + prow.tolist())

        if not gused:
            #add grow to df with blank pred row
            df.append(grow.tolist() + blank_row_pred)

    pred_not_used = pred[pred["used"] == 0]

    for n,pnu in pred_not_used.iterrows():
        df.append(blank_row_gold + pnu.tolist())

    #pd.DataFrame(df, columns=['Name', 'Age'])

    return pd.DataFrame(df, columns=["id_gold", "start_gold", "end_gold", "text_gold", "label_gold", "value_gold",
                                     "id_pred", "start_pred", "end_pred", "text_pred", "label_pred", "value_pred", "used_pred"])

    #print("done merging!")


def convert_xml_to_df(xml_file, chrono):
    """
            Parses an XML file and returns a data frame
            :param xml_file: the xml file path and name
            :param chrono: boolean is true if the data is coming from chrono
            :return df: a pandas dataframe.
        """

    ### parse the XML files
    try:
        xml_parsed = parse(xml_file)
    except:
        print(xml_parsed)
        raise

    tag_containers = xml_parsed.findall('TAGS')
    tag_container = tag_containers[0]
    timex_tags = tag_container.findall('TIMEX3')

    df = pd.DataFrame(columns=['id', 'start', 'end', 'text', 'type', 'value'])

    for timex_tag in timex_tags:
        id, base_label = timex_tag.attrib['id'], timex_tag.attrib['type']
        start_pos, end_pos, timex_text, value = timex_tag.attrib['start'], timex_tag.attrib['end'], timex_tag.attrib['text'], timex_tag.attrib['val']
        if chrono:
            start_pos = int(start_pos) + 1
            end_pos = int(end_pos) + 1
        start_pos, end_pos = int(start_pos) + 1, int(end_pos)
        df = df.append({'id':id, 'start':start_pos, 'end':end_pos, 'text':timex_text, 'type':base_label, 'value':value}, ignore_index=True)

    #print(df)
    return df

if __name__ == '__main__':

    ## Parse input arguments
    parser = argparse.ArgumentParser(
        description='Format XML i2b2 annotations into DATE-DUR format for Seq2Seq and SVM inputs.')
    parser.add_argument('-p', metavar='filedirectory', type=str,
                        help='Path to directory that has the Prediction XML files needed for conversion.',
                        required=True)
    parser.add_argument('-g', metavar='filedirectory', type=str,
                        help='Path to directory that has the Gold XML files needed for conversion.',
                        required=True)
    parser.add_argument('-c', metavar='chrono', type=bool,
                        help='If processing Chrono output it is always off by one.',
                        required=False, default=False)
    parser.add_argument('-b', metavar='chrono', type=int,
                        help='Number of iterations to run the bootstrap analysis with replacement. Default is 100.',
                        required=False, default=100)
    parser.add_argument('-o', metavar='output', type=str,
                        help='Unique path to and name of output file (or directory for individual file processing) minus extension.',
                        required=False, default='./error_analysis_output')

    args = parser.parse_args()

    pred_dir = os.path.join('', args.p)
    gold_dir = os.path.join('', args.g)

    gold_xml_filenames = [x for x in os.listdir(gold_dir) if x.endswith('xml')]
    pred_xml_filenames = [x for x in os.listdir(pred_dir) if x.endswith('xml')]
    print(str(gold_xml_filenames == pred_xml_filenames))

    all_data = pd.DataFrame()#columns=["id_gold", "label_gold", "start_gold", "end_gold", "text_gold",
                             #        "id_pred", "label_pred", "start_pred", "end_pred", "text_pred",
                             #        "used_pred", "file_name"])
    for xml_filename in gold_xml_filenames:
        gold_df = convert_xml_to_df(os.path.join(gold_dir, xml_filename), False)
        pred_df = convert_xml_to_df(os.path.join(pred_dir, xml_filename), args.c)
        merged_data = mergeLenient(gold_df, pred_df)
        merged_data["file_name"] = xml_filename
        all_data = all_data.append(merged_data, ignore_index=True)

    f = open(args.o + '.tsv', mode='w')
    all_data.to_csv(args.o + '.tsv')

    print("Completed error analysis, running bootstrap analysis...")

    ## Run Bootstrap Analysis
    ## code from https://carpentries-incubator.github.io/machine-learning-novice-python/07-bootstrapping/index.html

    # bootstrap predictions
    ## remove NAs
    all_data_filt = all_data.drop(all_data.index[all_data["label_pred"] == "na"].tolist())

    accuracy = []
    specificity = []
    precision = []
    recall = []
    f1 = []
    n_iterations = args.b
    n_sample = int(all_data.shape[0]*1)
    print(str(n_sample))
    tf = f1_score(all_data["label_gold"], all_data["label_pred"], average="weighted")
    tp = precision_score(all_data["label_gold"], all_data["label_pred"], average="weighted")
    tr = recall_score(all_data["label_gold"], all_data["label_pred"], average="weighted")
    ta = accuracy_score(all_data_filt["label_gold"], all_data_filt["label_pred"])

    for i in range(n_iterations):
        pred_y, gold_y = resample(all_data["label_pred"], all_data["label_gold"], n_samples=n_sample, replace=True, random_state=42+i)
        gold_y_filt = gold_y.drop(pred_y.index[pred_y == "na"].tolist())
        pred_y_filt = pred_y.drop(pred_y.index[pred_y == "na"].tolist())

        accuracy.append(accuracy_score(gold_y_filt, pred_y_filt))
        precision.append(precision_score(gold_y, pred_y, average="weighted", zero_division=0))
        recall.append(recall_score(gold_y, pred_y, average="weighted", zero_division=0))
        f1.append(f1_score(gold_y, pred_y, average="weighted", zero_division=0))

    ##create df and save
    boot_df = pd.DataFrame({'Precision':precision, 'Recall':recall, 'F1':f1, 'Accuracy':accuracy})
    f = open(args.o + '_bootstrap_eval_wReplace_'+str(args.b)+'_estimates_and_ci.tsv', mode='w')
    boot_df.to_csv(args.o + '_bootstrap_eval_wReplace_'+str(args.b)+'.tsv')

    ## get mean scores
    acc_mean = np.mean(accuracy)
    precision_mean = np.mean(precision)
    recall_mean = np.mean(recall)
    f1_mean = np.mean(f1)

    ## get confidence intervals
    alpha = 0.05
    acc_lower_bound = np.percentile(accuracy, 100 * (alpha / 2))
    acc_upper_bound = np.percentile(accuracy, 100 * (1 - alpha / 2))
    acc_stderr = np.std(accuracy)

    precision_lower_bound = np.percentile(precision, 100 * (alpha / 2))
    precision_upper_bound = np.percentile(precision, 100 * (1 - alpha / 2))
    precision_stderr = np.std(precision)

    recall_lower_bound = np.percentile(recall, 100 * (alpha / 2))
    recall_upper_bound = np.percentile(recall, 100 * (1 - alpha / 2))
    recall_stderr = np.std(recall)

    f1_lower_bound = np.percentile(f1, 100 * (alpha / 2))
    f1_upper_bound = np.percentile(f1, 100 * (1 - alpha / 2))
    f1_stderr = np.std(f1)

    f.write("P.total\tR.total\tF1.total\tAcc.totoal\tP.est\tP.lower\tP.upper\tP.stderr\tR.est\tR.lower\tR.upper\tR.stderr"
            "\tF1.est\tF1.lower\tF1.upper\tF1.stderr\tAcc.est\tAcc.lower\tAcc.upper\tAcc.stderr\n")
    f.write("\t".join([str(tp),str(tr),str(tf),str(ta),
                        str(precision_mean),str(precision_lower_bound),str(precision_upper_bound),str(precision_stderr),
                        str(recall_mean),str(recall_lower_bound),str(recall_upper_bound),str(recall_stderr),
                        str(f1_mean),str(f1_lower_bound),str(f1_upper_bound),str(f1_stderr),
                        str(acc_mean),str(acc_lower_bound),str(acc_upper_bound),str(acc_stderr)]))
    f.close()
    print("Completed bootstrap analysis!")








