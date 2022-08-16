from tqdm import tqdm
from transformers import pipeline
import numpy as np
import pandas as pd
import random
import warnings

from cohesion.utils import get_division


class ZeroShotClassifier:
    def __init__(self, path_dic_topic_id_to_name, path_df_docs_topics, topics, ground_truth_docs_list):
        self.model_name = "facebook/bart-large-mnli"
        self.df_docs_topics = pd.read_pickle(path_df_docs_topics)
        self.dic_topic_id_to_name = self.create_topic_ind_name_from_pkl(path_dic_topic_id_to_name)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.classifier = pipeline("zero-shot-classification", model=self.model_name)
        self.ground_truth_docs_list = ground_truth_docs_list
        self.docs, self.series_topics = self.get_topics_and_docs_filtered()

    def run_classify(self, division_labels, path_to_save, docs_percent, topics_percent):
        """
          Cohesion "Heart" - Uses MNLI to find similarity between docs and topics.
          results saved into 'path_to_save'
        """
        topic_names = []
        for i in sorted(self.dic_topic_id_to_name.keys()):
            topic_names.append(self.dic_topic_id_to_name[i])
        division_dict = get_division(division_labels)
        doc_values_to_test, docs_ind_of_selected_docs, topics_ind_of_selected_docs = self.get_selected_docs(
            division_dict, docs_percent)
        results = self.zero_shot_to_every_doc_and_random_topics(doc_values_to_test, topic_names,
                                                                topics_ind_of_selected_docs, docs_percent, topics_percent)
        df = self.create_df_zero_shot_results(topic_names, results, docs_ind_of_selected_docs)
        self.save_mnli_scores(df, path_to_save)

    def get_topics_and_docs_filtered(self):
        """
        drops docs that their topic id is not defined in dic_topic_id_to_name
        series topics is topic name for every doc that pass the filter
        returns docs and topics
        """
        series_topics = self.df_docs_topics["topic_ind"].apply(
            lambda x: self.dic_topic_id_to_name.get(x, np.nan)).dropna()
        docs = self.df_docs_topics["content"][series_topics.index].to_list()
        return docs, series_topics

    def get_selected_topics(self, topic_index_of_doc, topics, set_indexes_real_topics, topics_dict, topics_percent):
        selected_topics = []
        optional_topics = sorted(set_indexes_real_topics)
        optional_topics.remove(topic_index_of_doc)
        amount_of_topics_to_test = 1 + int(
            len(optional_topics) * topics_percent)  # % of relevant topics + 1 (to ensure negative topic as well) + 1 related topic! *MINIMUM amount of topics is 3(!)
        amount_of_topics_to_test = min(amount_of_topics_to_test, len(optional_topics))
        selected_topics_ind = list(random.sample(optional_topics, amount_of_topics_to_test))
        selected_topics_ind = [topic_index_of_doc] + selected_topics_ind  # related label is always at [0].
        for top_ind in selected_topics_ind:
            selected_topics.append(topics_dict[top_ind].split())
        return selected_topics

    def zero_shot_to_every_doc_and_random_topics(self, docs, topics, indexes_real_topic, docs_percent, topics_percent):
        topics_dict = dict()
        sorted_set_indexes_real_topic = sorted(set(indexes_real_topic))
        for i, top_ind in enumerate(sorted_set_indexes_real_topic):
            topics_dict[top_ind] = topics[i]
        results = []
        print_single = True
        set_indexes_real_topics = set(indexes_real_topic)
        for i, doc in enumerate(tqdm(docs)):
            topic_index_of_doc = indexes_real_topic[i]
            selected_topics = self.get_selected_topics(topic_index_of_doc, topics, set_indexes_real_topics, topics_dict, topics_percent)
            doc_topics_result = self.classifier(doc, selected_topics, multi_label=True)
            doc_topics_result["labels"] = [" ".join(label) for label in doc_topics_result["labels"]]
            results.append(doc_topics_result)
        return results

    def save_mnli_scores(self, df, path):
        df.to_pickle(path + ".pkl", protocol=4)
        df.to_csv(path + ".csv")

    def create_topic_ind_name_from_pkl(self, path):
        df_temp = pd.read_pickle(path)
        df_temp = df_temp[df_temp['topic_ind'] != -1]  # drops unassigned docs
        return dict(zip(df_temp["topic_ind"].to_list(), df_temp["topic_name"].to_list()))

    def get_selected_docs(self, division_dict, docs_percent):
        topics_ind_of_selected_docs = []
        docs_ind_of_selected_docs = []
        for i in division_dict.keys():
            docs_in_topic = division_dict.get(i)
            related_docs_amount = int(len(docs_in_topic) * docs_percent)
            related_docs_amount = related_docs_amount if related_docs_amount > 2 else min(2, len(docs_in_topic))
            try:
                selected_docs_from_this_topic = random.sample(docs_in_topic, related_docs_amount)
            except:
                print(related_docs_amount - 1)
                print(docs_in_topic)
            docs_ind_of_selected_docs += selected_docs_from_this_topic
            topics_ind_of_selected_docs += [i] * len(selected_docs_from_this_topic)
        doc_values_to_test = []
        for ind in docs_ind_of_selected_docs:
            doc_values_to_test.append(self.ground_truth_docs_list[ind])
        return doc_values_to_test, docs_ind_of_selected_docs, topics_ind_of_selected_docs

    def create_df_zero_shot_results(self, topic_names, results, docs_ind_of_selected_docs):
        topics_by_doc = self.series_topics.to_list()  # list of topics values, for every doc
        topic_names_with_ind = [str(i) + "_" + topic_name for i, topic_name in enumerate(topic_names)]
        df = pd.DataFrame(columns=["doc_num", "doc", "real_topic"] + topic_names_with_ind)
        for i, result in enumerate(results):
            doc_ind = docs_ind_of_selected_docs[i]
            doc_labels, doc_scores, doc_value = result["labels"], result["scores"], result["sequence"]
            doc_dict = {"doc_num": doc_ind, "doc": doc_value, "real_topic": topics_by_doc[doc_ind]}
            for topic_i, spec_topic in enumerate(self.dic_topic_id_to_name.values()):
                doc_topic_score = 0  # default value for doc - topic. there is no mandatory to test each topic against each doc.
                if spec_topic in doc_labels:
                    index_of_spec_topic = doc_labels.index(spec_topic)
                    doc_topic_score = doc_scores[index_of_spec_topic]
                doc_dict[str(topic_i) + "_" + spec_topic] = doc_topic_score
            df = df.append(doc_dict, ignore_index=True)
        return df