import os
import random
import pandas as pd
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import shutil
import warnings


def nltk_download():
    nltk.download('stopwords')
    nltk.download('punkt')


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    nltk_download()
stop_words = set(stopwords.words('english'))


def tokenizer(text):
    word_tokens = word_tokenize(text)
    bad_words_list = ["http"]
    filtered_sentence = [w.lower() for w in word_tokens if
                         not w.lower() in stop_words and w.isalpha() and len(w) > 2 and w not in bad_words_list]
    return filtered_sentence


def docs_to_groups(docs, clusters):
    clusters_docs = dict()
    for index, doc in enumerate(docs):
        cluster = int(clusters[index])
        docs_list = clusters_docs.get(cluster, list())
        docs_list.append(doc)
        clusters_docs[cluster] = docs_list
    return clusters_docs


def create_input_to_mnli_topics_content(content_dict, path):
    content_list = []
    topics_ind_list = []
    for topic_ind in sorted(content_dict.keys()):
        content_list += content_dict[topic_ind]
        topics_ind_list += [topic_ind] * len(content_dict[topic_ind])
    df_topic_id_content = pd.DataFrame({"topic_ind": topics_ind_list, "content": content_list})
    df_topic_id_content.to_pickle(path, protocol=4)


def create_topic_id_topic_value(topics_ind_list, topic_values, path):
    topic_values_joined = []
    for ten_words_seperated in topic_values:
        topic_values_joined.append(" ".join(ten_words_seperated))
    df_topic_id_value_coherence = pd.DataFrame({"topic_ind": topics_ind_list, "topic_name": topic_values_joined})
    df_topic_id_value_coherence.to_pickle(path, protocol=4)


def tf_idf_top_words(docs: list, words_count=6):
    vectorizer = TfidfVectorizer(use_idf=True, stop_words='english', ngram_range=(1, 1), tokenizer=tokenizer)
    vectors = vectorizer.fit_transform(docs)
    feature_names = vectorizer.get_feature_names_out()
    dense = vectors.todense()
    denselist = dense.tolist()
    df = pd.DataFrame(denselist, columns=feature_names)
    return list(df.mean(axis=0).sort_values(ascending=False).index)[0:words_count]


def docs_to_clusters(docs_list, docs_label):
    clusters_docs = dict()
    for index, label in enumerate(docs_label):
        doc_list = clusters_docs.get(label, list())
        doc_list.append(docs_list[index])
        clusters_docs[label] = doc_list
    return clusters_docs


def get_division(docs_label):
    clusters_docs = dict()
    for index, label in enumerate(docs_label):
        index_list = clusters_docs.get(label, list())
        index_list.append(index)
        clusters_docs[label] = index_list
    return clusters_docs


def create_topic_name(key, labels_array):
    topic_name = str(key)
    for label in labels_array:
        topic_name += '_' + label
    return topic_name


def create_topic_names_using_tf_idf(docs, labels):
    """
    docs - list of all docs values.
    labels - labels as list.
    returns dict of key:label ind, value: list of docs-related.
    """
    tf_idf_labels = dict()
    docs_groups = docs_to_clusters(docs, labels)  # dict where key is label ind, and value is list of docs values.
    for key, value in docs_groups.items():
        topic_names = tf_idf_top_words(value, words_count=5)
        tf_idf_labels[key] = create_topic_name(key, topic_names)
    return tf_idf_labels


def create_topic_names_using_tf_idf_with_optimization(docs, labels):
    # Optional: alternative function to 'create_topic_names_using_tf_idf'
    # Use this func to optimize TF-IDF selected words between topics
    porter = PorterStemmer()
    tf_idf_labels, topics_commonly, all_topics_words = dict(), dict(), dict()
    docs_groups = docs_to_clusters(docs, labels)
    for key, value in docs_groups.items():
        labels_array = tf_idf_top_words(value, words_count=7)
        all_topics_words[key] = labels_array
        for word in labels_array:
            stemmed_word = porter.stem(word)
            count = topics_commonly.get(stemmed_word, 0)
            topics_commonly[stemmed_word] = count + 1
    optimize_topic_words(all_topics_words, topics_commonly, tf_idf_labels, docs_groups, words_count=7,
                         final_words_count=3)
    return tf_idf_labels


def optimize_topic_words(all_topics_words, topics_commonly, tf_idf_labels, docs_groups, words_count=7,
                         final_words_count=5, tf_idf_common_threshold=0.4):
    porter = PorterStemmer()
    topics_amount = len(docs_groups.keys())
    for key, value in docs_groups.items():
        final_words = [""] * final_words_count
        replaced_count = 0
        for i in range(final_words_count - 1, -1, -1):
            stemmed_word = porter.stem(all_topics_words[key][i])
            if topics_commonly[
                stemmed_word] / topics_amount > tf_idf_common_threshold and replaced_count < words_count - final_words_count:
                alternative_word = all_topics_words[key][final_words_count + replaced_count]
                alternative_word_stemmed = porter.stem(alternative_word)
                if topics_commonly[stemmed_word] > topics_commonly[alternative_word_stemmed]:
                    print("stemmed word {} - replaced with {}".format(stemmed_word, alternative_word))
                    final_words[i] = alternative_word
                    replaced_count += 1
                else:
                    # The word is common but the alternative is even worst.
                    final_words[i] = all_topics_words[key][i]
                # IF we would like to extract this word from *some* of the topics, it should be done here.
                # topics_commonly[stemmed_word] = topics_commonly[stemmed_word] - 1
            else:
                final_words[i] = all_topics_words[key][i]
        tf_idf_labels[key] = create_topic_name(key, final_words)


# def create_division(ground_truth_df, random_percent):
#     # Helper function for create_random_divisions.
#     # Creates a division given ground truth division and random-error-percent
#     labels = ground_truth_df["label"].to_list()
#     result = labels.copy()  # [0,0,1,1,2,2]
#     labels_indexes = range(len(result))  # [0,1,2,3,4,5]
#     labels_set = sorted(set(labels))  # [0,1,2]
#     indexes_to_be_replaced = random.sample(labels_indexes, int((random_percent / 100) * len(labels_indexes)))
#     for index_to_be_replace in indexes_to_be_replaced:
#         alternative_labels = list(labels_set)
#         alternative_labels.remove(result[index_to_be_replace])
#         result[index_to_be_replace] = random.sample(alternative_labels, 1)[0]
#     return result


def extract_topics_labels(topics_names_dict):
    """
    returns list of topic names where each topic.
    each element returned is list of words that represent the topic.
    """
    topics_names = []
    for topic_ind in sorted(topics_names_dict.keys()):
        topics_names.append(topics_names_dict[topic_ind].split('_')[1:])
    return topics_names


def input_path_to_df(path):
    return pd.read_csv(path, sep='\t', usecols=['label', 'text'])


def detailed_topics_print(topics_names):
    print("\n", "*" * 5, "\tTopics\t", "*" * 5)
    for i, topic_name in enumerate(topics_names):
        print("Topic {} value is {}".format(i, topic_name))


def create_future_dirs():
    if not os.path.exists(".\\tmp_coh_exports"):
        os.mkdir(".\\tmp_coh_exports")
        os.mkdir(".\\tmp_coh_exports\\mnli")
        os.mkdir(".\\tmp_coh_exports\\topics_map")


def delete_tmp_dirs():
    if os.path.exists(".\\tmp_coh_exports"):
        shutil.rmtree(".\\tmp_coh_exports")
