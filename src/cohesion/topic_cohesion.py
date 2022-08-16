import pandas as pd
from cohesion.calculations import calculate_cohesion_score
from cohesion.utils import create_topic_names_using_tf_idf, extract_topics_labels, detailed_topics_print, \
    input_path_to_df, create_future_dirs, delete_tmp_dirs



def run_df(data, docs_percent=1, topics_percent=1, del_tmp=True):
    if type(data) is str:
        print("input is str")
        data = input_path_to_df(data)
    elif type(data) is pd.DataFrame:
        pass
    else:
        raise TypeError("Bad input type")
    return run_cohesion(data, docs_percent, topics_percent, del_tmp)
#
#
def run_cohesion(df, docs_percent, topics_percent, del_tmp):
    print('----------- Starting main pipeline -------------')
    docs = df['text'].tolist()
    labels = df['label'].tolist()
    create_future_dirs()
    topics = create_topic_names_using_tf_idf(docs, labels)
    topics_names = extract_topics_labels(topics)
    # detailed_topics_print(topics_names)
    cohesion_score = calculate_cohesion_score(ground_truth_docs_list=docs,
                                              division_labels=labels, topics_names=topics_names, docs_percent=docs_percent, topics_percent=topics_percent)
    topics_values = [" ".join(topic_words) for topic_words in topics_names]
    if del_tmp:
        delete_tmp_dirs()
    return cohesion_score, topics_values
