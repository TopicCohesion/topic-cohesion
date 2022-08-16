import pandas as pd

def cohesion_calculate(path):
    docs_col, topic_col = generate_docs_topics_collections(path)
    topics_with_scores = calculate_topics_score(docs_col, topic_col)
    score = calculate_topics_cohesion_ratio(topics_with_scores)
    return score

def generate_docs_topics_collections(path):
    data, topics = read_zero_shot_results(path)
    docs_collection = insert_data_to_docs_collection(data, topics)
    topics_collection = insert_data_to_topics_collection(topics)
    return docs_collection, topics_collection

def read_zero_shot_results(path):
    authorized_columns = ['Unnamed: 0', 'doc', 'real_topic']
    data = pd.read_csv(path)
    topics = data.columns
    topics_without_underscore = []
    for topic in topics:
        if topic in authorized_columns:
            topics_without_underscore.append(topic)
        elif topic == 'doc_num':
            continue
        else:
            split_topic = topic.split('_')
            topics_without_underscore.append([topic, split_topic[1]] if len(split_topic) == 2 else [topic, topic])
    return data, topics_without_underscore

# -------------- Insertion --------------
def insert_data_to_docs_collection(docs, topics):
    not_allow_topics = ['Unnamed: 0', 'doc', 'real_topic']
    list_to_insert = []
    scores = {}
    for index, row in docs.iterrows():
        for topic in topics:
            if topic not in not_allow_topics:
                scores[topic[1]] = row[topic[0]]
        object_doc = {'index': row['Unnamed: 0'], 'doc': row['doc'], 'real_topic': row['real_topic'], 'scores': scores.copy()}
        list_to_insert.append(object_doc)
        scores.clear()
    return list_to_insert

def insert_data_to_topics_collection(topics):
    not_allow_topics = ['Unnamed: 0', 'doc', 'real_topic']
    topics_dict = {}
    for index, topic in enumerate(topics):
        if topic in not_allow_topics:
            continue
        object_score = {'index': index - 3, 'description': topic[1], 'positive': 0, 'negative': 0}
        topics_dict[topic[1]] = object_score
    return topics_dict

# -------------- Calculations --------------

def calculate_topics_score(docs_collection, topics_collection):
    for index, (topic_name, topic_data) in enumerate(topics_collection.items()):
        positive, positive_length, negative, negative_length, positive_score, negative_score = 0, 0, 0, 0, 0, 0
        for doc in docs_collection:
            topic_doc_score = doc['scores'][topic_name]
            if doc['real_topic'] == topic_name:
                positive += topic_doc_score
                positive_length += 1
            elif topic_doc_score != 0:
                negative += topic_doc_score
                negative_length += 1
        positive_score = positive/positive_length
        negative_score = negative/negative_length
        topics_collection[topic_name]['positive'] = positive_score
        topics_collection[topic_name]['negative'] = negative_score
    return topics_collection

def calculate_topics_cohesion_ratio(topics_with_scores):
    # Main logic for Cohesion Formula aggregation
    positive, negative, total_score = 0, 0, 0
    for index, (topic_name, topic_data) in enumerate(topics_with_scores.items()):
      ratio = topic_data['positive']/topic_data['negative']
      if ratio>=1:
        topic_total_score = 1 - 1/(ratio+1)
      else:
        topic_total_score = ratio / (1+ratio)
      total_score += topic_total_score
    return round(total_score/len(topics_with_scores),2)