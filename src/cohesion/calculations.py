from cohesion.utils import docs_to_groups, create_input_to_mnli_topics_content, create_topic_id_topic_value
from cohesion.zero_shot_classifier import ZeroShotClassifier
from cohesion.formula import cohesion_calculate


def calculate_cohesion_score(ground_truth_docs_list, division_labels, topics_names, docs_percent, topics_percent):
    # prepare pickles files
    path_to_save = '.\\tmp_coh_exports\\mnli\\doc_topic_mnli_results'
    topic_id_content = '.\\tmp_coh_exports\\topics_map\\topic_id_content.pkl'
    topic_id_value_map = '.\\tmp_coh_exports\\topics_map\\topic_id_topic_value_map.pkl'

    clusters_docs = docs_to_groups(ground_truth_docs_list, division_labels)
    create_input_to_mnli_topics_content(clusters_docs, topic_id_content)
    create_topic_id_topic_value(sorted(clusters_docs.keys()), topics_names, topic_id_value_map)

    # create csv with mnli score
    zero_shot_classifier = ZeroShotClassifier(topic_id_value_map, topic_id_content, division_labels, ground_truth_docs_list)
    zero_shot_classifier.run_classify(division_labels, path_to_save, docs_percent, topics_percent)
    return cohesion_calculate(path=path_to_save + '.csv')