from cohesion.topic_cohesion import cohesion_df
import pandas as pd
def test_cohesion_df_with_text():
    res, topics = cohesion_df(r'../resources/5_topics_manual_created.txt')
    print("res",res)
    print("topics")
    print(topics)
    assert len(topics) == 5
    assert 0.85 > res > 0.8


def test_cohesion_df_with_df():
    df = pd.read_csv(r'../resources/5_topics_manual_created.txt', sep='\t', usecols=['label', 'text'])
    res, topics = cohesion_df(df)
    print("res",res)
    print("topics")
    print(topics)
    assert len(topics) == 5
    assert 0.85 > res > 0.8