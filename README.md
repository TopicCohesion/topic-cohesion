# Topic Cohesion


The Topic-Detection field deals mainly with providing names to given divisions of documents and lacks a quality measurement that provides a rating for the division, that represent a human-subjective score.

Given a division topic_cohesion will calculate the human-subjective score, and the related topic name to each label in a division.

The POC to this attitude can be found in the [colab-notebook](https://colab.research.google.com/drive/1IFWKF3CFfzZWT9WucdISpDbLDI4mtTjX?usp=sharing), or in the ["Topic Cohesion Project- Full report"](https://github.com/Berdugo1994/cohesion-pipeline/blob/main/Cohesion%20Pipeline%20Project%20-%20Full%20Report.pdf)

The useage example can be also found in the [colab-notebook-usage-example](https://colab.research.google.com/drive/1zAJs5px8HBMo99hPc-MhnlInKi7ze9yI?usp=sharing)


## Installation

```bash
pip install topic-cohesion
```

## Usage Example
The input to the cohesion_score function must be a csv, txt, tsv file with a tab ['\t'] seperator and must have 'label' and 'text' columns.
The 'text' is a list of strings which represents all the corpus senteces while the 'label' is a list of integers that repressents the corpus divison.
In the next example, senteces 1, 2, 3 are belong to group 1 and senteces 4 and 5 belongs to group 2.

```python
import pandas as pd
from cohesion import topic_cohesion

data = {'text':
            ["we like to play football",
             "I'm playing football better than neymar and cristano ronaldo",
             "I like Fifa more than I like football, My Fav team is #RealMadrid Hala Madrid",
             "Hamburger or Pizza? what would i choose? I will eat both of them, it so tasty!",
             "banana pancakes with syrup maple, thats my favorite meal"],
        'label':
            [1, 1, 1, 2, 2]}
df = pd.DataFrame(data)
score, topic_names  = topic_cohesion.run_df(df)
print("Cohesion Final score is: ", score)
print("Cohesion Topics are: ", topic_names)

```

Expected output
```
Cohesion Final score is: 0.99
Cohesion Topics are: ['like football play ronaldo playing', 'tasty pizza hamburger eat choose']
```
