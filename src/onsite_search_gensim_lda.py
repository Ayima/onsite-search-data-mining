import os
import pickle
import pandas as pd
import logging
from gensim.models import LdaModel
import click
DATA_PATH = '/Volumes/GoogleDrive/My Drive/ga-data-mining/data/'

@click.command()
@click.option(
    '--num-topics',
    default=5,
    help='Number of topics to for LDA model.'
)
def main(num_topics):
    f_path = os.path.join(DATA_PATH, 'interim', 'onsite_search_nlp_gensim_dictionary.pkl')
    with open(f_path, 'rb') as f:
        dictionary = pickle.load(f)
    print('Loaded dictionary: {}'.format(dictionary))

    f_path = os.path.join(DATA_PATH, 'interim', 'onsite_search_terms_2017_2019_nlp.pkl')
    df_search_terms = pd.read_pickle(f_path)
    print('Loaded search corpus: {} rows'.format(len(df_search_terms)))

    print('Logging to terminal')
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    print('Starting model training...')
    print()
    ldamodel = LdaModel(
        corpus=df_search_terms.corpus.dropna().tolist(),
        num_topics=num_topics,
        id2word=dictionary,
    )

    print()
    print('Done training, saving to file')
    f_path = 'onsite_search_terms_lda_2017_2019_{}_topic.model'.format(num_topics)
    ldamodel.save(f_path)




if __name__ == '__main__':
    main()

