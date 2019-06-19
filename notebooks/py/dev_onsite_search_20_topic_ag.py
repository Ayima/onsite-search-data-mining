
# coding: utf-8

# %load jupyter_default.py
import pandas as pd
import numpy as np
import os
import sys
import re
import datetime
import time
import glob
import json
from tqdm import tqdm_notebook
from colorama import Fore, Style

from dotenv import load_dotenv
load_dotenv('../../.env')

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import matplotlib.colors
import seaborn as sns

get_ipython().run_line_magic('config', "InlineBackend.figure_format='retina'")
sns.set() # Revert to matplotlib defaults
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['axes.labelpad'] = 20
plt.rcParams['legend.fancybox'] = True
plt.style.use('ggplot')

SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE = 14, 16, 20
plt.rc('font', size=SMALL_SIZE)
plt.rc('axes', titlesize=SMALL_SIZE)
plt.rc('axes', labelsize=MEDIUM_SIZE)
plt.rc('xtick', labelsize=SMALL_SIZE)
plt.rc('ytick', labelsize=SMALL_SIZE)
plt.rc('legend', fontsize=MEDIUM_SIZE)
plt.rc('axes', titlesize=BIGGER_SIZE)

def savefig(name):
    file = '../../results/figures/{}.png'.format(name)
    print('Saving figure to file {}'.format(file))
    plt.savefig(file, bbox_inches='tight', dpi=300)

get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
    
get_ipython().run_line_magic('reload_ext', 'version_information')
get_ipython().run_line_magic('version_information', 'pandas, numpy')


# # Onsite Search
# Alex's development notebook for onsite search.

# ## Load Data

# ### Tests - Load from the GA API v4

load_dotenv('../../.env')


from apiclient.discovery import build
from oauth2client.service_account import ServiceAccountCredentials
from oauth2client.client import OAuth2WebServerFlow

SCOPES = ['https://www.googleapis.com/auth/analytics.readonly']
KEY_FILE_LOCATION = os.environ.get('GOOGLE_SERVICE_ACCOUNT_CREDENTIALS_1')
VIEW_ID = os.environ.get('GOOGLE_ANALYTICS_VIEW_ID_1')


KEY_FILE_LOCATION = '../../ga-api-240120-d7d392f57597.json'


def initialize_api():
    credentials = ServiceAccountCredentials.from_json_keyfile_name(
        KEY_FILE_LOCATION, SCOPES
    )
    api = build('analyticsreporting', 'v4', credentials=credentials)
    return api

def get_report(api, body):
    """
    Run a request through the API.
    
    api : apiclient.discovery.build
        Returned from initialize_api function.
        
    body : dict
        Request body.
        
        e.g.
        body={
          'reportRequests': [
          {
            'viewId': VIEW_ID,
            'dateRanges': [{'startDate': '7daysAgo', 'endDate': 'today'}],
            'metrics': [{'expression': 'ga:sessions'}],
            'dimensions': [{'name': 'ga:country'}]
          }]
        }
    """
    return api.reports().batchGet(body=body).execute()


api = initialize_api()


body = {
  "reportRequests": [
    {
      "viewId": "17785508",
      "dateRanges": [
        {
          "startDate": "2daysAgo",
          "endDate": "yesterday"
        }
      ],
      "metrics": [
        {
          "expression": "ga:searchUniques"
        }
      ],
      "dimensions": [
        {
          "name": "ga:searchKeyword"
        }
      ]
    }
  ]
}

d = get_report(api, body=body)


# This is not working as I would need to add the API user to the account (see [here](https://stackoverflow.com/questions/12837748/analytics-google-api-error-403-user-does-not-have-any-google-analytics-account) for more info)
# 
# I was able to get it working with oath2 authentication via manual action at runtime

from apiclient.discovery import build
from oauth2client.client import OAuth2WebServerFlow

SCOPES = ['https://www.googleapis.com/auth/analytics.readonly']
KEY_FILE_LOCATION = os.environ.get('GOOGLE_SERVICE_ACCOUNT_CREDENTIALS_1')
VIEW_ID = os.environ.get('GOOGLE_ANALYTICS_VIEW_ID_1')


get_ipython().run_line_magic('pinfo', 'OAuth2WebServerFlow')


KEY_FILE_LOCATION = '../../ga-api-240120-d7d392f57597.json'


def initialize_api():
    flow = OAuth2WebServerFlow('1057949313358-doha6r9cl0d5utu5goa3d2advgqjcglo.apps.googleusercontent.com',
                               '25CUJ__eFe7TzZxs8N6avEea', 'https://www.googleapis.com/auth/analytics.readonly',
                              redirect_uri='urn:ietf:wg:oauth:2.0:oob')
    authorize_url = flow.step1_get_authorize_url()
    print('Receive code from:\n%s\n' % authorize_url)
    code = input('Enter code here:').strip()
    credentials = flow.step2_exchange(code)
    
    api = build('analyticsreporting', 'v4', credentials=credentials)
    return api

def get_report(api, body):
    """
    Run a request through the API.
    
    api : apiclient.discovery.build
        Returned from initialize_api function.
        
    body : dict
        Request body.
        
        e.g.
        body={
          'reportRequests': [
          {
            'viewId': '273664',
            'dateRanges': [{'startDate': '7daysAgo', 'endDate': 'today'}],
            'metrics': [{'expression': 'ga:sessions'}],
            'dimensions': [{'name': 'ga:country'}]
          }]
        }
    """
    return api.reports().batchGet(body=body).execute()


api = initialize_api()


body = {
    "reportRequests": [{
        "viewId": VIEW_ID,
        "dateRanges": [{
          "startDate": "5daysAgo",
          "endDate": "yesterday",
        }],
        "metrics": [{
            "expression": "ga:searchUniques",
        }],
        "dimensions": [{
            "name": "ga:searchKeyword",
            "name": "ga:date",
        }]
    }]
}

d = get_report(api, body=body)


d


print(json.dumps(d, indent=2)[:1000])


body = {
    "reportRequests": [{
        "viewId": VIEW_ID,
        "dateRanges": [{
          "startDate": "5daysAgo",
          "endDate": "yesterday",
        }],
        "metrics": [{
            "expression": "ga:searchUniques",
        }],
        "dimensions": [{
            "name": "ga:searchKeyword",
        },
        {
            "name": "ga:date",
        }]
    }]
}

d = get_report(api, body=body)


print(json.dumps(d, indent=2)[:1000])


rows = d.get('reports', [{}])[0].get('data', {}).get('rows', [])


data = []
for i, row in enumerate(rows):
    d = []
    
    dims = row.get('dimensions', [])
    if len(dims) == 2:
        d += [dims[1], dims[0]]
    else:
        raise Exception('Warning, response parse failed as len(dim) != 2 for row {}'.format(i))
        
    vals = row.get('metrics', [{}])[0].get('values', [])
    if len(vals) == 1:
        d += vals
    else:
        raise Exception('Warning, response parse failed as len(vals) != 1 for row {}'.format(i))
        
    data.append(d)
        
df = pd.DataFrame(data, columns=['date', 'search_term', 'num_searches'])
df.date = pd.to_datetime(df.date)
df.count = df.num_searches.astype(int)
df.head()


d['reports'][0].keys()


d['reports'][0]['nextPageToken']


body = {
    "reportRequests": [{
        "viewId": VIEW_ID,
        "pageToken": "1000",
        "pageSize": "1000",
        "dateRanges": [{
          "startDate": "5daysAgo",
          "endDate": "yesterday",
        }],
        "metrics": [{
            "expression": "ga:searchUniques",
        }],
        "dimensions": [{
            "name": "ga:searchKeyword",
        },
        {
            "name": "ga:date",
        }]
    }]
}

d = get_report(api, body=body)


d['reports'][0]


len(d['reports'])


d_1000 = d.copy()


body = {
    "reportRequests": [{
        "viewId": VIEW_ID,
        "pageToken": "2",
        "pageSize": "10000",
        "dateRanges": [{
          "startDate": "5daysAgo",
          "endDate": "yesterday",
        }],
        "metrics": [{
            "expression": "ga:searchUniques",
        }],
        "dimensions": [{
            "name": "ga:searchKeyword",
        },
        {
            "name": "ga:date",
        }]
    }]
}

d = get_report(api, body=body)


data = []
for i, row in enumerate(rows):
    d = []
    
    dims = row.get('dimensions', [])
    if len(dims) == 2:
        d += [dims[1], dims[0]]
    else:
        raise Exception('Warning, response parse failed as len(dim) != 2 for row {}'.format(i))
        
    vals = row.get('metrics', [{}])[0].get('values', [])
    if len(vals) == 1:
        d += vals
    else:
        raise Exception('Warning, response parse failed as len(vals) != 1 for row {}'.format(i))
        
    data.append(d)
        
df = pd.DataFrame(data, columns=['date', 'search_term', 'num_searches'])
df.date = pd.to_datetime(df.date)
df.count = df.num_searches.astype(int)
df.head()


len(df)


# This looks the same as not passing pageToken... hmm

# ### Load from the GA API v4

from apiclient.discovery import build
from oauth2client.client import OAuth2WebServerFlow

SCOPES = ['https://www.googleapis.com/auth/analytics.readonly']
KEY_FILE_LOCATION = os.environ.get('GOOGLE_SERVICE_ACCOUNT_CREDENTIALS_2')
VIEW_ID = os.environ.get('GOOGLE_ANALYTICS_VIEW_ID_2')


from typing import Tuple

class GoogleAnalyticsApi:
    def __init__(self, raise_errors=False):
        self.raise_errors = raise_errors
        self.api = self.initialize_api()

    def initialize_api(self):
        flow = OAuth2WebServerFlow(
            client_id=self.client_id,
            client_secret=self.client_secret,
            scope='https://www.googleapis.com/auth/analytics.readonly',
            redirect_uri='urn:ietf:wg:oauth:2.0:oob'
        )
        authorize_url = flow.step1_get_authorize_url()
        print('Receive code from:\n{}\n'.format(authorize_url))
        code = input('Enter code here: ').strip()
        credentials = flow.step2_exchange(code)
        
        api = build('analyticsreporting', 'v4', credentials=credentials)
        return api

    @property
    def client_id(self):
        try:
            with open(KEY_FILE_LOCATION, 'r') as f:
                return json.load(f)['client_id']
        except:
            sys.exit('Unable to load client_id from file {}'
                      .format(KEY_FILE_LOCATION))
            return ''
        
    @property
    def client_secret(self):
        try:
            with open(KEY_FILE_LOCATION, 'r') as f:
                return json.load(f)['client_secret']
        except:
            sys.exit('Unable to load client_secret from file {}'
                      .format(KEY_FILE_LOCATION))
            return ''

    def query_onsite_search(
        self,
        start_date,
        end_date, 
        api_delay=3,
        verbose=False,
        test=False,
        raise_errors=False,
    ) -> pd.DataFrame:
        """
        Make API request for on-site search data, grouped by day.
        
        start_date : str
            ISO formatted start date.
            
        end_date : str
            ISO formatted end date.
            
        api_delay : int
            Delay in between requests in seconds. Multiple requests
            will be made due to pagination.
        """
        dfs = []
        page_token = None
        while True:
            print('Making API request with page_token = {}'.format(page_token))
            df_, page_token = self.query_onsite_search_page(
                start_date,
                end_date,
                page_token,
                verbose,
                raise_errors,
            )
            dfs.append(df_)
            if ((len(df_) == 0) or (page_token is None)):
                break
            if verbose:
                print('Waiting for {} seconds'.format(api_delay))
            time.sleep(api_delay)
            
            if test:
                print('Testing mode - exiting API call loop')
                break
        
        return pd.concat(dfs, sort=False, ignore_index=True)

        
    def query_onsite_search_page(
        self,
        start_date,
        end_date,
        page_token,
        verbose,
        raise_errors,
    ) -> Tuple[pd.DataFrame, int]: 
        body = {
            "reportRequests": [{
                "viewId": VIEW_ID,
                "pageSize": "1000",                
                "dateRanges": [{
                  "startDate": start_date,
                  "endDate": end_date,
                }],
                "metrics": [{
                    "expression": "ga:searchUniques",
                }],
                "dimensions": [
                {
                    "name": "ga:date",
                },
                {
                    "name": "ga:searchKeyword",
                }]
            }]
        }
        if page_token is not None:
            body['reportRequests'][0]['pageToken'] = page_token
        
        try:
            if verbose:
                print('Making api request:\n{}'.format(json.dumps(body, indent=2)))
            response = self.api.reports().batchGet(body=body).execute()
        except Exception as e:
            print('Unable to execute API call. See debug info below.')
            print('date_start = {}'.format(start_date))
            print('date_end = {}'.format(end_date))
            print('error = {}'.format(e))
            if self.raise_errors:
                raise e
            else:
                return pd.DataFrame(), None

        try:
            print('Parsing response')
            df = self.parse_onsite_search_resp(response)
            next_page_token = self.parse_next_page_token(response)
        except Exception as e:
            print('Unable to parse API response. See debug info below.')
            print('date_start = {}'.format(start_date))
            print('date_end = {}'.format(end_date))
            print('response = {}'.format(response))
            print('error = {}'.format(e))
            if self.raise_errors or raise_errors:
                raise e
            else:
                return pd.DataFrame(), None
            
        return df, next_page_token

    @staticmethod
    def parse_next_page_token(response) -> pd.DataFrame:
        return response.get('reports', [{}])[0].get('nextPageToken', None)
    
    @staticmethod
    def parse_onsite_search_resp(response) -> pd.DataFrame:
        cols = ['date', 'search_term', 'num_searches']
        
        rows = response.get('reports', [{}])[0].get('data', {}).get('rows', [])
        data = []
        for i, row in enumerate(rows):
            d = []

            dims = row.get('dimensions', [])
            if len(dims) == 2:
                d += dims
            else:
                raise Exception('Warning, response parse failed as len(dim) != 2 for row = {} (#{})'.format(row, i))

            vals = row.get('metrics', [{}])[0].get('values', [])
            if len(vals) == 1:
                d += vals
            else:
                raise Exception('Warning, response parse failed as len(vals) != 1 for row = {} (#{})'.format(row, i))

            data.append(d)

        if len(data) == 0:
            print('Parsed 0 rows from API response')
            return pd.DataFrame(columns=cols)
        
        df = pd.DataFrame(data, columns=cols)
        try:
            df.date = pd.to_datetime(df.date)
        except:
            print('Warning, unable to parse date.\ndf.head() =\n{}'.format(df.head()))
            print('Attempting to parse with method `coerce` and dropping NaTs')
            len_0 = len(df)
            df.date = pd.to_datetime(df.date, errors='coerce')
            m_drop = df.date.isnull()
            df = df.loc[~m_drop]
            num_dropped = len_0 - len(df)
            print('Dropped {:,.0f} rows ({:.1f}%)'.format(num_dropped, num_dropped / len(df) * 100))

        df.num_searches = df.num_searches.astype(int)
        print('Parsed {} rows from API response'.format(len(df)))
        
        return df


api = GoogleAnalyticsApi()


month_date_pairs = [('{}-01'.format(x.strftime('%Y-%m')), x.strftime('%Y-%m-%d'))
 for x in pd.date_range(start='2017-10-01', periods=20, freq='M').tolist()]
month_date_pairs


# Let's test it out...

dfs = []
i = 0
for start_date, end_date in month_date_pairs:
    i += 1
    dfs.append(api.query_onsite_search(start_date, end_date, api_delay=0, verbose=True, test=True))
    if i ==2:
        break

df_search_terms = pd.concat(dfs)


df_search_terms.head(15)


df_search_terms.tail(15)


# Went back and added verbose flag, and sleep in between calls
# 
# Now pulling the data we want

from tqdm import tqdm_notebook


dfs = []
for start_date, end_date in tqdm_notebook(month_date_pairs):
    dfs.append(api.query_onsite_search(start_date, end_date, api_delay=1))

df_search_terms = pd.concat(dfs)


# Dump raw result data

os.getenv('DATA_PATH')


f_path = os.path.join(os.getenv('DATA_PATH'), 'interim', 'onsite_search_terms_2017_2019_raw')
if os.path.isfile('{}.csv'.format(f_path)) or os.path.isfile('{}.pkl'.format(f_path)):
    raise Exception('File at {} already exists. Will not overwrite.'.format(f_path))

df_search_terms.to_csv('{}.csv'.format(f_path), index=False)
df_search_terms.to_pickle('{}.pkl'.format(f_path))


# ### Load data from file

f_path = os.path.join(os.getenv('DATA_PATH'), 'interim', 'onsite_search_terms_2017_2019_raw')
df_search_terms = pd.read_pickle('{}.pkl'.format(f_path))
df_search_terms['search_term_lower'] = df_search_terms.search_term.str.strip().str.lower()


df_search_terms.head()


len(df_search_terms)


df_search_terms.groupby('date').num_searches.sum().plot()


# We were able to pull over 10M onsite searches!

# ## Data Modeling

# ### Wordclouds

from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator


get_ipython().run_line_magic('pinfo', 'WordCloud')


def make_wordcloud(data, title=None, colormap='cividis'):
    """
    Generate a wordcloud.
    
    data : dict
        Term frequencies.
    """
    wordcloud = WordCloud(
        background_color='white',
        colormap=colormap,
        stopwords=STOPWORDS,
        max_words=200,
        max_font_size=80,
        scale=10,
        width=800,
        height=400,
        random_state=19, # just used to set color I think
    ).generate_from_frequencies(data)

    fig = plt.figure(1, figsize=(12, 12))
    plt.axis('off')
    if title is not None:
        fig.suptitle(title, fontsize=20)
        fig.subplots_adjust(top=2.3)

    plt.imshow(wordcloud)
    return fig


m = df_search_terms.date.apply(lambda x: x.strftime('%Y')) == '2017'
term_counts = df_search_terms[m].groupby('search_term_lower').num_searches.sum()
fig = make_wordcloud(term_counts.to_dict(), '2017')
savefig('onsite_search_tag_cloud_2017')
plt.show()


m = df_search_terms.date.apply(lambda x: x.strftime('%Y')) == '2018'
term_counts = df_search_terms[m].set_index('search_term').num_searches
fig = make_wordcloud(term_counts.to_dict(), '2018')
savefig('onsite_search_tag_cloud_2018')
plt.show()


s = df_search_terms.date.apply(lambda x: x.month)
# Filter on Nov - Feb
m = ((s >= 11) & (s <= 12)) | ((s >= 1) & (s <= 2))
winter_term_counts = df_search_terms[m].groupby('search_term_lower').num_searches.sum()
fig = make_wordcloud(winter_term_counts.to_dict(), 'Winter', colormap='winter')
savefig('onsite_search_tag_cloud_winter')


s = df_search_terms.date.apply(lambda x: x.month)
# Filter on May - Aug
m = ((s >= 5) & (s <= 8))
summer_term_counts = df_search_terms[m].groupby('search_term_lower').num_searches.sum()
fig = make_wordcloud(summer_term_counts.to_dict(), 'Summer', colormap='summer')
savefig('onsite_search_tag_cloud_summer')


# Top 10 winter searches

winter_term_counts.sort_values(ascending=False).head(25)


# Top 10 summer searches

summer_term_counts.sort_values(ascending=False).head(25)


summer_terms = (
    set(summer_term_counts.sort_values(ascending=False).head(25).index)
    - set(winter_term_counts.sort_values(ascending=False).head(25).index)
)
top_summer_terms = summer_term_counts[summer_term_counts.index.isin(list(summer_terms))]
top_summer_terms.sort_values(ascending=False)


fig = plt.figure(figsize=(4, 8))
top_summer_terms.sort_values(ascending=False).head(10).iloc[::-1].plot.barh(color='g')
plt.ylabel('Top 10 Summer Searches')
ax = plt.gca()
ax.set_xticklabels([])
savefig('onsite_search_top_summer')


winter_terms = (
    set(winter_term_counts.sort_values(ascending=False).head(25).index)
    - set(summer_term_counts.sort_values(ascending=False).head(25).index)
)
top_winter_terms = winter_term_counts[winter_term_counts.index.isin(list(winter_terms))]
top_winter_terms.sort_values(ascending=False)


fig = plt.figure(figsize=(4, 8))
top_winter_terms.sort_values(ascending=False).head(10).iloc[::-1].plot.barh(color='b')
plt.ylabel('Top 10 Winter Searches')
ax = plt.gca()
ax.set_xticklabels([])
savefig('onsite_search_top_winter')


# ### Topic Modeling

import spacy
# Use spaCy instead of NLTK
# To run below, first do: python -m spacy download en

nlp = spacy.load('en')

my_stop_words = ['i', 'a', 'about', 'an', 'are', 'as', 'at', 'be', 'by',
                 'for', 'from', 'how', 'in', 'is', 'it', 'of', 'on', 'or',
                 'that', 'the', 'this', 'to', 'was', 'what', 'when', 'where',
                 'who', 'will', 'with', 'the']
for stopword in my_stop_words:
    lexeme = nlp.vocab[stopword]
    lexeme.is_stop = True


df_search_terms.head()


from tqdm import tqdm


def make_lemmas(text):
    nlp_text = nlp(text)
    lemmas = [w.lemma_ for w in nlp_text
              if not any((w.is_stop, w.is_punct, w.like_num))]
    return lemmas

tqdm.pandas()
search_terms = df_search_terms['search_term_lower'].drop_duplicates()
search_term_lemmas = search_terms.progress_apply(make_lemmas)
search_term_lemmas.name = 'lemmas'
s = pd.concat((search_terms, search_term_lemmas), axis=1).set_index('search_term_lower')['lemmas']

m = pd.Series(True, index=df_search_terms.index)
df_search_terms['lemmas'] = float('nan')
df_search_terms.loc[m, 'lemmas']     = df_search_terms.loc[m, 'search_term_lower'].map(s)


# For a larger dataset, would want to run cell above externally

# Dump terms to file
# f_path = os.path.join(os.getenv('DATA_PATH'), 'interim', 'onsite_search_lower_terms_dedupe.csv')
# df_search_terms.search_term_lower.drop_duplicates().to_csv(f_path, index=False)

# Run script externally....

# Load results from file


import gensim


m_not_empty = df_search_terms.lemmas.apply(lambda x: len(x) > 0)

# Extract bigrams
tqdm.pandas()
bigram = gensim.models.Phrases(
    df_search_terms.loc[m_not_empty, 'lemmas'].tolist()
)
df_search_terms['bigrams'] = float('nan')
df_search_terms.loc[m_not_empty, 'bigrams'] = (
    df_search_terms.loc[m_not_empty, 'lemmas']
    .progress_apply(lambda x: bigram[x])
)


# Create integer corpus for learning algorithms
dictionary = gensim.corpora.Dictionary(df_search_terms.bigrams.dropna().tolist())
df_search_terms['corpus'] = float('nan')
df_search_terms.loc[m_not_empty, 'corpus'] = (
    df_search_terms.loc[m_not_empty, 'bigrams']
    .progress_apply(lambda x: dictionary.doc2bow(x))
)


# Dump results of long computation

import pickle

f_path = os.path.join(os.getenv('DATA_PATH'), 'interim', 'onsite_search_nlp_gensim_dictionary.pkl')
# if os.path.isfile(f_path):
#     raise Exception(
#         'File exists! ({}) Run line below in separate cell to overwrite it. '
#         'Otherwise just run cell below to load file.'.format(f_path))

with open(f_path, 'wb') as f:
    pickle.dump(dictionary, f)

f_path = os.path.join(os.getenv('DATA_PATH'), 'interim', 'onsite_search_terms_2017_2019_nlp')
# if os.path.isfile('{}.csv'.format(f_path)):
#     raise Exception(
#         'File exists! ({}) Run line below in separate cell to overwrite it. '
#         'Otherwise just run cell below to load file.'.format(f_path))
    
df_search_terms.to_csv('{}.csv'.format(f_path), index=False)
df_search_terms.to_pickle('{}.pkl'.format(f_path))


# Ignore this cell

# tmp = df_search_terms.copy()
# tmp_dict = {**dictionary}

# del tmp
# del tmp_dict


# Load data

f_path = os.path.join(os.getenv('DATA_PATH'), 'interim', 'onsite_search_nlp_gensim_dictionary.pkl')
import pickle
with open(f_path, 'rb') as f:
    dictionary = pickle.load(f)

f_path = os.path.join(os.getenv('DATA_PATH'), 'interim', 'onsite_search_terms_2017_2019_nlp.pkl')
df_search_terms = pd.read_pickle(f_path)


dictionary


df_search_terms.head()


# Latent Semantic Indexing (LSI) 

# from gensim.models import LsiModel
# lsimodel = LsiModel(
#     corpus=df_search_terms.corpus.dropna().tolist(),
#     num_topics=10,
#     id2word=dictionary,
# )


# HDP Model

# from gensim.models import HdpModel
# hdpmodel = HdpModel(
#     corpus=df_search_terms.corpus.dropna().tolist(),
#     id2word=dictionary,
# )


# Note: default behaviour for Gensim e.g.
# > 2019-06-05 15:31:34,007 : INFO : running online (single-pass) LDA training, 5 topics, 1 passes over the supplied corpus of 8585316 documents, updating model once every 2000 documents, evaluating perplexity every 20000 documents, iterating 50x with a convergence threshold of 0.001000

# Log to file (you'll probably want to delete this after)

import logging
f_path = os.path.join(os.getenv('DATA_PATH'), 'interim', 'onsite_search_gensim.log')
logging.basicConfig(filename=f_path, format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


# Running a test first...

# Latent Dirichlet Allocation (LDA) model

from gensim.models import LdaModel
ldamodel = LdaModel(
    corpus=df_search_terms.corpus.dropna().tolist()[:10],
    num_topics=5,
    id2word=dictionary,
)


f_path


cat '/Volumes/GoogleDrive/My Drive/ga-data-mining/data/interim/onsite_search_gensim.log'


# Training on the full dataset (running externally `onsite_search_gensim_lda.py`)

# Latent Dirichlet Allocation (LDA) model

# from gensim.models import LdaModel
# ldamodel = LdaModel(
#     corpus=df_search_terms.corpus.dropna().tolist(),
#     num_topics=5,
#     id2word=dictionary,
# )


# Save the model

# f_path = '../../models/onsite_search_terms_lda_2017_2019_5_topic.model'
# if os.path.isfile(f_path):
#     raise Exception(
#         'File exists! ({}) Run line below in separate cell to overwrite it. '
#         'Otherwise just run cell below to load file.'.format(f_path))

# ldamodel.save(f_path)


# Load the model

import gensim
f_path = '../../models/onsite_search_terms_lda_2017_2019_20_topic.model'
ldamodel = gensim.models.LdaModel.load(f_path)


# Model visualization

import pyLDAvis.gensim
pyLDAvis.enable_notebook()
pyLDAvis.gensim.prepare(
    ldamodel,
    df_search_terms.corpus.dropna().tolist(),
    dictionary
)


# Export as web app

pyLDAvis.disable_notebook()
p = pyLDAvis.gensim.prepare(
    ldamodel,
    df_search_terms.corpus.dropna().tolist(),
    dictionary
)
pyLDAvis.save_html(p, '../../results/reports/onsite_search_terms_lda_2017_2019_20_topic.html')


# The size of the bubble measures the importance of the topics, relative to the data.
# 
# The terms are ordered by saliency (how much the term tells you about the topic).
# 
# The relevance slider can be used to adjust saliency scores.

num_topics = 20


censored = [9]
[x for x in ldamodel.show_topics(num_topics=num_topics) if x[0] not in censored]


ldamodel_topic_words = {i: list(re.findall(r'\*"([^"]+)"', v)) for i, v in ldamodel.show_topics(num_topics=num_topics)}
censored = [9]
{k: v for k, v in ldamodel_topic_words.items() if k not in censored}


manual_topic_map = {
    4: "Jeans",
    5: "Handbags",
    7: "Jeans",
    10: "Short Sleve Tops",
    11: "Exotic Styles",
    12: "Long Sleve Tops",
    15: "Dresses",
}


# ### Tracking trends over time
# 
# Given a gensim model, label a corpus by topic and plot them over time. How do they change relative to one another?
# 
# Top topics may follow similar trends to global search patterns. Instead, look at "% of searches that are topic".

# First, we need to label the training data

from tqdm import tqdm


tqdm.pandas()

m = ~(df_search_terms.corpus.isnull())
df_search_terms['lda_20_topic_scores'] = float('nan')
df_search_terms.loc[m, 'lda_20_topic_scores'] = (
    df_search_terms.loc[m, 'corpus']
    .progress_apply(lambda x: ldamodel[x])
)


df_search_terms.head()


m = ~(df_search_terms['lda_20_topic_scores'].isnull())

df_search_terms['lda_20_topic'] = float('nan')
df_search_terms.loc[m, 'lda_20_topic'] = (
    df_search_terms.loc[m, 'lda_20_topic_scores']
    .progress_apply(lambda x: sorted(x, key=lambda x: x[1], reverse=True)[0][0])
)

df_search_terms['lda_20_topic_score'] = float('nan')
df_search_terms.loc[m, 'lda_20_topic_score'] = (
    df_search_terms.loc[m, 'lda_20_topic_scores']
    .progress_apply(lambda x: sorted(x, key=lambda x: x[1], reverse=True)[0][1])
)


pd.options.display.max_colwidth = 100


df_search_terms.sample(10)


df_search_terms.lda_20_topic_score.plot.hist(bins=20)


# Let's throw out anything less than 0.5

m = df_search_terms['lda_20_topic_score'].fillna(0) >= 0.5

df_search_terms['topic'] = float('nan')
df_search_terms.loc[m, 'topic'] = df_search_terms.loc[m, 'lda_20_topic']


df_search_terms['topic_name'] = float('nan')
df_search_terms.loc[m, 'topic_name'] = df_search_terms.loc[m, 'topic'].map(manual_topic_map)


# Dump results of long computation

def dump_df(df, name, csv=True, pkl=True):
    if not any((csv, pkl)):
        print('Not saving to file')
    
    f_path = os.path.join(os.getenv('DATA_PATH'), 'interim', name)
    if os.path.isfile('{}.csv'.format(f_path)):
        raise Exception(
            'File exists! ({}) Run line below in separate cell to overwrite it. '
            'Otherwise just run cell below to load file.'.format(f_path))

    if csv:
        file = '{}.csv'.format(f_path)
        print('Writing to {}'.format(file))
        df_search_terms.to_csv(file, index=False)
        
    if pkl:
        file = '{}.pkl'.format(f_path)
        print('Writing to {}'.format(file))
        df_search_terms.to_pickle(file)
    
    
name = 'onsite_search_terms_2017_2019_nlp_topic_labels_'
dump_df(df_search_terms, name, csv=False)


# Load data

name = 'onsite_search_terms_2017_2019_nlp_topic_labels'
f_path = os.path.join(os.getenv('DATA_PATH'), 'interim', '{}.csv'.format(name))
df_search_terms = pd.read_csv(f_path)

df_search_terms['date'] = pd.to_datetime(
    df_search_terms['date']
)


df_search_terms.sample(5)


# We have labels (given our confidence cutoff of 0.5) for about half of the rows

df_search_terms.topic.isnull().sum() / df_search_terms.shape[0]


# Now we have topic labels, let's generate the charts.

topics = df_search_terms.topic.value_counts().index.sort_values().tolist()
topics


fig, ax = plt.subplots()
for topic in topics:
    m = df_search_terms.topic == topic
    df_search_terms[m].groupby('date').num_searches.sum().plot(ax=ax, label='topic {:.0f}'.format(topic))
    plt.legend()


# Plot by week

fig, ax = plt.subplots()
for topic in topics:
    m = df_search_terms.topic == topic
    s = df_search_terms[m].groupby(pd.Grouper(key='date', freq='W-MON')).num_searches.sum().dropna()
    s.plot(ax=ax, label='topic {:.0f}'.format(topic))
    plt.legend()


# Now look at the ratios...

# Being careful to handle the cases where there might not be a search in a given week, we use `pd.date_range` to generate a canvas and post the data onto it.

def plot_trends(
    df,
    output_name='onsite_search_topic_relative_trends',
    groupby='week',
):
    if groupby == 'week':
        freq = '7D'
        grouper_freq = 'W-MON'
    elif groupby == 'month':
        freq = 'M'
        grouper_freq = 'M'
    else:
        raise ValueError('Invalid option for groupby ({})'.format(groupby))

    s = df.groupby(pd.Grouper(key='date', freq=grouper_freq)).size().index.sort_values()
    min_day = s.min()
    max_day = s.max()
    s_dates = pd.date_range(
        start=min_day,
        end=max_day,
        freq=freq,
    )
    canvas_post = lambda x: pd.DataFrame(s_dates, columns=['date']).set_index('date').join(x, how='outer')['num_searches']
    s_total = df.groupby(pd.Grouper(key='date', freq=grouper_freq)).num_searches.sum()
    s_total = canvas_post(s_total)

    fig, ax = plt.subplots()
    for topic in topics:
        m = df.topic == topic
        s = df[m].groupby(pd.Grouper(key='date', freq=grouper_freq)).num_searches.sum()
        # Post the data to the canvas
        s = (canvas_post(s) / s_total)
        s.dropna().plot(ax=ax, label='topic {:.0f}'.format(topic), lw=2)
        plt.legend()

    plt.xlabel('Date')
    plt.ylabel('Onsite Search Topic Frequency')
    savefig(output_name)


# m = (df_search_terms.date < '2018-05-01') | (df_search_terms.date > '2018-08-01')
plot_trends(df_search_terms)


plot_trends(df_search_terms, groupby='month')


s = df_search_terms.groupby(pd.Grouper(key='date', freq='W-MON')).size()
s.sort_index()


# Looking at the counts above, we see that we're missing data for 2018-06 :(
# 
# ```
# 2018-05-21    168697
# 2018-05-28    169620
# 2018-06-04     10400
# 2018-06-11     23104
# 2018-06-18    153420
# 2018-06-25    126540
# ```

m = (df_search_terms.date < '2018-06-01') | (df_search_terms.date > '2018-07-01')
plot_trends(df_search_terms[m], groupby='month')


# This is very close to the final result I'm looking for.
# 
# Now I need to add manual labels where it makes sense and look for trends.

ldamodel_topic_words = {i: list(re.findall(r'\*"([^"]+)"', v)) for i, v in ldamodel.show_topics(num_topics=20)}
censored = [9]
{k: v for k, v in ldamodel_topic_words.items() if k not in censored}


manual_topic_map = {
    4: "Jeans",
    5: "Handbags",
    7: "Jeans",
    10: "Short Sleeve Tops",
    11: "Exotic Styles",
    12: "Long Sleeve Tops",
    15: "Dresses",
}


df_search_terms['topic_name'] = float('nan')
m = ~(df_search_terms.topic.isnull())
df_search_terms.loc[m, 'topic_name'] = df_search_terms.loc[m, 'topic'].map(manual_topic_map)


topic_names = df_search_terms.topic_name.value_counts().index.sort_values().tolist()
topic_names


def plot_trends(
    df,
    topic_names,
    output_name='onsite_search_topic_relative_trends',
    groupby='week',
    filter_on_topics=[],
):
    if groupby not in (('day', 'week', 'month')):
        raise ValueError('Invalid option for groupby ({})'.format(groupby))

    elif groupby == 'day':
        freq = 'D'
        grouper_freq = 'D'
        
    elif groupby == 'week':
        freq = '7D'
        grouper_freq = 'W-MON'

    elif groupby == 'month':
        freq = 'M'
        grouper_freq = 'M'


    s = df.groupby(pd.Grouper(key='date', freq=grouper_freq)).size().index.sort_values()
    min_day = s.min()
    max_day = s.max()
    s_dates = pd.date_range(
        start=min_day,
        end=max_day,
        freq=freq,
    )
    canvas_post = lambda x: pd.DataFrame(s_dates, columns=['date']).set_index('date').join(x, how='outer')['num_searches']
    s_total = df.groupby(pd.Grouper(key='date', freq=grouper_freq)).num_searches.sum()
    s_total = canvas_post(s_total)

    # Save results for later
    df_topic_freq = pd.DataFrame()
    
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color'] * 10 * 10
    fig, ax = plt.subplots()
    for topic_name, c in zip(topic_names, colors):
        m = df.topic_name == topic_name
        s = df[m].groupby(pd.Grouper(key='date', freq=grouper_freq)).num_searches.sum()
        s = (canvas_post(s) / s_total) # Post the data to the canvas

        # Save for later
        s.name = topic_name
        df_topic_freq = pd.concat((df_topic_freq, s), axis=1)

        # Plot it
        alpha = 1
        if filter_on_topics:
            if topic_name not in filter_on_topics:
                alpha = 0.2
        s.dropna().plot(ax=ax, label=topic_name, lw=2, color=c, alpha=alpha)
#         if filter_on_topics:
#             if topic_name not in filter_on_topics:
#                 ax.lines[-1].set_visible(False)
            
        plt.legend()

    plt.xlabel('Date')
    plt.ylabel('Onsite Search Topic Frequency')
    if output_name:
        savefig(output_name)
    else:
        print('No output name, not saving fig')
    
    return df_topic_freq


plot_trends(
    df_search_terms,
    topic_names,
);


m = (df_search_terms.date < '2018-06-01') | (df_search_terms.date > '2018-07-01')
df_topic_freq = plot_trends(
    df_search_terms[m],
    topic_names,
)


m = (df_search_terms.date < '2018-06-01') | (df_search_terms.date > '2018-07-01')
df_topic_freq = plot_trends(
    df_search_terms[m],
    topic_names,
    output_name='onsite_search_topic_relative_trends_filter_1',
    filter_on_topics=['Dresses', 'Exotic Styles'],
)


m = (df_search_terms.date < '2018-06-01') | (df_search_terms.date > '2018-07-01')
df_topic_freq = plot_trends(
    df_search_terms[m],
    topic_names,
    output_name='onsite_search_topic_relative_trends_filter_2',
    filter_on_topics=['Jeans'],
)


m = (df_search_terms.date < '2018-06-01') | (df_search_terms.date > '2018-07-01')

df_topic_freq = plot_trends(
    df_search_terms[m],
    topic_names,
    output_name='',
    groupby='day',
)

df_topic_freq_daily = df_topic_freq.reset_index().melt(
    id_vars=['date'],
    value_vars=df_topic_freq.columns.tolist()
).rename(columns={'variable': 'topic', 'value': 'num_searches'})


df_topic_freq_daily.sample(5)


m = (df_search_terms.date < '2018-06-01') | (df_search_terms.date > '2018-07-01')

df_topic_freq = plot_trends(
    df_search_terms[m],
    topic_names,
    output_name='',
    groupby='week',
)

df_topic_freq_weekly = df_topic_freq.reset_index().melt(
    id_vars=['date'],
    value_vars=df_topic_freq.columns.tolist()
).rename(columns={'variable': 'topic', 'value': 'num_searches'})


df_topic_freq_weekly.sample(5)


m = (df_search_terms.date < '2018-06-01') | (df_search_terms.date > '2018-07-01')

df_topic_freq = plot_trends(
    df_search_terms[m],
    topic_names,
    output_name='',
    groupby='month',
)

df_topic_freq_monthly = df_topic_freq.reset_index().melt(
    id_vars=['date'],
    value_vars=df_topic_freq.columns.tolist()
).rename(columns={'variable': 'topic', 'value': 'num_searches'})


df_topic_freq_monthly.sample(5)


# ### Modeling trend frequency

from fbprophet import Prophet
import warnings
# Ignore warnings from prophet lib
warnings.filterwarnings('ignore', 'Conversion of the second argument of issubdtype')


def build_prophet_df(df, segment_col, value_col='num_searches', date_col='date'):
    df_prophet = (
        df.groupby([date_col, segment_col])[value_col].sum()
        .reset_index()
        .rename(columns={value_col: 'y'})
    )
    df_prophet['ds'] = df_prophet[date_col].apply(lambda x: x.strftime('%Y-%m-%d'))
    df_prophet = df_prophet.sort_values(date_col, ascending=True)
    return df_prophet

def segment_forecast(
    df,
    segment_col,
    groupby='day',
    segments=None,
    max_num_segments=6,
    daily_seasonality=False,
    weekly_seasonality=2,
    yearly_seasonality=10,
    seasonality_prior_scale=0.1,
    y_label='Search Frequency',
    plot_trends=True,
):
    if groupby not in (('day', 'week', 'month')):
        raise ValueError('Bad value for groupby ({})'.format(groupby))
        
    elif groupby == 'day':
        freq = 'D'
        periods = 365
        
    elif groupby == 'week':
        freq = '7D'
        periods = 52
        
    elif groupby == 'month':
        freq = 'M'        
        periods = 12
    
    df_prophet = build_prophet_df(df, segment_col)
    
    # Get top segments (by sum of target variable)
    if not segments:
        segments = (
            df_prophet.groupby(segment_col)['y'].sum()
            .sort_values(ascending=False)
            .head(max_num_segments).index.tolist()
        )
    models = []
    forecasts = []
    for seg in segments:
        m = Prophet(
            daily_seasonality=daily_seasonality,
            weekly_seasonality=weekly_seasonality,
            yearly_seasonality=yearly_seasonality,
            seasonality_prior_scale=seasonality_prior_scale,
        )
        m.fit(df_prophet[df_prophet[segment_col] == seg])
        models.append(m)
        
        future = m.make_future_dataframe(periods=periods, freq=freq)
        forecast = m.predict(future)
        forecasts.append(forecast)
        
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color'] * 10 * 10
    fig, axes = plt.subplots(len(forecasts), sharex=True)
    for i, (ax, seg, forecast, c) in enumerate(zip(axes, segments, forecasts, colors)):
        forecast.set_index('ds')['yhat'].plot(label=seg, ax=ax, color=c)
        (df_prophet[df_prophet[segment_col] == seg]
         .set_index('date')['y'].plot(label='_', ax=ax, color=c, marker='o', linewidth=0))
        ax.legend()
        if i == int(len(forecasts) / 2):
            ax.set_ylabel(y_label)
        else:
            ax.set_ylabel('')

    plt.xlabel('')
    savefig('onsite_search_{}_{}_forecast'.format(
        '-'.join(segment_col.split(' ')),
        groupby,
    ))
    plt.show()

    if plot_trends:
        for seg, m, forecast, c in zip(segments, models, forecasts, colors):
            fig = m.plot_components(forecast)
            
            fig.axes[0].set_xlabel('{} - Date'.format(seg.title()))
            fig.axes[0].set_ylabel('Overall Trend')
            fig.axes[0].lines[0].set_color(c)
            
            fig.axes[1].set_xlabel('{} - Day of Week'.format(seg.title()))
            fig.axes[1].set_ylabel('Weekly Trend')
            fig.axes[1].lines[0].set_color(c)
            
            fig.axes[2].set_xlabel('{} - Day of Year'.format(seg.title()))
            fig.axes[2].set_ylabel('Yearly Trend')
            fig.axes[2].lines[0].set_color(c)
    
            savefig('onsite_search_{}={}_{}_forecast_trends'.format(
                '-'.join(segment_col.split(' ')),
                '-'.join(seg.split(' ')),
                groupby,
            ))
            plt.show()


df_topic_freq_daily.sample(5)


print('Warning, padding values (avoid this in the future)')
segment_forecast(
    df_topic_freq_daily.fillna(method='pad'),
    segment_col='topic',
    groupby='day',
    segments=topic_names,
)


print('Warning, padding values (avoid this in the future)')
segment_forecast(
    df_topic_freq_weekly.fillna(method='pad'),
    segment_col='topic',
    groupby='week',
    segments=topic_names,
    plot_trends=False,
)


print('Warning, padding values (avoid this in the future)')
segment_forecast(
    df_topic_freq_monthly.fillna(method='pad'),
    segment_col='topic',
    groupby='month',
    segments=topic_names,
    plot_trends=False,
)


# **How can this be applied?**
# 
# - Use trend patterns (seasonal and non-seasonal) to inform marketing strategy.
# - Label users (realtime) who use onsite search. Curate experience (e.g. banner ads, recommended items) based on topic grouping.
# 
# **Similar research ideas**
# 
# - Apply topic labels to sessions and train predictive model on many features (e.g. region, browser, source, pages per session, etc...). Use interpretation library like [Lime](https://github.com/marcotcr/lime) to understand these user groups better.
# - In a semi-structured approach, manually label a set of onsite searches (e.g. 1k rows), train model and apply predictions to remaining dataset. Pros: topic groups are easier to interpret. Cons: time spent labelling data, must be careful to spotcheck results.

# ### Compare with Google Trends

trends_data = [row.split('\t') for row in '''
Week	long sleeve tops: (United States)	short sleeve tops: (United States)
2014-06-22	14	20
2014-06-29	19	11
2014-07-06	17	10
2014-07-13	26	14
2014-07-20	21	13
2014-07-27	17	13
2014-08-03	22	8
2014-08-10	23	14
2014-08-17	35	10
2014-08-24	26	6
2014-08-31	25	11
2014-09-07	24	6
2014-09-14	31	7
2014-09-21	42	7
2014-09-28	41	8
2014-10-05	36	7
2014-10-12	47	7
2014-10-19	45	11
2014-10-26	48	8
2014-11-02	49	5
2014-11-09	41	7
2014-11-16	51	5
2014-11-23	53	12
2014-11-30	39	10
2014-12-07	47	9
2014-12-14	42	9
2014-12-21	31	6
2014-12-28	30	11
2015-01-04	31	5
2015-01-11	30	7
2015-01-18	35	12
2015-01-25	29	8
2015-02-01	25	11
2015-02-08	34	13
2015-02-15	38	12
2015-02-22	32	11
2015-03-01	20	17
2015-03-08	27	15
2015-03-15	26	21
2015-03-22	17	17
2015-03-29	19	12
2015-04-05	21	19
2015-04-12	25	20
2015-04-19	23	17
2015-04-26	14	10
2015-05-03	26	18
2015-05-10	25	23
2015-05-17	27	11
2015-05-24	36	19
2015-05-31	18	23
2015-06-07	15	19
2015-06-14	24	14
2015-06-21	17	18
2015-06-28	15	16
2015-07-05	18	13
2015-07-12	18	13
2015-07-19	23	18
2015-07-26	20	19
2015-08-02	25	4
2015-08-09	23	9
2015-08-16	35	15
2015-08-23	31	9
2015-08-30	40	12
2015-09-06	29	11
2015-09-13	40	13
2015-09-20	34	10
2015-09-27	56	14
2015-10-04	55	8
2015-10-11	46	7
2015-10-18	58	9
2015-10-25	35	10
2015-11-01	42	10
2015-11-08	46	6
2015-11-15	45	7
2015-11-22	42	11
2015-11-29	54	12
2015-12-06	51	12
2015-12-13	39	10
2015-12-20	24	7
2015-12-27	50	12
2016-01-03	36	13
2016-01-10	35	14
2016-01-17	63	6
2016-01-24	51	5
2016-01-31	45	17
2016-02-07	53	21
2016-02-14	33	23
2016-02-21	31	24
2016-02-28	42	20
2016-03-06	31	35
2016-03-13	38	24
2016-03-20	36	17
2016-03-27	36	23
2016-04-03	34	25
2016-04-10	28	34
2016-04-17	39	31
2016-04-24	34	23
2016-05-01	34	19
2016-05-08	27	34
2016-05-15	42	20
2016-05-22	35	26
2016-05-29	38	33
2016-06-05	30	28
2016-06-12	35	30
2016-06-19	32	18
2016-06-26	33	22
2016-07-03	29	28
2016-07-10	36	18
2016-07-17	29	19
2016-07-24	26	24
2016-07-31	49	20
2016-08-07	37	16
2016-08-14	36	22
2016-08-21	35	10
2016-08-28	47	15
2016-09-04	47	11
2016-09-11	55	16
2016-09-18	45	14
2016-09-25	57	17
2016-10-02	64	13
2016-10-09	69	17
2016-10-16	63	12
2016-10-23	76	7
2016-10-30	56	9
2016-11-06	63	9
2016-11-13	62	7
2016-11-20	78	20
2016-11-27	79	13
2016-12-04	54	18
2016-12-11	58	9
2016-12-18	53	11
2016-12-25	64	10
2017-01-01	50	12
2017-01-08	49	6
2017-01-15	63	16
2017-01-22	44	13
2017-01-29	51	25
2017-02-05	52	20
2017-02-12	49	23
2017-02-19	47	19
2017-02-26	44	22
2017-03-05	46	20
2017-03-12	47	24
2017-03-19	49	23
2017-03-26	45	35
2017-04-02	47	30
2017-04-09	43	29
2017-04-16	39	32
2017-04-23	46	27
2017-04-30	43	25
2017-05-07	48	15
2017-05-14	40	19
2017-05-21	26	24
2017-05-28	39	23
2017-06-04	47	24
2017-06-11	33	26
2017-06-18	29	25
2017-06-25	41	24
2017-07-02	38	19
2017-07-09	63	23
2017-07-16	45	19
2017-07-23	68	22
2017-07-30	47	18
2017-08-06	52	18
2017-08-13	44	19
2017-08-20	54	16
2017-08-27	51	23
2017-09-03	66	19
2017-09-10	78	20
2017-09-17	61	18
2017-09-24	64	21
2017-10-01	82	18
2017-10-08	85	19
2017-10-15	92	22
2017-10-22	79	14
2017-10-29	73	11
2017-11-05	93	17
2017-11-12	90	16
2017-11-19	95	12
2017-11-26	87	19
2017-12-03	85	18
2017-12-10	73	24
2017-12-17	69	13
2017-12-24	64	17
2017-12-31	59	12
2018-01-07	66	15
2018-01-14	60	16
2018-01-21	39	16
2018-01-28	61	18
2018-02-04	56	29
2018-02-11	67	29
2018-02-18	62	30
2018-02-25	50	23
2018-03-04	54	25
2018-03-11	53	20
2018-03-18	41	22
2018-03-25	53	23
2018-04-01	35	30
2018-04-08	51	32
2018-04-15	55	29
2018-04-22	56	34
2018-04-29	49	49
2018-05-06	46	49
2018-05-13	49	43
2018-05-20	44	38
2018-05-27	47	51
2018-06-03	65	30
2018-06-10	45	33
2018-06-17	43	34
2018-06-24	41	34
2018-07-01	49	34
2018-07-08	41	31
2018-07-15	40	26
2018-07-22	43	31
2018-07-29	44	33
2018-08-05	68	31
2018-08-12	55	24
2018-08-19	72	21
2018-08-26	65	21
2018-09-02	48	25
2018-09-09	80	25
2018-09-16	64	20
2018-09-23	85	25
2018-09-30	80	16
2018-10-07	82	29
2018-10-14	90	13
2018-10-21	100	18
2018-10-28	69	9
2018-11-04	94	18
2018-11-11	95	18
2018-11-18	86	19
2018-11-25	91	16
2018-12-02	92	25
2018-12-09	93	17
2018-12-16	62	15
2018-12-23	67	12
2018-12-30	66	25
2019-01-06	64	16
2019-01-13	71	18
2019-01-20	70	20
2019-01-27	61	20
2019-02-03	60	22
2019-02-10	66	19
2019-02-17	60	16
2019-02-24	53	24
2019-03-03	55	34
2019-03-10	39	35
2019-03-17	50	27
2019-03-24	50	27
2019-03-31	56	41
2019-04-07	53	56
2019-04-14	61	38
2019-04-21	48	32
2019-04-28	48	33
2019-05-05	46	47
2019-05-12	50	26
2019-05-19	55	50
2019-05-26	33	30
2019-06-02	51	38
2019-06-09	43	34
2019-06-16	54	13
'''.split('\n') if row.strip()]


df_trends = pd.DataFrame(trends_data[1:], columns=trends_data[0])

trend_cols = ['Google Trend: Long Sleeve Tops', 'Google Trend: Short Sleeve Tops']
df_trends.columns = ['date'] + trend_cols
for col in trend_cols:
    df_trends[col] = df_trends[col].astype(int)

df_trends['date'] = pd.to_datetime(df_trends['date'])
df_trends['date_month'] = df_trends.date.apply(lambda x: x.strftime('%Y-%m'))
df_trends = df_trends.groupby('date_month')[trend_cols].mean().reset_index()


df_trends.head()


df_trends.tail()


df_topic_freq_monthly['date_month'] = df_topic_freq_monthly.date.apply(lambda x: x.strftime('%Y-%m'))
df_merge = pd.merge(df_topic_freq_monthly[df_topic_freq_monthly.topic.isin(['Long Sleeve Tops', 'Short Sleeve Tops'])], df_trends, on='date_month', how='inner')


df_merge.head()


topics = ['Long Sleeve Tops', 'Short Sleeve Tops']
for topic in topics:
    col = 'Google Trend: {}'.format(topic)
    norm_fac = df_merge[df_merge.topic == topic].num_searches.max()
    df_merge[col] = df_merge[col] / df_merge[col].max() * norm_fac


fig, ax = plt.subplots(2)

topic = 'Long Sleeve Tops'
label = 'Onsite Search: {}'.format(topic)
google_trend_col = 'Google Trend: {}'.format(topic)
(df_merge[df_merge.topic == topic].set_index('date')
    [['num_searches', google_trend_col]]
    .fillna(method='pad').rename(columns={'num_searches': label})
    .plot(style=['k-', 'k--'], lw=3, ax=ax[0], alpha=0.7))

topic = 'Short Sleeve Tops'
label = 'Onsite Search: {}'.format(topic)
google_trend_col = 'Google Trend: {}'.format(topic)
(df_merge[df_merge.topic == topic].set_index('date')
    [['num_searches', google_trend_col]]
    .fillna(method='pad').rename(columns={'num_searches': label})
    .plot(style=['m-', 'm--'], lw=3, ax=ax[1]))

ax[0].set_yticklabels([])
ax[1].set_yticklabels([])

plt.xlabel('Date')
savefig('onsite_search_topic=Sleeve-Tops_google_trends')



from IPython.display import HTML
HTML('<style>div.text_cell_render{font-size:130%;padding-top:50px;padding-bottom:50px}</style>')

