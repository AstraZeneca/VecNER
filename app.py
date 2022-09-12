import argparse
import json
from gensim.models import KeyedVectors
from flask import Flask
from flask_restful import reqparse, Api
from vecner import ExactMatcher, ThresholdMatcher, ExtendedMatcher
from gensim.models import KeyedVectors

parser = argparse.ArgumentParser()

parser.add_argument(
    "--model_path",
    type = str,
    help = "type the path from which to load the w2vec-gensim model (Keyed Vectors model)",
)

user_args = vars(parser.parse_args())

MODEL = KeyedVectors.load(
    user_args['model_path']
)

app = Flask(__name__)
api = Api(app)

# argument parsing
reqparser = reqparse.RequestParser()
reqparser.add_argument(
    'text',
    type = str,
    location = 'form'
)

reqparser.add_argument(
    'lexicon',
    type = list,
    location = 'form'
)

reqparser.add_argument(
    'pipeline_steps',
    type = list,
    location = 'form'
)

@app.route('/', methods = ['GET', 'POST'])
def test():

    # parse the user's query (i.e. insight)
    args = reqparser.parse_args()

    user_text = args['text']

    user_lexicon = json.loads(
        ''.join(args['lexicon'])
    )['data']

    # init the Exact Matcher for finding entities
    # as exactly mentioned in the lexicon
    matcher = ExactMatcher(
    user_lexicon,
    spacy_model='en_core_web_sm'
    )

    # init the ThresholdMatcher which finds entities
    # based on a cosine similarity threshold
    thresholdmatcher = ThresholdMatcher(
        user_lexicon,
        w2vec_model=MODEL,
        in_pipeline=True,
        spacy_model='en_core_web_sm',
        chunking_method='noun_chunking',
        threshold = 0.55
    )

    # map exact entities
    output = matcher.map(
        text = user_text
    )

    # use in pipeline to map inexact entities
    output = thresholdmatcher.map(
        document = output['doc'],
        ents = output['ents'],
        ids = output['ids']
    )

    output['doc'] = output['doc'].text
    return output

if __name__ == '__main__':

    app.run(
        debug=True
    )
