from sklearn.feature_extraction.text import TfidfVectorizer as Tfidf
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import make_pipeline, make_union, Pipeline
from sklearn.feature_extraction import DictVectorizer
from operator import itemgetter
import pandas as pd

class Vectorizer():
    def __init__(self):
        self.vectorizer = None

    def on_field(self, f: str, *vec):
        return make_pipeline(FunctionTransformer(itemgetter(f), validate=False), *vec)

    def to_records(self, df: pd.DataFrame):
        return df.to_dict(orient='records')
    
    def tfidf_vectorizer(self, title_feat=100000, description_feat=500000)
        self.vectorizer = make_union(
                                    self.on_field("title", Tfidf(max_features=title_feat, token_pattern="\w+")),
                                    self.on_field("description", Tfidf(max_features=description_feat, token_pattern="\w+", ngram_range=(1, 2))),
                                    self.on_field(['shipping', 'status'],
                                    FunctionTransformer(self.to_records, validate=False), DictVectorizer())
                                    )
        return self.vectorizer
