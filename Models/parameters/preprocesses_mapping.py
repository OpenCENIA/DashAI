from preprocess.bow import BOWPreprocess
from preprocess.distil_emb import DistilBertEmbedding
from preprocess.tfidf import TFIDFPreprocess

preprocesses_mapping = {
    'bow': BOWPreprocess,
    'distil': DistilBertEmbedding,
    'tfidf': TFIDFPreprocess
}
