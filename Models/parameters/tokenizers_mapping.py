from preprocess.tokenizer.lem_tokenizer import Lemmatization
from preprocess.tokenizer.remove_punctuations import RemovePunctuations
from preprocess.tokenizer.remove_stopwords import RemoveStopwords

tokenizers_mapping = {
    "lemmatization": Lemmatization,
    "remove_punctuation": RemovePunctuations,
    "remove_stopwords": RemoveStopwords,
}
