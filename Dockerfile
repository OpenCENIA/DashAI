FROM python:3.8.7
ENV PYTHONUNBUFFERED=1
WORKDIR /code
COPY Pipfile /code/
COPY Pipfile.lock /code/
RUN pip3 install pipenv \
 && pipenv install --deploy --system --ignore-pipfile
RUN python3 -m spacy download es_core_news_sm
RUN python3 -c "import nltk;nltk.download('stopwords')"
COPY . /code/