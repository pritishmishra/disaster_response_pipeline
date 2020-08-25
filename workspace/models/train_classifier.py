import sys
# import libraries
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import nltk
import re
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

def load_data(database_filepath):
    """
    This function loads the clean data stored in the database and 
    populates the dataframe from the data. It also populates the 
    Features, Targets and Category Names.
    Input: Filepath of the database file
    Output: Features (X), Targets (Y), Category Names
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('InsertTableName', engine)
    df = df.dropna(subset=df.select_dtypes(float).columns, how='all')
    X = df.message.values
    Y = df[df.columns[4:]].values
    category_names = df.columns[4:]
    return (X, Y, category_names)


special_char_regex = '[^\w\*]'
def tokenize(text):
    """
    This function normalizes, lemmatizes and tokenizes the
    input text. It then returns these clean tokens.
    Input: Text
    Output: Clean tokens
    """
    special_chars = re.findall(special_char_regex, text)
    for special_char in special_chars:
        text = text.replace(special_char, " ")
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens
    


def build_model(useGridSearch=False):
    """
    This function creates the ML model. Primarily, it defines a
    ML pipeline and the steps to be executed as part of this pipeline.
    You can turn the feature of GridSearch on or off. When GridSearch
    is turned on, the model takes a huge amount of time to train.
    """
    pipeline = Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer()),
                ('clf', MultiOutputClassifier(KNeighborsClassifier()))
        ])
    
    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'vect__max_df': (0.5, 0.75, 1.0),
        'vect__max_features': (None, 5000, 10000),
        'tfidf__use_idf': (True, False)
    }
    
    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    if (useGridSearch):
        return cv
    else:
        return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    """
    This function checks the accuracy of the Ml model. It checks and outputs values
    for accuracy and also prints a report of F1 score and other accuracy metrics for
    each classification category. This accuracy is measured on the test data.
    """
    Y_pred = model.predict(X_test)
    accuracy = (Y_pred == Y_test).mean()
    print("Accuracy:", accuracy)
    for i in range(36):
        print("Statistics for category: ", category_names[i], "\n", classification_report(Y_test[:,i],Y_pred[:,i]))


def save_model(model, model_filepath):
    """
    This function saves the model into a pickle file.
    Input: ML model to be saved, Pickle Filepath where ML model should be stored at 
    """
    # Save the model as a pickle in a file 
    joblib.dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
