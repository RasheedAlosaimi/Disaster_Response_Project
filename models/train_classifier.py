import sys
import pandas as pd
from sqlalchemy import create_engine
import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download(['punkt', 'wordnet', 'stopwords'])
import re
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report


def load_data(database_filepath):
    db = 'sqlite:///' + database_filepath
    engine = create_engine(db)
    df = pd.read_sql_table('disaster_response', engine)
    X = df['message'].values
    Y = df.iloc[:,4:].values
    category_names = df.iloc[:,4:].columns.values
    print(Y.shape)
    return X, Y, category_names
    
   
def tokenize(text):
    
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for t in tokens:
        clean_t = lemmatizer.lemmatize(t).lower().strip()
        clean_tokens.append(clean_t)
        
    return clean_tokens


def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
                        ])
    
    parameters = {'clf__estimator__max_depth': [8, 18],
                  'clf__estimator__min_samples_split': [2, 5],  
                  'clf__estimator__n_estimators': [14, 100]
                  }

    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1)
    
    return cv
    


def evaluate_model(model, X_test, Y_test, category_names):
    
    Y_pred = model.predict(X_test)
    for i, c in enumerate(category_names):
        print(c, classification_report(Y_test[i],Y_pred[i]))
    


def save_model(model, model_filepath):
    
    pickle.dump(model, open(model_filepath, 'wb'))

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