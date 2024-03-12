import logging as log
import pickle
import re
import concurrent.futures
import time
import openai
import nltk
import pandas as pd
import requests
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, balanced_accuracy_score
from sklearn.metrics import cohen_kappa_score, matthews_corrcoef
from tqdm import tqdm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from redis_handler import RedisHandler
import numpy as np

from deap import base, creator, tools, algorithms
import random
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import cross_val_score

import random
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

from sklearn.base import clone

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
logging = log.getLogger("ms-ml-page-analyzer")

INITIAL_PAGE = 1
OTHER_PAGES = 0


class ModelsTuning:
    """
    Classe para classificação de documentos.
    """

    def __init__(self, model,fractioned_file_url: str, test_size=0.3, random_state=42, redis_handler: RedisHandler = None):
        self.fractioned_file_url = fractioned_file_url
        self.test_size = test_size
        self.random_state = random_state
        self.redis_handler = redis_handler
        self.model = model

        self.results = []  # For storing results
        
    def genetic_algorithm_optimization(self, X, y, ngen=10, pop_size=50, cxpb=0.5, mutpb=0.2):
        """
        Apply genetic algorithm for hyperparameter optimization.
        """
        # Define the hyperparameter space
        n_estimators_range = range(10, 200)
        max_depth_range = range(1, 20)

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        toolbox = base.Toolbox()
        toolbox.register("attr_n_estimators", random.choice, n_estimators_range)
        toolbox.register("attr_max_depth", random.choice, max_depth_range)
        toolbox.register("individual", tools.initCycle, creator.Individual, 
                         (toolbox.attr_n_estimators, toolbox.attr_max_depth), n=1)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        def evalModel(individual):
            model = RandomForestClassifier(n_estimators=individual[0], max_depth=individual[1])
            return (cross_val_score(model, X, y, cv=5, scoring='accuracy').mean(),)

        toolbox.register("evaluate", evalModel)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutUniformInt, low=[10, 1], up=[200, 20], indpb=0.1)
        toolbox.register("select", tools.selTournament, tournsize=3)

        pop = toolbox.population(n=pop_size)
        final_pop = algorithms.eaSimple(pop, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=ngen, verbose=True)

        top_individual = tools.selBest(pop, 1)[0]
        logging.info(f'Best Individual: {top_individual}, Best Score: {top_individual.fitness.values[0]}')
        
        # Store the best hyperparameters
        self.best_params = {'n_estimators': top_individual[0], 'max_depth': top_individual[1]}
        return self.best_params


    import openai




    def evaluate_and_tune_model(self, X_train, y_train, model_name, model, param_grid):
        """
        Perform grid search to find the best parameters for the model and evaluate its performance.
        """
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words=stopwords.words('portuguese'))),
            ('clf', model)
        ])

        grid_search = GridSearchCV(pipeline, param_grid, scoring='accuracy', cv=5, verbose=3)
        grid_search.fit(X_train, y_train)

        best_params = grid_search.best_params_
        best_score = grid_search.best_score_

        self.results.append({
            'Model': model_name,
            'Best Score': best_score,
            'Best Parameters': best_params
        })

    def run_model_comparisons(self):
        """
        Run comparisons across different models using a random search approach.
        """
        # Split the data
        X_train, X_test, y_train, y_test = self._split_data(self.df, 'pageContent', 'pageType')

        # Define models and random parameter spaces
        models_and_parameters = {
            'Logistic Regression': {
                'model': OneVsRestClassifier(LogisticRegression(solver='liblinear')),
                'params': [
                    {
                        'clf__estimator__C': 10 ** random.uniform(-3, 3),
                        'clf__estimator__penalty': random.choice(['l1', 'l2'])
                    } for _ in range(10)
                ]
            },
            'Random Forest': {
                'model': RandomForestClassifier(),
                'params': [
                    {
                        'clf__n_estimators': random.randint(10, 200),
                        'clf__max_depth': None if random.random() > 0.5 else random.randint(1, 20)
                    } for _ in range(10)
                ]
            },
            'SVM': {
                'model': OneVsRestClassifier(SVC()),
                'params': [
                    {
                        'clf__estimator__C': 10 ** random.uniform(-3, 3),
                        'clf__estimator__kernel': random.choice(['linear', 'rbf', 'poly', 'sigmoid'])
                    } for _ in range(10)
                ]
            },
            'MultinomialNB': {
                'model': MultinomialNB(),
                'params': [
                    {
                        'clf__alpha': random.uniform(0.0, 1.0)
                    } for _ in range(10)
                ]
            }
        }

        # List to store results
        self.results = []  # Reset results for this run

        # Evaluate each model
        for name, mp in models_and_parameters.items():
            for params in mp['params']:
                model_clone = clone(mp['model'])  # Clone the model to ensure a fresh model instance
                pipeline = Pipeline([
                    ('tfidf', TfidfVectorizer(stop_words=stopwords.words('portuguese'))),
                    ('clf', model_clone)
                ])
                pipeline.set_params(**params)
                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                # Save the performance and parameters
                self.results.append({
                    'Model': name,
                    'Accuracy': accuracy,
                    'Parameters': params
                })

        # Convert results to DataFrame and save to CSV
        results_df = pd.DataFrame(self.results)
        results_df.to_csv('model_comparison_results.csv', index=False)
        print(results_df)

    @staticmethod
    def _split_data(df, text_column, label_column, test_size=0.18, random_state=42):
        # Your existing split data method
        X_train, X_test, y_train, y_test = train_test_split(df[text_column], df[label_column],
                                                            test_size=test_size, random_state=random_state)
        return X_train, X_test, y_train, y_test

        

    @staticmethod
    def get_model_metrics(y_true, y_pred):
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro')  # 'macro' para classificação multiclasse
        recall = recall_score(y_true, y_pred, average='macro')
        f1 = f1_score(y_true, y_pred, average='macro')
        confusion = confusion_matrix(y_true, y_pred)
        class_report = classification_report(y_true, y_pred, zero_division=0)
        # ToDo: Consertar
        # roc_auc = roc_auc_score(y_true, y_pred, multi_class='ovo')
        # avg_precision = average_precision_score(y_true, y_pred)
        balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
        kappa = cohen_kappa_score(y_true, y_pred)
        matthews_corr = matthews_corrcoef(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred, squared=False)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred)

        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1-score:", f1)
        """        print("Confusion Matrix:")
        print(confusion)
        print("Classification Report:")
        print(class_report)
        # print("ROC AUC Score:", roc_auc)
        # print("Average Precision Score:", avg_precision)
        print("Balanced Accuracy:", balanced_accuracy)
        print("Cohen's Kappa Score:", kappa)
        print("Matthews Correlation Coefficient:", matthews_corr)
        print("Mean Squared Error (MSE):", mse)
        print("Mean Absolute Error (MAE):", mae)
        print("Mean Absolute Percentage Error (MAPE):", mape)
        print("R-squared (R²):", r2)    
        """

        metrics_dict = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "d1_score": f1,
            "confusion_matrix": confusion,
            "classification_report": class_report,
            "balanced_accuracy": balanced_accuracy,
            "cohens_kappa_score": kappa,
            "matthews_correlation_coefficient": matthews_corr,
            "root_mean_squared_error": mse,
            "mean_absolute_error": mae,
            "mean_absolute_percentage_error": mape,
            "r2": r2
        }

        return metrics_dict





    def train_and_evaluate_model(self, model,X_train_vectorized, X_test_vectorized, y_train, y_test):
        """
        Treina e avalia o modelo.
        :param X_train_vectorized: Matriz de treinamento.
        :param X_test_vectorized: Matriz de teste.
        :param y_train: Rótulos de treinamento.
        :param y_test: Rótulos de teste.
        :return: Modelo treinado.
        """
        
        self.model = model
        #self.model = OneVsRestClassifier(LogisticRegression(solver='liblinear',C=1,penalty='l2'))
        # Treina o modelo
        self.model.fit(X_train_vectorized, y_train)

        # Faz previsões no conjunto de teste
        y_pred = self.model.predict(X_test_vectorized)

        # Métricas de avaliação
        metrics = self.get_model_metrics(y_test, y_pred)

        return self.model, metrics

    @staticmethod
    def vectorize_text(self,X_train: pd.Series, X_test: pd.Series) -> (pd.DataFrame, pd.DataFrame, TfidfVectorizer):
        """
        Vetoriza o texto.
        :param X_train: Matriz de treinamento.
        :param X_test: Matriz de teste.
        :return: Matrizes vetorizadas e o vetorizador.
        """
        # Usa o TfidfVectorizer para vetorizar o texto
        # Use a pipeline as a high-level helper
        from transformers import pipeline
        from transformers import AutoTokenizer, AutoModel
        import torch

# Load model directly
# Tokenize and encode X_train and X_test
        tokenizer = TfidfVectorizer(stop_words=stopwords.words('portuguese'))
        X_train_vectorized = tokenizer.fit_transform(X_train)
        vectorizer = tokenizer
        
        X_test_vectorized = tokenizer.transform(X_test)
        return X_train_vectorized, X_test_vectorized, vectorizer

    



    def load_data_from_csv(self, file_path: str) -> pd.DataFrame:
        """
        Carrega os dados de um arquivo CSV.
        :param file_path: Caminho do arquivo CSV.
        :return: DataFrame com os dados carregados.
        """
        logging.debug(f"Loading data from CSV file '{file_path}'...")
        return pd.read_csv(file_path)

    def compare_models_and_hyperparameters(self, file_name,classification_type,X_train, y_train, X_test, y_test):
        import time
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import Pipeline
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import ParameterGrid
        from sklearn.svm import LinearSVC
        from sklearn.linear_model import SGDClassifier, RidgeClassifier, PassiveAggressiveClassifier
        from sklearn.metrics import classification_report
        import numpy as np
        # Define your models and hyperparameters to compare
        
        solver_penalty_pairs_lr = [
    {'solver': 'newton-cg', 'penalties': ['l2', 'none']},
    #{'solver': 'lbfgs', 'penalties': ['l2', 'none']},
    #{'solver': 'liblinear', 'penalties': ['l1', 'l2']},
    {'solver': 'sag', 'penalties': ['l2', 'none']},
    #{'solver': 'saga', 'penalties': ['l1', 'l2', 'elasticnet', 'none']}
]

# Generate parameter combinations respecting solver-penalty compatibilities
        params_lr = []
        for pair in solver_penalty_pairs_lr:
            for penalty in pair['penalties']:
                for C in np.logspace(-4, 1,2):
                    for max_iter in [10,100]:
                        params_lr.append({
                            'model__solver': pair['solver'],
                            'model__penalty': penalty,
                            'model__C': C,
                            'model__max_iter': max_iter
                        })

        model_hyperparameters = {
            
            'MultinomialNB': {
                'model': MultinomialNB(),
                'params': [
                    {
                        'model__alpha': alpha,
                        'model__fit_prior': fit_prior
                    }
                    for alpha in np.linspace(0.0, 1.0, 11)  # Explore alpha from 0 to 1
                    for fit_prior in [True, False]  # Whether to learn class prior probabilities or not.
                ]
            },
        
            
            'LinearSVC': {
                'model': LinearSVC(),
                'params': [
                    {
                        'model__C': C,
                        'model__max_iter': max_iter,
                        'model__loss': loss
                    }
                    for C in np.logspace(-4, 4, 9)
                    for max_iter in [500, 1000, 2000]
                    for loss in ['hinge', 'squared_hinge']
                ]
            },
            
            'SGDClassifier': {
                'model': SGDClassifier(),
                'params': [
                    {
                        'model__loss': loss,
                        'model__alpha': alpha,
                        'model__penalty': penalty,
                        'model__max_iter': max_iter,
                        'model__learning_rate': learning_rate,
                        'model__eta0': eta0  # Make sure this is a positive value
                    }
                    for loss in ['hinge', 'log','perceptron']
                    for alpha in np.logspace(-6, -1, 6)
                    for penalty in ['l2', 'l1', 'elasticnet']
                    for max_iter in [100,200]
                    for learning_rate in ['constant', 'optimal', 'adaptive']
                    for eta0 in [0.01, 0.1, 1]  # Example values, adjust based on your needs
                ]
            },
            
            
            'RidgeClassifier': {
                'model': RidgeClassifier(),
                'params': [
                    {
                        'model__alpha': alpha,
                        'model__solver': solver,
                        'model__max_iter': max_iter
                    }
                    for alpha in np.logspace(-6, 6, 13)
                    for solver in ['auto']
                    for max_iter in [None, 500, 1000, 2000]
                ]
            },
            
            
            'LogisticRegression': {
                'model': LogisticRegression(),
                'params': params_lr
                
            }
        }


        results = []

        # Iterate through each model and set of hyperparameters
        for model_name, model_info in model_hyperparameters.items():
            for params in model_info['params']:
                pipeline = Pipeline([
                    ('model', model_info['model'])
                ])

                # Measure time to fit the model
                start_time = time.time()
                pipeline.set_params(**params)
                pipeline.fit(X_train, y_train)
                fit_time = time.time() - start_time

                # Measure time to predict with the model
                start_time = time.time()
                y_pred = pipeline.predict(X_test)
                predict_time = time.time() - start_time

                # Evaluate
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='macro')

                # Store the results
                results.append({
                    'Type': classification_type,
                    'Model': model_name,
                    'Parameters': params,
                    'Accuracy': accuracy,
                    'Precision': precision,
                    'Fit Time': fit_time,
                    'Predict Time': predict_time
                })

                # Print out the metrics for each model configuration
                print(f"Type {classification_type} - Report for {model_name} with {params}:")           
                print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Fit Time: {fit_time:.4f}s, Predict Time: {predict_time:.4f}s")

            # Convert results to DataFrame and save to CSV
            results_df = pd.DataFrame(results)
            results_df.to_csv(file_name, index=False)

        return results_df
    
def bert_encode(texts, tokenizer, max_len=128):
    """Function to tokenize and encode a batch of texts"""
    # Tokenization and encoding
    encoded_batch = tokenizer.batch_encode_plus(
        texts, 
        add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
        max_length=max_len,       # Pad & truncate all sentences
        padding=True,
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'       # Return PyTorch tensors
    )
    
    return encoded_batch


def extract_embeddings(model, encoded_batch):
    """Function to extract embeddings from the BERT model"""
    import torch
    # Move encoded batch to the same device as the model
    input_ids = encoded_batch['input_ids'].to(model.device)
    attention_mask = encoded_batch['attention_mask'].to(model.device)

    # Get embeddings
    with torch.no_grad():
        output = model(input_ids=input_ids, attention_mask=attention_mask)
        # Extract the tokens corresponding to the '[CLS]' tokens
        embeddings = output.last_hidden_state[:, 0, :].cpu().numpy()

    return embeddings

def get_embedding(text):
    # Encode text
    import torch
    from transformers import AutoTokenizer, AutoModel
    import pandas as pd
    tokenizer = AutoTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")
    model = AutoModel.from_pretrained("neuralmind/bert-base-portuguese-cased")
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=1)
    # Get model output
    with torch.no_grad():
        outputs = model(**inputs)
    # Extract embeddings from the last hidden state
    last_hidden_state = outputs.last_hidden_state
    # Pool the outputs into a single mean vector
    mean_pooling = torch.mean(last_hidden_state, dim=1)
    embedding_list = mean_pooling.squeeze().tolist()  # Convert tensor to a list
    print(embedding_list)
    return embedding_list

# Apply function to each row in the DataFrame

def generate_embeddings_openai(text, openai_api_key = "key"):
    """
    Generate embeddings for a list of texts using OpenAI's Ada model.
    
    :param texts: A list of strings (text data).
    :param openai_api_key: Your OpenAI API key.
    :return: A list of embeddings.
    """
    openai.api_key = openai_api_key
    # Get the embedding from OpenAI's Ada model
    response = openai.Embedding.create(
        input=[text],  # API expects a list of texts
        model="text-embedding-ada-002"  # Use the appropriate embedding model
    )
    
    # Extract the embedding vector
    embedding_list = response['data'][0]['embedding']
    print(embedding_list)
    
    return embedding_list



import numpy as np
import ast  # Import abstract syntax grammar module to safely evaluate strings containing Python literals

# Convert string representations of lists back to actual lists of floats
def convert_embeddings(embedding):
    if isinstance(embedding, str):  # Check if the embedding is a string
        embedding = ast.literal_eval(embedding)  # Safely evaluate the string as a Python literal
    return np.array(embedding).flatten()  # Flatten the list of lists into a single list
import numpy as np
import ast  # For safely evaluating a string literal as a Python object

# Convert the string representation of list back to a list
def string_to_list(embedding_str):
    return np.array(ast.literal_eval(embedding_str)).flatten()

def string_to_array(embedding_str):
    try:
        # Safely convert string to list
        embedding_list = ast.literal_eval(embedding_str.replace('\n', ','))
        # Convert list to numpy array and flatten
        return np.array(embedding_list).flatten()
    except (ValueError, SyntaxError):
        # Return a default value or handle the error as needed
        return np.zeros(768)  # Assuming 768 is the embedding size


def string_to_float_array(embedding_str):
    # Convert the string representation to a list
    embedding_list = ast.literal_eval(embedding_str)
    # Flatten the list to remove nested structure if necessary
    flattened_list = [val for sublist in embedding_list for val in sublist]
    # Convert the list to a numpy array
    return np.array(flattened_list)

def model_and_classification_test(doc):
        file_name = 'doctype_model_hyperparameter_comparison_results.csv'
        classification_type = "doc type"
        X_train, X_test, y_train, y_test = doc._split_data(doc.df, 'pageContent', 'documentTypeId')
        X_train, X_test, vectorizer = doc.vectorize_text(doc,X_train, X_test)
        
        df = doc.compare_models_and_hyperparameters(file_name,classification_type,X_train, y_train, X_test, y_test)
        
        file_name = 'page_model_hyperparameter_comparison_results.csv'
        classification_type = "page type"
        X_train, X_test, y_train, y_test = doc._split_data(doc.df, 'pageContent', 'pageType')
        X_train, X_test, vectorizer = doc.vectorize_text(doc,X_train, X_test)
        
        df = doc.compare_models_and_hyperparameters(file_name,classification_type,X_train, y_train, X_test, y_test)
# Apply this function to each embedding in your dataframe

def geracao_embeddings(doc,tipo="openai"):
    if tipo == "openai":
        doc.df['embedding'] = doc.df['pageContent'].apply(generate_embeddings_openai)
    if tipo == "bert":
        doc.df['embedding'] = doc.df['pageContent'].apply(get_embedding)

    return doc.df

def salvar_embeddings(doc):
    doc.df.to_csv('embeddings.csv', index=False)
    return

if __name__ == '__main__':
    
    ## Carregamento de arquivo
    doc = ModelsTuning('out.csv', 0.2, 0)
    doc.df = doc.load_data_from_csv('out.csv')
    
    ## Teste de modelos e hiperparametros 
    
    model_and_classification_test(doc)
    exit(8)
    
    ### Para testes de enbeddings
    
    doc.df = geracao_embeddings(doc,"openai")
    salvar_embeddings(doc)
    doc.df = geracao_embeddings(doc,"bert")
    salvar_embeddings(doc)

    