import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import string
import nltk
import warnings
from nltk.corpus import stopwords
import tensorflow as tf
import tensorflow_federated as tff
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import emoji
from tensorflow.keras import regularizers
import json
import ftfy

# Suppress TensorFlow logs and download NLTK datasets
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings('ignore')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')

# Directory for saving results
output_directory = 'Client Tweets'
os.makedirs(output_directory, exist_ok=True)
os.makedirs("Results", exist_ok=True)

# Constants for dataset and model
MAX_SEQUENCE_LENGTH = 30
VOCAB_SIZE = 30000
BATCH_SIZE = 10
EPOCHS = 6
ROUND = 5
CLIENTS = 100
PARTITION_STRATEGY = "label_imbalance" # label_imbalance/ quantity_skew


# Function to preprocess text by removing URLs, emojis, punctuation (!, @...), and stopwords (and, the...), and applying lemmatization
def preprocess_text(text):
    text = ftfy.fix_text(text) # For enconding issues like thereâ€™s
    text = emoji.demojize(text, delimiters=("", ""))
    text = re.sub(r'https?://\S+|www\.\S+|@\w+|\d+', '', text)
    text = re.sub(r'\wâ€\S+|\wâ€™\S+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation)) # creates a translation table (x, y, z), which is a mapping of integers or characters to integers, strings, or None.
    # stop_words = set(stopwords.words('english'))
    # words = [word for word in text.lower().split() if word not in stop_words and word != 'rt']
    words = [word for word in text.lower().split() if word != 'rt']
    lemmatizer = nltk.stem.WordNetLemmatizer() # Initialization of Lemmatizer
    return ' '.join([lemmatizer.lemmatize(word) for word in words]) # words like "running", "ran", and "runs" will all be reduced to "run"

# Load and preprocess the dataset
def load_dataset(file_path):
    data = pd.read_csv(file_path)
    print("Original Data:")
    print(data.head())
    print(f"Data shape: {data.shape}")
    print(f"Data type of tweets: {data['tweet'].dtype}")
    data['tweet'] = data['tweet'].apply(preprocess_text)
    print("\nData after preprocessing:")
    print(data.head())
    print(f"Data type of tweets: {data['tweet'].dtype}")

    # Apply labeling
    data['label'] = data['label'].apply(lambda x: 1 if x == 'offensive' or x == 'fake' else 0)

    # Exclude tweets that become empty after preprocessing
    data = data[data['tweet'].str.strip().ne('')]

    print("\nData after labeling:")
    print(data.head())

    # After preprocessing, analyze tweet lengths and types
    data['tweet_length'] = data['tweet'].apply(lambda x: len(x.split()))
    min_tweet_length = data['tweet_length'].min()
    max_tweet_length = data['tweet_length'].max()
    print(f"Minimum tweet length (in words): {min_tweet_length}")
    print(f"Maximum tweet length (in words): {max_tweet_length}")

    # Drop the temporary 'tweet_length' column after analysis
    data.drop(columns=['tweet_length'], inplace=True)
    print(f"Data type of tweets: {data['tweet'].dtype}")
        
    return data

# Function to define and return the model for TensorFlow Federated (TFF)
def create_tff_model():
    def model_fn():
        model = tf.keras.Sequential([
            # Input Layer
            tf.keras.layers.Embedding(VOCAB_SIZE, 64, input_length=MAX_SEQUENCE_LENGTH),
            # Hidden Layers
            tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64, return_sequences=True, kernel_regularizer=regularizers.l2(0.001), recurrent_dropout=0.5)),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.GRU(64, kernel_regularizer=regularizers.l2(0.001), recurrent_dropout=0.5),
            tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
            tf.keras.layers.Dropout(0.5),
            # Output Layer
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        return tff.learning.models.from_keras_model(
            keras_model=model,
            input_spec=(tf.TensorSpec(shape=[None, MAX_SEQUENCE_LENGTH], dtype=tf.int32), 
                        tf.TensorSpec(shape=[None, 1], dtype=tf.float32)),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
    return model_fn

# Standalone model for metrics (the weights of the tff model are passed here for evaluation)
def create_keras_model_for_prediction():
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(VOCAB_SIZE, 64, input_length=MAX_SEQUENCE_LENGTH),
        tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64, return_sequences=True, kernel_regularizer=regularizers.l2(0.001), recurrent_dropout=0.5)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.GRU(64, kernel_regularizer=regularizers.l2(0.001), recurrent_dropout=0.5),
        tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

# Function to shuffle data and create TensorFlow datasets for each client in a federated setting
def make_federated_data(sequences, labels, client_indices):
    """
    Modifies the federated data preparation to work with specific client indices.

    Args:
    sequences: Padded sequences of the dataset.
    labels: Labels corresponding to the sequences.
    client_indices: A dictionary mapping client ids to their data indices.

    Returns:
    A list of tf.data.Dataset objects for each client.
    """
    client_datasets = []
    for client_id, data_indices in client_indices.items():
        client_sequences = sequences[data_indices] # selects the sequences (padded text data) that belong to it
        client_labels = labels[data_indices] # selects the labels corresponding to the client's sequences
        client_dataset = tf.data.Dataset.from_tensor_slices((client_sequences, client_labels)) # A TensorFlow dataset is created which pairs each sequence with its label, preparing it for model training
        client_dataset = client_dataset.shuffle(buffer_size=len(client_labels)).batch(BATCH_SIZE) # Shuffle each clients dataset
        client_datasets.append(client_dataset)
        # print(f"Client {client_id} has {len(data_indices)} data points.")
        
    return client_datasets

def partition_data_label_imbalance(labels, n_clients=CLIENTS, alpha=400):
    print(f"Applying label imbalance with alpha = {alpha}")
    n_classes = len(np.unique(labels)) # Count the number of unique classes [0, 1] in the labels
    data_distribution = np.random.dirichlet([alpha] * n_clients, n_classes) # Generate a Dirichlet distribution to simulate label imbalance across clients
    print(f"Dirichlet distribution per class across clients: {data_distribution}")
    
    client_indices = {i: np.array([], dtype='int') for i in range(n_clients)} # Initialize an empty array for each client to store their data indices
    # Iterate over each class to distribute indices based on the simulated imbalance
    for c in range(n_classes):
        indices_c = np.where(labels == c)[0] # Find the indices of all examples belonging to class c
        np.random.shuffle(indices_c) # Shuffle these indices to randomize data allocation to clients
        
        # Calculate the cumulative proportions of data to allocate to each client
        proportions = data_distribution[c]
        proportions = np.cumsum(proportions)
        # Convert to integer and exclude the last element to avoid overshooting the total count due to rounding
        proportions = (proportions * len(indices_c)).astype(int)[:-1]
        # Split the indices according to the calculated proportions
        client_data_splits = np.split(indices_c, proportions)
        
        # Allocate the split data indices to each client
        for i in range(n_clients):
            client_indices[i] = np.concatenate((client_indices[i], client_data_splits[i]))

    client_data_counts = {}
    # Print the label distribution for each client
    for client_id, data_indices in client_indices.items():
        client_labels = labels[data_indices]
        unique, counts = np.unique(client_labels, return_counts=True)
        label_distribution = dict(zip(unique, counts))
        client_data_counts[f"Client {client_id + 1} label distribution"] = label_distribution
        print(f"Client {client_id + 1} label distribution: {label_distribution}")

    return client_indices, client_data_counts

def partition_data_quantity_skew(data_len, n_clients=CLIENTS, alpha=400, min_data_points_per_client=BATCH_SIZE):
    print(f"Applying quantity skew with alpha = {alpha}")

    # Generates proportions for each client using a Dirichlet distribution to create skew
    proportions = np.random.dirichlet([alpha] * n_clients)
    # Adjusts the proportions to account for the actual dataset size minus a minimum data size reserve for each client
    proportions = (proportions * (data_len - n_clients * min_data_points_per_client)).astype(int)
    print(f"Initial proportions across clients before rounding and minimum adjustment: {proportions}")

    # Ensure each client gets at least the minimum required data points
    proportions += min_data_points_per_client
    # Calculates any remaining data points after the initial distribution
    remaining_data_points = data_len - np.sum(proportions)
    
    # If there are any remaining data points due to rounding, distribute them randomly among the clients
    if remaining_data_points > 0:
        extra_indices = np.random.choice(n_clients, remaining_data_points, replace=True)
        for i in extra_indices:
            proportions[i] += 1

    print(f"Proportions across clients after adjusting for minimum data points: {proportions}")

    # Randomly permutes the indices of the dataset to shuffle the data
    indices = np.random.permutation(data_len)
    client_indices = {}
    start = 0
    # Allocates data to each client based on the calculated proportions
    for i, p in enumerate(proportions):
        end = start + p
        client_indices[i] = indices[start:end]
        start = end

    client_data_counts = {}
    # Print the final data distribution across clients
    for client_id in client_indices:
        client_data_counts[f"Client {client_id + 1}"] = len(client_indices[client_id])
        print(f"Client {client_id + 1} has {len(client_indices[client_id])} data points.")

    return client_indices, client_data_counts

# Tokenize and pad sequences to ensure uniform length
def prepare_data(raw_data):
    print("\nPreparing data...")
    tokenizer = Tokenizer(num_words=VOCAB_SIZE)
    
    # Check for the correct tweet column
    tweet_column = None
    for col in ['tweet', 'tweet_x']:
        if col in raw_data.columns:
            tweet_column = col
            break
    if not tweet_column:
        raise ValueError("Tweet column not found in the dataframe.")
    
    # Fit the tokenizer and prepare sequences
    tokenizer.fit_on_texts(raw_data[tweet_column])
    sequences = tokenizer.texts_to_sequences(raw_data[tweet_column])
    print(f"First 5 sequences: {sequences[:5]}")
    padded_sequences = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
    print(f"First 5 padded sequences: {padded_sequences[:5]}")

    # Proceed similarly with label column as before
    label_column = 'label' if 'label' in raw_data.columns else 'label_x' if 'label_x' in raw_data.columns else None
    if not label_column:
        raise ValueError("Label column not found in the dataframe.")
    
    labels = raw_data[label_column].values.astype('float32').reshape(-1, 1)
    print(f"Labels shape: {labels.shape}")
    print(f"Data type of labels: {labels.dtype}")
    
    return padded_sequences, labels


# Function to integrate new partition strategies into federated learning setup
def integrate_partition_strategies(raw_data, partition_strategy=PARTITION_STRATEGY, alpha=400):
    labels = raw_data['label'].values
    if partition_strategy == "label_imbalance":
        client_indices, client_data_counts = partition_data_label_imbalance(labels, n_clients=CLIENTS, alpha=alpha)
    elif partition_strategy == "quantity_skew":
        client_indices, client_data_counts = partition_data_quantity_skew(len(labels), n_clients=CLIENTS, alpha=alpha)
    else:
        raise ValueError("Unknown partition strategy")
    return client_indices, client_data_counts

# Randomly select a subset of clients for each round of federated training
def select_clients_randomly(client_datasets, min_clients=1):
    selected_clients_indices = np.random.choice(range(len(client_datasets)), size=np.random.randint(min_clients, len(client_datasets) + 1), replace=False)
    selected_client_data = [client_datasets[i] for i in selected_clients_indices]
    return selected_client_data

def select_clients_randomlyFakeNews(x, client_datasets, min_clients=1):
    selected_clients_indices = np.random.choice(range(len(client_datasets)), size=np.random.randint(min_clients, len(client_datasets) + 1), replace=False)
    selected_client_data = [client_datasets[i] for i in selected_clients_indices]
    return selected_client_data, selected_clients_indices


# def convert_to_native_types(obj):
#     if isinstance(obj, np.generic):
#         return obj.item()  # Converts numpy datatype to Python native datatype
#     elif isinstance(obj, dict):
#         # Convert both keys and values, ensuring keys are converted to str if they are integers
#         return {str(convert_to_native_types(key)): convert_to_native_types(value) for key, value in obj.items()}
#     elif isinstance(obj, list):
#         return [convert_to_native_types(value) for value in obj]  # Convert list elements
#     else:
#         return obj



# def save_metrics_to_json(config, filename, metrics_history_native, execution_time):
#     data = {
#         'config': config,
#         'metrics': metrics_history_native,
#         'execution_time': execution_time
#     }
#     with open(filename, 'w') as f:
#         json.dump(data, f, indent=4)

# Create plots to display the overall perfomance of the model
def plot_training_metrics(metrics_history, file_name_suffix):
    # Ensure the 'rounds' list is correctly sized to the length of the metrics arrays.

    print(len(metrics_history['auc']))
    print(len(metrics_history['f1_score']))
    print(len(metrics_history['train_accuracy']))
    # Add similar prints for other metrics.

    max_length = max(len(metrics_history[key]) for key in metrics_history if isinstance(metrics_history[key], list))
    rounds = list(range(1, max_length + 1))

    # Create subplots
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Model Performance Metrics Over FL ROUNDS')

    # Ensure that each metric plot checks for length before plotting
    if len(metrics_history['auc']) == max_length:
        axs[0, 0].plot(rounds, metrics_history['auc'], marker='o')
    axs[0, 0].set_title('AUC')
    axs[0, 0].set_xlabel('FL ROUNDS')
    axs[0, 0].set_ylabel('AUC')

    if len(metrics_history['f1_score']) == max_length:
        axs[0, 1].plot(rounds, metrics_history['f1_score'], marker='o')
    axs[0, 1].set_title('F1 Score')
    axs[0, 1].set_xlabel('FL ROUNDS')
    axs[0, 1].set_ylabel('F1 Score')

    if len(metrics_history['train_accuracy']) == max_length:
        axs[0, 2].plot(rounds, metrics_history['train_accuracy'], marker='o', label='Train Accuracy')
    if len(metrics_history['test_accuracy']) == max_length:
        axs[0, 2].plot(rounds, metrics_history['test_accuracy'], marker='o', label='Test Accuracy')
    axs[0, 2].legend()
    axs[0, 2].set_title('Test-Train Accuracy')
    axs[0, 2].set_xlabel('FL ROUNDS')
    axs[0, 2].set_ylabel('Accuracy')

    if len(metrics_history['train_loss']) == max_length:
        axs[1, 0].plot(rounds, metrics_history['train_loss'], marker='o', label='Train Loss')
    if len(metrics_history['test_loss']) == max_length:
        axs[1, 0].plot(rounds, metrics_history['test_loss'], marker='o', label='Test Loss')
    axs[1, 0].legend()
    axs[1, 0].set_title('Test-Train Loss')
    axs[1, 0].set_xlabel('FL ROUNDS')
    axs[1, 0].set_ylabel('Loss')

    if len(metrics_history['precision']) == max_length:
        axs[1, 1].plot(rounds, metrics_history['precision'], marker='o')
    axs[1, 1].set_title('Precision')
    axs[1, 1].set_xlabel('FL ROUNDS')
    axs[1, 1].set_ylabel('Precision')

    if len(metrics_history['recall']) == max_length:
        axs[1, 2].plot(rounds, metrics_history['recall'], marker='o')
    axs[1, 2].set_title('Recall')
    axs[1, 2].set_xlabel('FL ROUNDS')
    axs[1, 2].set_ylabel('Recall')

    # Adjust layout and save to file
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'Results/plots_{file_name_suffix}.png')
    plt.close()


def create_latex_summary(metrics_history, dataset_name, num_clients, alpha):
    # Calculate mean and standard deviation for each metric
    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'F1 Score'],
        'Mean': [np.mean(metrics_history['test_accuracy']), np.mean(metrics_history['f1_score'])],
        'Std': [np.std(metrics_history['test_accuracy']), np.std(metrics_history['f1_score'])]
    })

    def format_latex(row):
        return f"${row['Mean']:.2f} \pm {row['Std']:.3f}$"

    # Apply the formatting
    metrics_df['Result'] = metrics_df.apply(format_latex, axis=1)

    # Add static columns for dataset name, number of clients, and alpha
    metrics_df['Dataset'] = dataset_name
    metrics_df['# Clients'] = num_clients
    metrics_df['Alpha'] = alpha

    # Reorder DataFrame to display new columns appropriately
    metrics_df = metrics_df[['Dataset', '# Clients', 'Alpha', 'Metric', 'Result']]

    # Convert to LaTeX
    latex_table = metrics_df.to_latex(index=False, escape=False)
    print(latex_table)


def evaluate_global_model(keras_model, test_features, test_labels):
    # Generate predictions
    predictions = keras_model.predict(test_features)
    
    # Convert predictions to binary (0 or 1) based on threshold of 0.5
    predicted_classes = (predictions > 0.5).astype(int)
    
    # Compute metrics
    accuracy = accuracy_score(test_labels, predicted_classes)
    precision = precision_score(test_labels, predicted_classes,average="weighted")
    recall = recall_score(test_labels, predicted_classes,average="weighted")
    f1 = f1_score(test_labels, predicted_classes,average="weighted")
    
    # Compute loss
    test_loss = np.mean(tf.keras.losses.binary_crossentropy(tf.convert_to_tensor(test_labels, dtype=tf.float32), tf.convert_to_tensor(predictions, dtype=tf.float32)))
    
    # Compute AUC if more than one unique class is present
    if len(np.unique(test_labels)) > 1:
        auc = roc_auc_score(test_labels, predictions,average="weighted")
    else:
        print("AUC not defined with single class present")
        auc = 0

    return test_loss, accuracy, precision, recall, f1, auc


# Function to load data for a specific client
def load_client_data(client_id):
    train_path = os.path.join(output_directory, f'tweets_for_client_{client_id}_train.csv')
    train_data = pd.read_csv(train_path)
    return train_data

# Function to create TensorFlow datasets for clients
def create_tf_datasets_for_clients(client_id):
    train_data = load_client_data(client_id)
    train_sequences, train_labels = prepare_data(train_data)
    # Repeat the dataset for the number of EPOCHS
    train_dataset = tf.data.Dataset.from_tensor_slices((train_sequences, train_labels))
    train_dataset = train_dataset.repeat(EPOCHS).shuffle(len(train_labels)).batch(BATCH_SIZE)
    return train_dataset

def create_global_dataset_and_test_set(client_datasets):
    # Combine all client data into one DataFrame
    all_data = pd.DataFrame()
    for data in client_datasets.values():
        all_data = pd.concat([all_data, data], ignore_index=True)
    
    # Remove duplicates
    all_data = all_data.drop_duplicates().reset_index(drop=True)
    
    # Split the data to create a global test set
    train_data, test_data = train_test_split(all_data, test_size=0.2, random_state=42)
    
    return train_data, test_data

def exclude_test_data_from_clients(client_datasets, test_data):
    active_clients = {}
    for client_id, data in client_datasets.items():
        # Exclude test data from client's dataset
        filtered_data = pd.merge(data, test_data, indicator=True, how='outer', on='tweet_id').query('_merge == "left_only"').drop('_merge', axis=1)
        if not filtered_data.empty:
            active_clients[client_id] = filtered_data
    return active_clients

# Main execution
if __name__ == "__main__":

    # dataset_path = 'Datasets/abusive_dataset.csv'
    # dataset_path = 'Datasets/racism_sexism_dataset.csv'
    # dataset_path = 'Datasets/sarcastic_dataset.csv'
    dataset_path = 'Datasets/offensive_dataset_raw.csv'
    # dataset_path = 'Datasets/tweets_and_labels.csv'
    data = pd.read_csv(dataset_path)

    # Check for the presence of specific labels
    offensive_labels = {'offensive', 'neither'}
    fake_news_labels = {'fake', 'real'}

    global x
    x = 0

    client_datasets = {client_id: create_tf_datasets_for_clients(client_id) for client_id in range(CLIENTS)}

    if any(label in fake_news_labels for label in data['label'].unique()):
        
        x = 1

        full_dataset = load_dataset(dataset_path)
        tweets_to_clients = pd.read_csv('Datasets/tweets_to_100clients_.csv')

        client_data = {}
        for client_id in range(CLIENTS):
            # Filter tweets for each client based on assigned tweet IDs
            filtered_ids = tweets_to_clients[tweets_to_clients['client'] == client_id]['tweet_id']
            client_data[client_id] = full_dataset[full_dataset['tweet_id'].isin(filtered_ids)]

        # Create global dataset and test set
        global_train_data, global_test_set = create_global_dataset_and_test_set(client_data)

        # Filter out the global test set from each client's data
        active_clients = exclude_test_data_from_clients(client_data, global_test_set)

        clients_fake = len(active_clients)

        # Save the filtered datasets and the global test set
        for client_id, data in active_clients.items():
            data.to_csv(f'Client Tweets/tweets_for_client_{client_id}_train.csv', index=False)
        global_test_set.to_csv('Client Tweets/global_test_set.csv', index=False)

        # Assuming global_test_data is a tuple (global_test_features, global_test_labels)
        global_test_features, global_test_labels = prepare_data(global_test_set)

        config = {
            'clients': CLIENTS,
            'epochs': EPOCHS,
            'rounds': ROUND
        }

    else:
        raw_data = load_dataset(dataset_path) # raw_data = pandas Dataframe

        # Shuffle the dataset
        raw_data = raw_data.sample(frac=1).reset_index(drop=True)
        print("\nData after shuffling:")
        print(raw_data)

        # Logic for determine the max clients that can be supported (247 clients)
        # total_data_points = len(raw_data)  # Total number of data points in the dataset
        # min_data_points_per_client = 100  # Assuming each client should have at least 100 data points
        # max_clients = calculate_max_clients(total_data_points, min_data_points_per_client)
        # print(f"Maximum number of clients that can be supported: {max_clients}")

        # Split the preprocessed data into training and testing sets
        train_data, test_data = train_test_split(raw_data, test_size=0.2, random_state=42, stratify=raw_data['label'])

        # Save the split data to CSV files
        train_data.to_csv('train.csv', index=False)
        test_data.to_csv('test.csv', index=False)

        train_data = pd.read_csv('train.csv')
        test_data = pd.read_csv('test.csv')

        train_sequences, train_labels = prepare_data(train_data)
        test_sequences, test_labels = prepare_data(test_data)

        # Choose either "label_imbalance" or "quantity_skew" as partition_strategy
        client_indices, client_data_counts = integrate_partition_strategies(train_data, partition_strategy=PARTITION_STRATEGY, alpha=400)

        config = {
            'alpha': 15,
            'partition_strategy': PARTITION_STRATEGY,
            'clients': CLIENTS,
            'epochs': EPOCHS,
            'rounds': ROUND,
            'client_distribution': client_data_counts
        }
        
        # Use client_indices to create federated training data
        federated_train_data = make_federated_data(train_sequences, train_labels, client_indices)

    # Initialize the TFF process for federated averaging
    iterative_process = tff.learning.algorithms.build_weighted_fed_avg(
        model_fn=create_tff_model(),
        client_optimizer_fn=lambda: tf.keras.optimizers.Adam(learning_rate=0.001),
        server_optimizer_fn=lambda: tf.keras.optimizers.Adam(learning_rate=0.001)
    )

    state = iterative_process.initialize()

    # Standalone Keras model identical to federated model for evaluation purposes
    # The standalone model is used here to evaluate the performance of the federated model on a validation set
    # This approach allows for direct evaluation using traditional metrics outside the federated learning framework
    keras_model = create_keras_model_for_prediction()

    metrics_history = {
    'train_loss': [],
    'train_accuracy': [],
    'test_loss': [],
    'test_accuracy': [],
    'auc': [],
    'f1_score': [],
    'precision': [],
    'recall': []
    }

    execution_time = []


    for round_num in range(1, ROUND + 1):
        start_time = time.time()

        # Initialize empty lists to store test metrics for all clients
        round_test_loss = []
        round_test_accuracy = []
        round_auc = []
        round_f1 = []
        round_precision = []
        round_recall = []
        
        if x == 0:
            selected_federated_train_data = select_clients_randomly(federated_train_data, min_clients=1)
            state, metrics = iterative_process.next(state, selected_federated_train_data)
            round_duration = time.time() - start_time
            execution_time.append(round_duration)

            train_metrics = metrics['client_work']['train']

            print(f'Round {round_num}, Training Metrics: {train_metrics}')
            metrics_history['train_loss'].append(train_metrics['loss'])
            metrics_history['train_accuracy'].append(train_metrics['binary_accuracy'])

            # Extract model weights
            model_weights = iterative_process.get_model_weights(state)
            # standalone_keras_model.set_weights(model_weights.trainable)
            keras_model.set_weights(model_weights.trainable)

            # Predictions and evaluations using test_sequences and test_labels
            predictions = keras_model.predict(test_sequences).flatten()
            test_loss, accuracy, precision, recall, f1, auc = evaluate_global_model(keras_model, test_sequences, test_labels)

            round_test_loss.append(test_loss)
            round_test_accuracy.append(accuracy)
            round_auc.append(auc)
            round_f1.append(f1)
            round_precision.append(precision)
            round_recall.append(recall)

            average_test_loss = np.mean(round_test_loss)
            average_test_accuracy = np.mean(round_test_accuracy)
            average_auc = np.mean(round_auc)
            average_f1 = np.mean(round_f1)
            average_precision = np.mean(round_precision)
            average_recall = np.mean(round_recall)

        else:
            selected_client_ids = np.random.choice(list(active_clients.keys()), size=np.random.randint(1, clients_fake + 1), replace=False)
            selected_federated_train_data = [create_tf_datasets_for_clients(client_id) for client_id in selected_client_ids]

            # Proceed with federated learning
            state, metrics = iterative_process.next(state, selected_federated_train_data)
            round_duration = time.time() - start_time
            execution_time.append(round_duration)

            # Training metrics
            train_metrics = metrics['client_work']['train']
            print(f'Round {round_num}, Training Metrics: {train_metrics}')
            metrics_history['train_loss'].append(train_metrics['loss'])
            metrics_history['train_accuracy'].append(train_metrics['binary_accuracy'])

            # Load model weights into standalone Keras model for evaluation
            model_weights = iterative_process.get_model_weights(state)
            keras_model.set_weights(model_weights.trainable)
                
            test_loss, accuracy, precision, recall, f1, auc = evaluate_global_model(keras_model, global_test_features, global_test_labels)

            round_test_loss.append(test_loss)
            round_test_accuracy.append(accuracy)
            round_auc.append(auc)
            round_f1.append(f1)
            round_precision.append(precision)
            round_recall.append(recall)
            
            average_test_loss = np.mean(round_test_loss)
            average_test_accuracy = np.mean(round_test_accuracy)
            average_auc = np.mean(round_auc)
            average_f1 = np.mean(round_f1)
            average_precision = np.mean(round_precision)
            average_recall = np.mean(round_recall)

        # Store average test metrics for the round
        metrics_history['test_loss'].append(average_test_loss)
        metrics_history['test_accuracy'].append(average_test_accuracy)
        metrics_history['auc'].append(average_auc)
        metrics_history['f1_score'].append(average_f1)
        metrics_history['precision'].append(average_precision)
        metrics_history['recall'].append(average_recall)

        # Final evaluation and save metrics
        # metrics_history_native = convert_to_native_types(metrics_history)
        # save_metrics_to_json(config, 'Results/metrics.json', metrics_history_native, execution_time)

        # Generate and save plots
        plot_training_metrics(metrics_history, 'final_metrics')

        accuracy_std = np.std(metrics_history['test_accuracy'])
        f1_score_std = np.std(metrics_history['f1_score'])

        print(f"Round {round_num} Validation Metrics - Accuracy: {average_test_accuracy:.2f}, Precision: {average_precision:.2f}, Recall: {average_recall:.2f}, F1 Score: {average_f1:.2f}, AUC: {average_auc:.2f}")
    create_latex_summary(metrics_history, 'MM-Covid', 50, 0.5)
    print("Federated Training Finished.")