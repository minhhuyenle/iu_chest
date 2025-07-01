#!/usr/bin/env python3
"""
Chest X-ray Image Captioning using CheXNet Feature Extraction and LSTM Caption Generation
Based on the Indiana University Chest X-ray Dataset

This script implements an image captioning system for chest X-rays using:
- CheXNet (DenseNet-121) for feature extraction 
- LSTM-based decoder for caption generation
- GloVe embeddings for word representations
"""

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import numpy as np
from numpy import array
import pandas as pd
import cv2
from glob import glob
import PIL
import time
from tqdm import tqdm
import os
import re
import pickle
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, GlobalAveragePooling2D
from keras.layers import Flatten, Concatenate, Dropout, BatchNormalization
from keras.regularizers import l2
from keras import backend as K
from keras.applications.densenet import DenseNet121
from keras.applications.densenet import preprocess_input
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.layers import Dense, Input, Conv2D, MaxPool2D, LSTM, add
from tensorflow.keras.layers import Activation, Dropout, Flatten, Embedding


def load_dataset(image_path):
    """
    Load the dataset images
    
    Args:
        image_path (str): Path to the directory containing chest X-ray images
        
    Returns:
        list: List of image file paths
    """
    print("Loading dataset...")
    images = glob(image_path + "*.png")
    print(f"Found {len(images)} images")
    return images


def create_image_caption_dictionary(projections_csv, reports_csv):
    """
    Create a dictionary mapping image filenames to their corresponding captions
    
    Args:
        projections_csv (str): Path to projections CSV file
        reports_csv (str): Path to reports CSV file
        
    Returns:
        dict: Dictionary with filename as key and list of captions as value
    """
    print("Creating image-caption dictionary...")
    
    # Read in the projections data
    projections = pd.read_csv(projections_csv)
    
    # Read in the reports data
    reports = pd.read_csv(reports_csv)
    
    # Merge the projections and reports data on the UID column
    reports = pd.merge(projections, reports, on='uid')
    
    # Create a dictionary of image filenames and their corresponding captions
    data = {}
    for i in range(len(reports)):
        filename = reports.loc[i, 'filename']
        caption = reports.loc[i, 'impression']
        if filename not in data:
            data[filename] = []
        if isinstance(caption, str) and re.match(r'^\d+\.', caption):
            data[filename].append(caption.split('. ')[1])
        else:
            if data[filename]:
                data[filename][-1] += " " + caption
            else:
                data[filename].append(caption)
    
    print(f"Created dictionary with {len(data)} image-caption pairs")
    return data


def cleanse_data(data):
    """
    Clean and preprocess the caption data
    
    Args:
        data (dict): Raw image-caption dictionary
        
    Returns:
        dict: Cleaned image-caption dictionary
    """
    print("Cleaning caption data...")
    
    dict_2 = dict()
    for key, value in data.items():
        for i in range(len(value)):
            lines = ""
            line1 = value[i]
            if isinstance(line1, str):
                for j in line1.split():
                    if len(j) < 2:
                        continue
                    j = j.lower()
                    lines += j + " "
                if key not in dict_2:
                    dict_2[key] = list()
                dict_2[key].append(lines)
    
    print(f"Cleaned data contains {len(dict_2)} entries")
    return dict_2


def create_vocabulary(data2):
    """
    Create vocabulary from the caption data
    
    Args:
        data2 (dict): Cleaned image-caption dictionary
        
    Returns:
        set: Set of unique words in the vocabulary
    """
    print("Creating vocabulary...")
    
    all_desc = set()
    for key in data2.keys():
        [all_desc.update(d.split()) for d in data2[key]]
    
    print(f"Vocabulary contains {len(all_desc)} unique words")
    return all_desc


def save_dict(data2, filename):
    """
    Save the image-caption dictionary to a text file
    
    Args:
        data2 (dict): Image-caption dictionary
        filename (str): Output filename
    """
    lines = list()
    for key, value in data2.items():
        for desc in value:
            lines.append(key + ' ' + desc)
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()
    print(f"Saved captions to {filename}")


def chexnet(input_shape=(224, 224, 3), weights_path=None):
    """
    Create CheXNet model for feature extraction
    
    Args:
        input_shape (tuple): Input image shape
        weights_path (str): Path to pre-trained weights
        
    Returns:
        Model: CheXNet model
    """
    input_layer = Input(shape=input_shape, name='input_1')
    densenet = DenseNet121(weights=None, include_top=False, input_tensor=input_layer)

    if weights_path is not None:
        densenet.load_weights(weights_path, by_name=True)

    x = densenet.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    predictions = Dense(14, activation='sigmoid', kernel_regularizer=l2(0.0001))(x)
    model = Model(inputs=densenet.input, outputs=predictions)

    return model


def encode_image(image, model):
    """
    Encode a given image into a feature vector
    
    Args:
        image (numpy.ndarray): Input image
        model (Model): CheXNet model
        
    Returns:
        numpy.ndarray: Feature vector
    """
    image = preprocess_input(image)  # preprocess the image
    fea_vec = model.predict(image)  # Get the encoding vector for the image
    fea_vec = np.reshape(fea_vec, fea_vec.shape[1])  # reshape
    return fea_vec


def extract_features(img_dir, weights_path, input_shape=(224, 224, 3)):
    """
    Extract features from all images using CheXNet
    
    Args:
        img_dir (str): Directory containing images
        weights_path (str): Path to CheXNet weights
        input_shape (tuple): Input image shape
        
    Returns:
        dict: Dictionary mapping filenames to feature vectors
    """
    print("Extracting image features using CheXNet...")
    
    # Load the pre-trained CheXNet model
    base_model = chexnet(input_shape=input_shape, weights_path=weights_path)
    
    # Get a list of all the image filenames in the directory
    img_list = os.listdir(img_dir)
    
    encoding = {}
    
    for img_filename in tqdm(img_list, desc="Extracting features"):
        # Load the image from the file
        img_path = os.path.join(img_dir, img_filename)
        img = load_img(img_path, target_size=input_shape[:2])
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        
        # Encode the image and store the encoding vector
        encoding[img_filename] = encode_image(x, base_model)
    
    # Save the encoding vectors as a pickle file
    with open("encodings.pkl", "wb") as f:
        pickle.dump(encoding, f)
    
    print(f"Extracted features for {len(encoding)} images")
    return encoding


def prepare_training_data(data2, word_count_threshold=10):
    """
    Prepare training data including vocabulary and word mappings
    
    Args:
        data2 (dict): Cleaned image-caption dictionary
        word_count_threshold (int): Minimum word frequency threshold
        
    Returns:
        tuple: (all_train_captions, wordtoix, ixtoword, vocab_size, max_length)
    """
    print("Preparing training data...")
    
    # Create a list of all the training captions
    all_train_captions = []
    for key, val in data2.items():
        for cap in val:
            all_train_captions.append(cap)
    
    print(f"Total training captions: {len(all_train_captions)}")
    
    # Consider only words which occur at least word_count_threshold times in the corpus
    word_counts = {}
    nsents = 0
    
    for sent in all_train_captions:
        nsents += 1
        for w in sent.split(' '):
            word_counts[w] = word_counts.get(w, 0) + 1
    
    vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
    print('Preprocessed words %d -> %d' % (len(word_counts), len(vocab)))
    
    # Converting the words to indices and vice versa
    ixtoword = {}
    wordtoix = {}
    
    ix = 1
    for w in vocab:
        wordtoix[w] = ix
        ixtoword[ix] = w
        ix += 1
    
    vocab_size = len(ixtoword) + 1  # one for appended 0's
    
    # Save the words to index and index to word as pickle files
    with open("words.pkl", "wb") as encoded_pickle:
        pickle.dump(wordtoix, encoded_pickle)
    
    with open("words1.pkl", "wb") as encoded_pickle:
        pickle.dump(ixtoword, encoded_pickle)
    
    # Calculate the maximum sequence length
    def to_lines(descriptions):
        all_desc = list()
        for key in descriptions.keys():
            [all_desc.append(d) for d in descriptions[key]]
        return all_desc
    
    def max_length(descriptions):
        lines = to_lines(descriptions)
        return max(len(d.split()) for d in lines)
    
    max_length_val = max_length(data2)
    print('Maximum description length: %d' % max_length_val)
    
    return all_train_captions, wordtoix, ixtoword, vocab_size, max_length_val


def data_generator(descriptions, photos, wordtoix, max_length, num_photos_per_batch, vocab_size):
    """
    Data generator for training the caption model
    
    Args:
        descriptions (dict): Image-caption dictionary
        photos (dict): Image features dictionary
        wordtoix (dict): Word to index mapping
        max_length (int): Maximum sequence length
        num_photos_per_batch (int): Number of photos per batch
        vocab_size (int): Vocabulary size
        
    Yields:
        tuple: ([image_features, sequences], targets)
    """
    X1, X2, y = list(), list(), list()
    n = 0
    # loop for ever over images
    while 1:
        for key, desc_list in descriptions.items():
            n += 1
            # retrieve the photo feature
            photo = photos[key]
            for desc in desc_list:
                # encode the sequence
                seq = [wordtoix[word] for word in desc.split(' ') if word in wordtoix]
                # split one sequence into multiple X, y pairs
                for i in range(1, len(seq)):
                    # split into input and output pair
                    in_seq, out_seq = seq[:i], seq[i]
                    # pad input sequence
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    # encode output sequence
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                    # store
                    X1.append(photo)
                    X2.append(in_seq)
                    y.append(out_seq)
            # yield the batch data
            if n == num_photos_per_batch:
                yield ([array(X1), array(X2)], array(y))
                X1, X2, y = list(), list(), list()
                n = 0


def create_embedding_matrix(wordtoix, vocab_size, glove_path, embedding_dim=200):
    """
    Create embedding matrix using GloVe vectors
    
    Args:
        wordtoix (dict): Word to index mapping
        vocab_size (int): Vocabulary size
        glove_path (str): Path to GloVe vectors file
        embedding_dim (int): Embedding dimension
        
    Returns:
        numpy.ndarray: Embedding matrix
    """
    print("Creating embedding matrix...")
    
    # Load the Glove vectors
    embeddings_index = {}  # empty dictionary
    f = open(glove_path, encoding="utf-8")
    
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Found %s word vectors.' % len(embeddings_index))
    
    # Get embedding_dim-dimensional dense vector for each word in vocabulary
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    
    for word, i in wordtoix.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # Words not found in the embedding index will be all zeros
            embedding_matrix[i] = embedding_vector
    
    print(f"Embedding matrix shape: {embedding_matrix.shape}")
    return embedding_matrix


def create_caption_model(vocab_size, embedding_dim, max_length, embedding_matrix):
    """
    Create the caption generation model architecture
    
    Args:
        vocab_size (int): Vocabulary size
        embedding_dim (int): Embedding dimension
        max_length (int): Maximum sequence length
        embedding_matrix (numpy.ndarray): Pre-trained embedding matrix
        
    Returns:
        Model: Caption generation model
    """
    print("Creating caption generation model...")
    
    # Image feature input
    inputs1 = Input(shape=(None, 10, 14))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)
    
    # Sequence input
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)
    
    # Decoder
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    
    # Set embedding weights
    model.layers[2].set_weights([embedding_matrix])
    model.layers[2].trainable = False
    
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    
    print("Model created successfully")
    model.summary()
    return model


def train_model(model, data2, features, wordtoix, max_length, vocab_size, epochs=4, batch_size=3):
    """
    Train the caption generation model
    
    Args:
        model (Model): Caption generation model
        data2 (dict): Image-caption dictionary
        features (dict): Image features dictionary
        wordtoix (dict): Word to index mapping
        max_length (int): Maximum sequence length
        vocab_size (int): Vocabulary size
        epochs (int): Number of training epochs
        batch_size (int): Batch size
    """
    print("Training the model...")
    
    steps = len(data2) // batch_size
    
    # Enable eager execution for TensorFlow 2.x compatibility
    tf.config.run_functions_eagerly(True)
    
    for i in range(epochs):
        print(f"Epoch {i+1}/{epochs}")
        generator = data_generator(data2, features, wordtoix, max_length, batch_size, vocab_size)
        model.fit(generator, epochs=1, steps_per_epoch=steps, verbose=1)
        model.save(f'model_{i}.h5')
        print(f"Model saved as model_{i}.h5")


def load_trained_model(model_path):
    """
    Load a trained model for inference
    
    Args:
        model_path (str): Path to the saved model
        
    Returns:
        Model: Loaded model
    """
    print(f"Loading model from {model_path}")
    return load_model(model_path)


def generate_caption(picture, model, words_to_index, index_to_words, max_length):
    """
    Generate caption for a given image
    
    Args:
        picture (numpy.ndarray): Image feature vector
        model (Model): Trained caption generation model
        words_to_index (dict): Word to index mapping
        index_to_words (dict): Index to word mapping
        max_length (int): Maximum sequence length
        
    Returns:
        str: Generated caption
    """
    in_text = 'startseq'
    for i in range(max_length):
        sequence = [words_to_index[w] for w in in_text.split() if w in words_to_index]
        sequence = pad_sequences([sequence], maxlen=max_length)
        # Ensure that both arrays have the same number of samples
        yhat = model.predict([np.repeat(picture, len(sequence), axis=0), sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = index_to_words[yhat]
        in_text += ' ' + word
        if word == 'endseq':
            break
    return in_text


def main():
    """
    Main function to run the complete pipeline
    """
    # Configuration
    IMAGE_PATH = "/path/to/chest-xrays-indiana-university/images/images_normalized/"
    PROJECTIONS_CSV = "/path/to/chest-xrays-indiana-university/indiana_projections.csv"
    REPORTS_CSV = "/path/to/chest-xrays-indiana-university/indiana_reports.csv"
    CHEXNET_WEIGHTS = "/path/to/chexnet/weights.h5"
    GLOVE_PATH = "/path/to/glove.6B.200d.txt"
    
    # Step 1: Load dataset
    images = load_dataset(IMAGE_PATH)
    
    # Step 2: Create image-caption dictionary
    data = create_image_caption_dictionary(PROJECTIONS_CSV, REPORTS_CSV)
    
    # Step 3: Clean captions
    data2 = cleanse_data(data)
    vocabulary_data = create_vocabulary(data2)
    save_dict(data2, 'captions1.txt')
    
    # Step 4: Extract image features
    features = extract_features(IMAGE_PATH, CHEXNET_WEIGHTS)
    
    # Step 5: Prepare training data
    all_train_captions, wordtoix, ixtoword, vocab_size, max_length = prepare_training_data(data2)
    
    # Step 6: Create embedding matrix
    embedding_matrix = create_embedding_matrix(wordtoix, vocab_size, GLOVE_PATH)
    
    # Step 7: Create and train model
    model = create_caption_model(vocab_size, 200, max_length, embedding_matrix)
    train_model(model, data2, features, wordtoix, max_length, vocab_size)
    
    # Step 8: Load trained model for inference
    trained_model = load_trained_model('model_3.h5')
    
    # Load required data for inference
    words_to_index = pickle.load(open("words.pkl", "rb"))
    index_to_words = pickle.load(open("words1.pkl", "rb"))
    
    print("Training complete! Model ready for caption generation.")
    
    # Example usage for generating captions
    # features = pickle.load(open("encodings.pkl", "rb"))
    # sample_image_feature = features['sample_image.png']  # Replace with actual filename
    # caption = generate_caption(sample_image_feature, trained_model, words_to_index, index_to_words, max_length)
    # print(f"Generated caption: {caption}")


if __name__ == "__main__":
    main()