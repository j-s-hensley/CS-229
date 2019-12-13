import numpy as np
from scipy import sparse
import torch
from transformers import *
import progressbar

from train import Linear, Gaussian, load_data, Square

def load_raw(filename, size):
    """ Load the text files output by the clean_data.py script. """
    with open(filename, 'r') as f:
        X = [a.rstrip('#\n').split('#') for a in list(f)]
    return X[:size]


def make_vocab(X_train, X_val, X_test):
    """ Create a set of the unique ingredients. """
    vocab = set()
    for X in [X_train, X_val, X_test]:
        for ing_list in X:
            for ing in ing_list:
                vocab.add(ing)
    vocab = list(vocab)
    print(len(vocab))
    return vocab


def one_hot(X, vocab, filename):
    """ Save the design matrix with a one-hot encoding. """
    X_one_hot = np.append(np.ones((len(X), 1)), np.zeros((len(X), len(vocab))),axis=1)
    for i,recipe in enumerate(X):
        for ing in recipe:
            X_one_hot[i, vocab.index(ing)] = 1

    X_one_hot = sparse.coo_matrix(X_one_hot)
    sparse.save_npz(filename, X_one_hot)


def bert(X, filename):
    """ Save the design matrix with BERT embedding. """
    X_out = np.zeros((len(X),768))

    with torch.no_grad():
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        X_preprocess = ["[CLS] " + ' ; '.join(x)+" [SEP]" for x in X]
        tokens = [tokenizer.tokenize(x) for x in X_preprocess]
        indexed_tokens = [tokenizer.convert_tokens_to_ids(recipe) for recipe in tokens]
        model = BertModel.from_pretrained('bert-base-uncased')
        model.eval()
        for i, recipe_embedding in enumerate(indexed_tokens):
            if i % 100 == 0:
                print(i)
            tokens_tensor = torch.tensor([recipe_embedding])
            segments_tensor = torch.ones_like(tokens_tensor)
            encoded_layers, _ = model(tokens_tensor, segments_tensor)

            X_out[i, :] = np.array(torch.mean(encoded_layers[0], dim=0))
    np.save(filename, X_out)


def get_kernel(X_train, X_test, kernel_func, symmetric = False):
    """ Create the kernel matrix. Use symmetric to speed up computation if X_train and X_test are identical. """
    K = np.zeros((X_train.shape[0], X_test.shape[0]))
    if symmetric:
        for i in progressbar.progressbar(range(X_train.shape[0])):
            for j in range(i+1):
                K[i,j] = K[j,i] = kernel_func(X_train[i,:], X_test[j,:])
    else:
        for i in progressbar.progressbar(range(X_train.shape[0])):
            for j in range(X_test.shape[0]):
                K[i,j] = kernel_func(X_train[i,:], X_test[j,:])
    return K


def kernelize(embedding, kernel):
    """ Create the kernel matrices for the training, validation, and test sets. """
    (X_train, X_val, X_test, Y_train, Y_val, Y_test) = load_data(embedding)
    np.save('K_train_'+embedding+'_'+kernel.__name__, get_kernel(X_train, X_train, kernel, symmetric = True))
    np.save('K_val_'+embedding+'_'+kernel.__name__, get_kernel(X_train, X_val, kernel, symmetric = False))
    np.save('K_test_'+embedding+'_'+kernel.__name__, get_kernel(X_train, X_test, kernel, symmetric = False))


if __name__ == '__main__':
    """ Load the outputs of clean_data and build and save the feature vectors and kernels used in training. """
    X_train = load_raw('X_train.txt', 5000)
    X_val = load_raw('X_train.txt', 1000)
    X_test = load_raw('X_train.txt', 1000)

    vocab = make_vocab(X_train, X_val, X_test)

    one_hot(X_train, vocab, 'X_train_one_hot')
    one_hot(X_val, vocab, 'X_val_one_hot')
    one_hot(X_test, vocab, 'X_test_one_hot')

    bert(X_train, 'X_train_bert')
    bert(X_val, 'X_val_bert')
    bert(X_test, 'X_test_bert')

    kernelize("One-Hot", Square)
    kernelize("One-Hot", Gaussian)
    kernelize("BERT", Square)
    kernelize("BERT", Gaussian)
