#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from os import listdir, makedirs
from os.path import join, exists, isdir
from time import time
import sys
import numpy as np
from sklearn.cross_validation import KFold
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import matplotlib.pyplot as pl
from scipy.misc import imread
from PIL import Image, ImageOps
from bunch import Bunch

POKEMON_PATH = './pokemon'
POKEMON_PROC_PATH = './pokemon_processed'



def plot_gallery(images, titles, h, w, n_row = 3, n_col = 4):
    pl.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    pl.subplots_adjust(bottom = 0, left = .01, right = .99, top = .90, hspace = .35)
    for i in range(n_row * n_col):
        pl.subplot(n_row, n_col, i + 1)
        pl.imshow(images[i].reshape((h, w)), cmap = pl.cm.gray)
        pl.title(titles[i], size = 12)
        pl.xticks(())
        pl.yticks(())


def title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)


def get_pokemon(h = 200, w = 200):
    # Processing data: make all images h X w in size and gray scale, save them in diff folder
    if not exists(POKEMON_PROC_PATH):
        makedirs(POKEMON_PROC_PATH)
    pokemon_names, pokemon_paths = [], []
    
    for pokemon_name in sorted(listdir(POKEMON_PATH)):
        folder = join(POKEMON_PATH, pokemon_name)
        if not isdir(folder):
            continue

        # Create new directory for each pokemon
        new_folder = join(POKEMON_PROC_PATH, pokemon_name)
        if not exists(new_folder):
            makedirs(new_folder)
            
        paths = [join(folder, f) for f in sorted(listdir(folder))]
        
        # Iterate over existing pokemon's pictures and process each one
        for i, path in enumerate(paths):
            img = Image.open(path).convert('L')
            img = ImageOps.fit(img, (w, h), Image.ANTIALIAS, 0, (0.5, 0.5))
            new_path = join(POKEMON_PROC_PATH, pokemon_name, str(i) + '.jpg')
            img.save(new_path)
        
    # Read pokemon names and paths
    for pokemon_name in sorted(listdir(POKEMON_PROC_PATH)):
        folder = join(POKEMON_PROC_PATH, pokemon_name)
        if not isdir(folder):
            continue
        paths = [join(folder, f) for f in sorted(listdir(folder))]
        n_images = len(paths)
        pokemon_names.extend([pokemon_name] * n_images)
        pokemon_paths.extend(paths)
    
    # Numerialize label
    n_pokemon = len(pokemon_paths)
    target_names = np.unique(pokemon_names)
    target = np.searchsorted(target_names, pokemon_names)
    
    # Read data
    pokemons = np.zeros((n_pokemon, h, w), dtype = np.float32)
    for i, pokemon_path in enumerate(pokemon_paths):
        img = Image.open(pokemon_path)
        pokemon = np.asarray(img, dtype = np.uint32)
        pokemons[i, ...] = pokemon

    # Shuffle Pokemon
    indices = np.arange(n_pokemon)
    np.random.RandomState(42).shuffle(indices)
    pokemons, target = pokemons[indices], target[indices]

    return Bunch(data = pokemons.reshape(len(pokemons), -1), images = pokemons, target = target, target_names = target_names, DESCR = "Pokemon dataset")

# Main function
def main():
    np.random.seed(3)
    pokemon = get_pokemon()
    h, w = pokemon.images[0].shape
    X = pokemon.data
    y = pokemon.target
    n_classes = pokemon.target_names.shape[0]
    precision = 0.0
    recall = 0.0
    kf = KFold(len(y), n_folds = 4, shuffle = True)
    t0 = time()

    
    for train_index, test_index in kf:
        X_train = np.array([X[i] for i in train_index])
        X_test = np.array([X[i] for i in test_index])
        y_train = np.array([y[i] for i in train_index])
        y_test = np.array([y[i] for i in test_index])
        
        # Apply PCA
        y_pred = ''
        svm = ''
        kn = ''

        n_components = 18
        pca = PCA(n_components = n_components, whiten = False).fit(X_train)
        eigenpokemons = pca.components_
        print('Projecting the input data on the eigenpokemon orthonormal basis')
        X_train_pca = pca.transform(X_train)
        X_test_pca = pca.transform(X_test)

        if sys.argv[1] == 'SVM' or sys.argv[1] == 'svm':
            # Train an SVM classification model 
            svm = SVC(class_weight = 'balanced')
            svm.fit(X_train, y_train)
            y_pred = svm.predict(X_test)

            print('Fitting SVM to the training set')
            param_grid = {
                'kernel': ['rbf', 'linear'],
                'C': [1e3, 1e4, 1e5],
                'gamma': [0.0001, 0.001, 0.01]
            }
            clf = GridSearchCV(SVC(class_weight = 'balanced'), param_grid)
            clf = clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
        elif sys.argv[1] == 'KNN' or sys.argv[1] == 'knn':
            # Train an KNN classification model
            kn = KNeighborsClassifier(n_neighbors = 5, p = 2)
            kn.fit(X_train, y_train)
            y_pred = kn.predict(X_test)
        
        # Quantitative evaluation of the model quality on the test set
        print("Predicting pokemon names on the testing set")
        print(classification_report(y_test, y_pred, target_names = pokemon.target_names))
        print(confusion_matrix(y_test, y_pred, labels = range(n_classes)))
        
        precision += precision_score(y_test, y_pred, average = 'weighted')
        recall += recall_score(y_test, y_pred, average = 'weighted')
        
        # View results
        prediction_titles = [title(y_pred, y_test, pokemon.target_names, i) for i in range(y_pred.shape[0])]
        eigenpokemons_titles = ["eigenpokemon %d" % i for i in range(eigenpokemons.shape[0])]

        plot_gallery(X_test, prediction_titles, h, w)
        plot_gallery(eigenpokemons, eigenpokemons_titles, h, w)
        pl.show()

    print ("Computed in %0.3fs" % (time() - t0))
    print('precision = %0.3f' %(precision / len(kf)))
    print('recall = %0.3f' %(recall / len(kf)))


if __name__ == "__main__":
    main()