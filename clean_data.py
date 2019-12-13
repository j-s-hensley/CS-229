"""
Script to clean the data from https://www.kaggle.com/shuyangli94/food-com-recipes-and-user-interactions.
"""

import numpy as np
import csv
import collections

# Load recipe data
with open("food-com-recipes-and-user-interactions/RAW_recipes.csv", 'r') as f:
  reader = csv.reader(f)
  raw_read = list(reader)

# Get the ingredients for each recipe and associate them with the recipe's unique ID
id_to_ing = dict()
id_to_rating = dict()
for i,recipe in enumerate(raw_read[1:]):
    # Remove junk characters from ingredients
    id_to_ing[recipe[1]] = [a.strip("'").strip('"') for a in recipe[10].strip('"[').strip(']"').split(", ")]
    id_to_rating[recipe[1]] = []

del raw_read

# Find ingredients that appear in fewer than five recipes
wordCounter = collections.Counter()
for id in id_to_ing:
    for word in set(id_to_ing[id]):
        wordCounter[word] += 1
for key in wordCounter:
    if wordCounter[key] < 5:
        wordCounter[key] = -1
wordCounter = +wordCounter
goodWords = set(wordCounter.keys())

to_delete = []

# Delete ingredients that appear in fewer than five recipes
for id in id_to_ing:
    id_to_ing[id] = [a for a in id_to_ing[id] if a in goodWords]
    if id_to_ing == []:
        to_delete += [id]
for item in to_delete:
    del id_to_rating[item]

# Load the user interaction data
with open("food-com-recipes-and-user-interactions/RAW_interactions.csv", 'r') as f:
  reader = csv.reader(f)
  raw_ratings = list(reader)

# Get the ratings for ecah recipe
for i, interaction in enumerate(raw_ratings[1:]):
    if interaction[1] in id_to_rating.keys() and interaction[3] != "0":
        id_to_rating[interaction[1]] += [int(interaction[3])]

del raw_ratings

to_delete = []

# Delete recipes that have no rating
for recipe in id_to_rating:
    if id_to_rating[recipe] == []:
        to_delete += [recipe]
    else:
        id_to_rating[recipe] = sum(id_to_rating[recipe])/len(id_to_rating[recipe])
for item in to_delete:
    del id_to_rating[item]


# Get rid of the recipe ID, instead associating the ingredients with the rating
X = [0 for i in range(len(id_to_rating))]
Y = [0 for i in range(len(id_to_rating))]
for i, id in enumerate(id_to_rating.keys()):
    X[i] = id_to_ing[id]
    Y[i] = id_to_rating[id]

# Save as a training set, valdiation set, and test set (70/15/15 split)
Y = np.array(Y)
with open('X_train.txt', 'w') as f:
    for ingredients in X[:158612]:
        for ing in ingredients:
            f.write(ing+'#')
        f.write('\n')
with open('X_val.txt', 'w') as f:
    for ingredients in X[158612:192601]:
        for ing in ingredients:
            f.write(ing+'#')
        f.write('\n')
with open('X_test.txt', 'w') as f:
    for ingredients in X[192601:]:
        for ing in ingredients:
            f.write(ing+'#')
        f.write('\n')
np.save('Y_train',Y[:158612])
np.save('Y_val',Y[158612:192601])
np.save('Y_test',Y[192601:])
