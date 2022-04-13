import numpy as np
from helper_functions.color_patterns import *
from helper_functions.get_data import *
from manimlib import *
from scipy.stats import entropy
from tqdm import tqdm as ProgressDisplay

# Functions associated with entropy calculation

def get_weights(words, priors):
    frequencies = np.array([priors[word] for word in words])
    total = frequencies.sum()
    if total == 0:
        return np.zeros(frequencies.shape)
    return frequencies / total


def get_pattern_distributions(allowed_words, possible_words, weights):
    """
    For each possible guess in allowed_words, this finds the probability
    distribution across all of the 3^5 wordle patterns you could see, assuming
    the possible answers are in possible_words with associated probabilities
    in weights.

    It considers the pattern hash grid between the two lists of words, and uses
    that to bucket together words from possible_words which would produce
    the same pattern, adding together their corresponding probabilities.
    """
    pattern_matrix = get_pattern_matrix(allowed_words, possible_words)

    n = len(allowed_words)
    distributions = np.zeros((n, 3**5))
    n_range = np.arange(n)
    for j, prob in enumerate(weights):
        distributions[n_range, pattern_matrix[:, j]] += prob
    return distributions


def entropy_of_distributions(distributions, atol=1e-12):
    axis = len(distributions.shape) - 1
    return entropy(distributions, base=2, axis=axis)


def get_entropies(allowed_words, possible_words, weights):
    if weights.sum() == 0:
        return np.zeros(len(allowed_words))
    distributions = get_pattern_distributions(allowed_words, possible_words, weights)
    return entropy_of_distributions(distributions)


def max_bucket_size(guess, possible_words, weights):
    dist = get_pattern_distributions([guess], possible_words, weights)
    return dist.max()


def words_to_max_buckets(possible_words, weights):
    return dict(
        (word, max_bucket_size(word, possible_words, weights))
        for word in ProgressDisplay(possible_words)
    )

    words_and_maxes = list(w2m.items())
    words_and_maxes.sort(key=lambda t: t[1])
    words_and_maxes[:-20:-1]


def get_bucket_sizes(allowed_words, possible_words):
    """
    Returns a (len(allowed_words), 243) shape array reprenting the size of
    word buckets associated with each guess in allowed_words
    """
    weights = np.ones(len(possible_words))
    return get_pattern_distributions(allowed_words, possible_words, weights)


def get_bucket_counts(allowed_words, possible_words):
    """
    Returns the number of separate buckets that each guess in allowed_words
    would separate possible_words into
    """
    bucket_sizes = get_bucket_sizes(allowed_words, possible_words)
    return (bucket_sizes > 0).sum(1)

def prune_allowed_words(allowed_words, possible_words):
    """
    Returns a list of words in allowed_words which have at least one character common with a word in possible_words
    """
    chars = set()
    for word in possible_words:
        letters = [word[i] for i in range(5)]
        chars.update(letters)
    return [word for word in allowed_words if any(c in word for c in chars)]