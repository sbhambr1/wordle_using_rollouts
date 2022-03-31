import numpy as np
from helper_functions.color_patterns import *
from helper_functions.get_data import *
from algorithms.entropy import *
from manimlib import *
from tqdm import tqdm as ProgressDisplay

# Functions to analyze second guesses

def get_average_second_step_entropies(first_guesses, allowed_second_guesses, possible_words, priors):
    result = []
    weights = get_weights(possible_words, priors)
    if weights.sum() == 0:
        return np.zeros(len(first_guesses))

    distributions = get_pattern_distributions(first_guesses, possible_words, weights)
    for first_guess, dist in ProgressDisplay(list(zip(first_guesses, distributions)), leave=False, desc="Searching 2nd step entropies"):
        word_buckets = get_word_buckets(first_guess, possible_words)
        # List of maximum entropies you could achieve in
        # the second step for each pattern you might see
        # after this setp
        ents2 = np.array([
            get_entropies(
                allowed_words=allowed_second_guesses,
                possible_words=bucket,
                weights=get_weights(bucket, priors)
            ).max()
            for bucket in word_buckets
        ])
        # Multiply each such maximal entropy by the corresponding
        # probability of falling into that bucket
        result.append(np.dot(ents2, dist))
    return np.array(result)