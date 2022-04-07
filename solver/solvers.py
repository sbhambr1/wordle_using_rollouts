import warnings

import numpy as np
from algorithms.entropy import *
from algorithms.second_guesses import *
from helper_functions.color_patterns import *
from helper_functions.get_data import *
from manimlib import *
from tqdm import tqdm as ProgressDisplay

warnings.filterwarnings("ignore")

# Solvers

def get_guess_values_array(allowed_words, possible_words, priors, look_two_ahead=False):
    weights = get_weights(possible_words, priors)
    ents1 = get_entropies(allowed_words, possible_words, weights)
    probs = np.array([
        0 if word not in possible_words else weights[possible_words.index(word)]
        for word in allowed_words
    ])

    if look_two_ahead: # Extend this to multi-step lookahead
        # Look two steps out, but restricted to where second guess is
        # amoung the remaining possible words
        ents2 = np.zeros(ents1.shape)
        top_indices = np.argsort(ents1)[-250:]
        ents2[top_indices] = get_average_second_step_entropies(
            first_guesses=np.array(allowed_words)[top_indices],
            allowed_second_guesses=allowed_words,
            possible_words=possible_words,
            priors=priors
        )
        return np.array([ents1, ents2, probs])
    else:
        return np.array([ents1, probs])

def entropy_to_expected_score(ent):
    """
    Based on a regression associating entropies with typical scores
    from that point forward in simulated games, this function returns
    what the expected number of guesses required will be in a game where
    there's a given amount of entropy in the remaining possibilities.
    """
    # Assuming you can definitely get it in the next guess,
    # this is the expected score
    min_score = 2**(-ent) + 2 * (1 - 2**(-ent))

    # To account for the likely uncertainty after the next guess,
    # and knowing that entropy of 11.5 bits seems to have average
    # score of 3.5, we add a line to account
    # we add a line which connects (0, 0) to (3.5, 11.5)
    return min_score + 1.5 * ent / 11.5

def get_expected_scores_with_lookahead(allowed_words, possible_words, look_ahead_steps, expected_scores, n_top_candidates, H0, H1s, weights, word_to_weight):
        for i in range(look_ahead_steps):
            sorted_indices = np.argsort(expected_scores)
            allowed_second_guesses = get_word_list()
            expected_scores += 1  # Push up the rest
            for i in ProgressDisplay(sorted_indices[:n_top_candidates], leave=False):
                guess = allowed_words[i]
                H1 = H1s[i]
                dist = get_pattern_distributions([guess], possible_words, weights)[0]
                buckets = get_word_buckets(guess, possible_words)
                second_guesses = [
                    optimal_guess(allowed_second_guesses, bucket, priors, look_two_ahead=False)
                    for bucket in buckets
                ]
                H2s = [
                    get_entropies([guess2], bucket, get_weights(bucket, priors))[0]
                    for guess2, bucket in zip(second_guesses, buckets)
                ]
                temp_scores = expected_scores
                temp_scores += 1
                H3list = []
                for word in second_guesses:
                    dist2 = get_pattern_distributions([word], possible_words, weights)[0]
                    buckets2 = get_word_buckets(word, possible_words)
                    third_guesses = [
                        optimal_guess(allowed_second_guesses, bucket2, priors, look_two_ahead=False)
                        for bucket2 in buckets2
                    ]
                    H3s = [
                        get_entropies([guess3], bucket2, get_weights(bucket2, priors))[0]
                        for guess3, bucket2 in zip(third_guesses, buckets2)
                    ]
                    H3 = entropy_of_distributions(np.array(H3s))
                    H3list.append(H3)

                prob = word_to_weight.get(guess, 0)
                expected_scores[i] = sum((
                    # 1 times Probability guess1 is correct
                    1 * prob,
                    # 2 times probability guess2 is correct
                    2 * (1 - prob) * sum(
                        p * word_to_weight.get(g2, 0)
                        for p, g2 in zip(dist, second_guesses)
                    ),
                    # 2 plus expected score two steps from now
                    (1 - prob) * (2 + sum(
                        p * (1 - word_to_weight.get(g2, 0)) * entropy_to_expected_score(H0 - H1 - H2 - H3)
                        for p, g2, H2, H3 in zip(dist, second_guesses, H2s, H3list)
                    ))
                ))
        return expected_scores

def get_expected_scores(allowed_words, possible_words, priors,
                        look_two_ahead=False,
                        look_three_ahead=False,
                        ):
    # Currenty entropy of distribution

    weights = get_weights(possible_words, priors)
    H0 = entropy_of_distributions(weights)
    H1s = get_entropies(allowed_words, possible_words, weights)

    word_to_weight = dict(zip(possible_words, weights))
    probs = np.array([word_to_weight.get(w, 0) for w in allowed_words])
    # If this guess is the true answer, score is 1. Otherwise, it's 1 plus
    # the expected number of guesses it will take after getting the corresponding
    # amount of information.
    expected_scores = probs + (1 - probs) * (1 + entropy_to_expected_score(H0 - H1s))

    # For the top candidates, refine the score by looking two steps out
    # This is currently quite slow, and could be optimized to be faster.
    # But why?

    if look_two_ahead and not look_three_ahead:
        return get_expected_scores_with_lookahead(allowed_words, possible_words, look_ahead_steps=1, expected_scores=expected_scores, n_top_candidates=25, H0=H0, H1s=H1s, weights=weights, word_to_weight=word_to_weight)
    elif not look_two_ahead and look_three_ahead:
        return get_expected_scores_with_lookahead(allowed_words, possible_words, look_ahead_steps=2, expected_scores=expected_scores, n_top_candidates=2, H0=H0, H1s=H1s, weights=weights, word_to_weight=word_to_weight)
    else:    
        return expected_scores

def get_score_lower_bounds(allowed_words, possible_words):
    """
    Assuming a uniform distribution on how likely each element
    of possible_words is, this gives the a lower boudn on the
    possible score for each word in allowed_words
    """
    bucket_counts = get_bucket_counts(allowed_words, possible_words)
    N = len(possible_words)
    # Probabilities of getting it in 1
    p1s = np.array([w in possible_words for w in allowed_words]) / N
    # Probabilities of getting it in 2
    p2s = bucket_counts / N - p1s
    # Otherwise, assume it's gotten in 3 (which is optimistics)
    p3s = 1 - bucket_counts / N
    return p1s + 2 * p2s + 3 * p3s

def optimal_guess(allowed_words, possible_words, priors,
                  look_two_ahead=False,
                  look_three_ahead=False,
                  optimize_for_uniform_distribution=False,
                  purely_maximize_information=False,
                  ):
    if purely_maximize_information: 
        if len(possible_words) == 1: # If there's only one possible word, it's the answer
            return possible_words[0] 
        weights = get_weights(possible_words, priors) # Get the weights
        ents = get_entropies(allowed_words, possible_words, weights) # Entropies of each word
        return allowed_words[np.argmax(ents)]

    # Just experimenting here...
    if optimize_for_uniform_distribution: # If optimizing for uniform distribution
        expected_scores = get_score_lower_bounds( 
            allowed_words, possible_words
        )
    else:
        expected_scores = get_expected_scores( 
            allowed_words, possible_words, priors,
            look_two_ahead=look_two_ahead, look_three_ahead=look_three_ahead
        )
    return allowed_words[np.argmin(expected_scores)]

def brute_force_optimal_guess(all_words, possible_words, priors, n_top_picks=10, display_progress=False): #n_top_picks=10 also takes a long time, use this for the end_game
    if len(possible_words) == 0:
        # Doesn't matter what to return in this case, so just default to first word in list.
        return all_words[0]
    # For the suggestions with the top expected scores, just
    # actually play the game out from this point to see what
    # their actual scores are, and minimize.
    expected_scores = get_score_lower_bounds(all_words, possible_words)
    top_choices = [all_words[i] for i in np.argsort(expected_scores)[:n_top_picks]]
    true_average_scores = []
    if display_progress:
        iterable = ProgressDisplay(
            top_choices,
            desc=f"Possibilities: {len(possible_words)}",
            leave=False
        )
    else:
        iterable = top_choices

    for next_guess in iterable:
        scores = []
        for answer in possible_words:
            score = 1
            possibilities = list(possible_words)
            guess = next_guess
            while guess != answer:
                possibilities = get_possible_words(
                    guess, get_pattern(guess, answer),
                    possibilities,
                )
                # Make recursive? If so, we'd want to keep track of
                # the next_guess map and pass it down in the recursive
                # subcalls
                guess = optimal_guess(
                    all_words, possibilities, priors,
                    optimize_for_uniform_distribution=True
                )
                score += 1
            scores.append(score)
        true_average_scores.append(np.mean(scores))
    return top_choices[np.argmin(true_average_scores)]

if __name__ == "__main__":

    all_words = get_word_list()
    wordle_answers = get_word_list(short=True)
    priors = get_true_wordle_prior()
    # priors = get_frequency_based_priors()

    # result = brute_force_optimal_guess(all_words, wordle_answers, priors, n_top_picks=10, display_progress=True) returns 'Salet' with both types of priors

    # result = optimal_guess(all_words, wordle_answers, priors, look_two_ahead=False, optimize_for_uniform_distribution=False, purely_maximize_information=False) # returns 'Soare' with both types of priors
    # result = optimal_guess(all_words, wordle_answers, priors, look_two_ahead=False, optimize_for_uniform_distribution=False, purely_maximize_information=True) # returns 'Soare' with both types of priors
    # result = optimal_guess(all_words, wordle_answers, priors, look_two_ahead=False, optimize_for_uniform_distribution=True, purely_maximize_information=False) # returs 'Trace' with both types of priors
    # result = optimal_guess(all_words, wordle_answers, priors, look_two_ahead=False, optimize_for_uniform_distribution=True, purely_maximize_information=True) # returns 'Soare' with both types of priors

    # result = optimal_guess(all_words, wordle_answers, priors, look_two_ahead=True, optimize_for_uniform_distribution=False, purely_maximize_information=False) # returns 'slate' and 'trace' with both types of priors
    # result = optimal_guess(all_words, wordle_answers, priors, look_two_ahead=True, optimize_for_uniform_distribution=False, purely_maximize_information=True) # returns 'Soare' with both types of priors
    # result = optimal_guess(all_words, wordle_answers, priors, look_two_ahead=True, optimize_for_uniform_distribution=True, purely_maximize_information=False) # returs 'Trace' with both types of priors
    # result = optimal_guess(all_words, wordle_answers, priors, look_two_ahead=True, optimize_for_uniform_distribution=True, purely_maximize_information=True) # returns 'Soare' with both types of priors

    result = optimal_guess(all_words, wordle_answers, priors, look_two_ahead=True, look_three_ahead=False ,optimize_for_uniform_distribution=False, purely_maximize_information=False) # returns 'slate' and 'trace' with both types of priors
    # result = optimal_guess(all_words, wordle_answers, priors, look_two_ahead=False, look_three_ahead=True ,optimize_for_uniform_distribution=False, purely_maximize_information=False) # returns 'slate' and 'trace' with both types of priors
    
    print(result)
