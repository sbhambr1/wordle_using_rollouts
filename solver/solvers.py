import warnings

import numpy as np
from algorithms.entropy import *
from algorithms.second_guesses import *
from helper_functions.color_patterns import *
from helper_functions.get_data import *
from manimlib import *
import random
from tqdm import tqdm as ProgressDisplay
import multiprocessing

warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(
    os.path.dirname(os.path.realpath("__init__.py")),
    "data",
)

APPROXIMATION_CURVE_FILE = os.path.join(DATA_DIR, "approximation_curve_data.npy")

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
    # and knowing that entropy of 12.54 bits seems to have average
    # score of 3.5, we add a line to account
    # we add a line which connects (0, 0) to (3.5, 12.54)
    return min_score + 3.5 * ent / 12.54

def get_score_i(i, expected_scores, allowed_words, possible_words, priors, H0, H1s, weights, word_to_weight, allowed_second_guesses):
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
                        p * (1 - word_to_weight.get(g2, 0)) * entropy_to_expected_score(H0 - H1 - H2)
                        for p, g2, H2 in zip(dist, second_guesses, H2s)
                    ))
                ))
                return expected_scores

def get_expected_scores_with_2lookahead(allowed_words, possible_words, priors, look_ahead_steps, expected_scores, n_top_candidates, H0, H1s, weights, word_to_weight):
        for i in range(look_ahead_steps):
            sorted_indices = np.argsort(expected_scores)
            allowed_second_guesses = get_word_list()
            expected_scores += 1  # Push up the rest
            
            arg_list=[]
            for j in sorted_indices[:n_top_candidates]:
                tup = (j, expected_scores, allowed_words, possible_words, priors, H0, H1s, weights, word_to_weight, allowed_second_guesses)
                arg_list.append(tup)
            
            thread_pool = multiprocessing.Pool()
            expected_scores = thread_pool.starmap(get_score_i, arg_list, chunksize=5)

            # for i in ProgressDisplay(sorted_indices[:n_top_candidates], leave=False):
            #     guess = allowed_words[i]
            #     H1 = H1s[i]
            #     dist = get_pattern_distributions([guess], possible_words, weights)[0]
            #     buckets = get_word_buckets(guess, possible_words)
            #     second_guesses = [
            #         optimal_guess(allowed_second_guesses, bucket, priors, look_two_ahead=False)
            #         for bucket in buckets
            #     ]
            #     H2s = [
            #         get_entropies([guess2], bucket, get_weights(bucket, priors))[0]
            #         for guess2, bucket in zip(second_guesses, buckets)
            #     ]
            
            #     prob = word_to_weight.get(guess, 0)
            #     expected_scores[i] = sum((
            #         # 1 times Probability guess1 is correct
            #         1 * prob,
            #         # 2 times probability guess2 is correct
            #         2 * (1 - prob) * sum(
            #             p * word_to_weight.get(g2, 0)
            #             for p, g2 in zip(dist, second_guesses)
            #         ),
            #         # 2 plus expected score two steps from now
            #         (1 - prob) * (2 + sum(
            #             p * (1 - word_to_weight.get(g2, 0)) * entropy_to_expected_score(H0 - H1 - H2)
            #             for p, g2, H2 in zip(dist, second_guesses, H2s)
            #         ))
            #     ))
        return expected_scores

def get_expected_scores_with_3lookahead(allowed_words, possible_words, priors, look_ahead_steps, expected_scores, n_top_candidates, H0, H1s, weights, word_to_weight):
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
        return get_expected_scores_with_2lookahead(allowed_words, possible_words, priors, look_ahead_steps=1, expected_scores=expected_scores, n_top_candidates=5, H0=H0, H1s=H1s, weights=weights, word_to_weight=word_to_weight)
    elif not look_two_ahead and look_three_ahead:
        return get_expected_scores_with_3lookahead(allowed_words, possible_words, priors, look_ahead_steps=1, expected_scores=expected_scores, n_top_candidates=5, H0=H0, H1s=H1s, weights=weights, word_to_weight=word_to_weight)
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

def get_entropy_scores(allowed_words, possible_words, priors):
    if len(possible_words) == 1: # If there's only one possible word, it's the answer
        return possible_words[0] 
    weights = get_weights(possible_words, priors) # Get the weights
    ents = get_entropies(allowed_words, possible_words, weights) # Entropies of each word
    return ents

def get_expected_scores_using_approximation_curve(all_words, possible_words, pattern):
    """
    Return the scores for all possible guess words in all_words, using the approximation curve.
    """
    curve = np.load(APPROXIMATION_CURVE_FILE)
    expected_scores = np.zeros(len(all_words))
    for i, word in enumerate(all_words):
        possibility_counts = len(get_possible_words(word, pattern, possible_words))
        if possibility_counts == 0:
            expected_scores[i] = np.float('inf')
        elif possibility_counts == 1:
            expected_scores[i] = 1
        elif possibility_counts == 2:
            expected_scores[i] = 1.5
        else:
            expected_scores[i] = np.interp(possibility_counts, curve[:, 0], curve[:, 1])
    return expected_scores

def optimal_guess(allowed_words, possible_words, priors, pattern,
                  look_two_ahead=False,
                  look_three_ahead=False,
                  optimize_using_lower_bound=False,
                  purely_maximize_information=True,
                  use_approximation_curve=False,
                #   top_candidates=100,
                  ):

    if use_approximation_curve:
        if len(possible_words) == 1:
            return possible_words[0]
        expected_scores = get_expected_scores_using_approximation_curve(allowed_words, possible_words, pattern)
        return allowed_words[np.argmin(expected_scores)]

    if purely_maximize_information: 
        if len(possible_words) == 1: # If there's only one possible word, it's the answer
            return possible_words[0] 
        weights = get_weights(possible_words, priors) # Get the weights
        ents = get_entropies(allowed_words, possible_words, weights) # Entropies of each word
        # indices = np.argsort(ents)[::-1] # Sort the words by entropy
        # return top 100 candidates
        # return np.array(allowed_words)[indices[:top_candidates]]
        return allowed_words[np.argmax(ents)]

    # Just experimenting here...
    if optimize_using_lower_bound: # If optimizing for uniform distribution
        expected_scores = get_score_lower_bounds( 
            allowed_words, possible_words
        )
    else:
        expected_scores = get_expected_scores( 
            allowed_words, possible_words, priors,
            look_two_ahead=look_two_ahead, look_three_ahead=look_three_ahead
        )
    return allowed_words[np.argmin(expected_scores)]

def max_info_gain_guess(allowed_words, possible_words, priors):
    if len(possible_words) == 1: # If there's only one possible word, it's the answer
            return possible_words[0] 
    weights = get_weights(possible_words, priors) # Get the weights
    ents = get_entropies(allowed_words, possible_words, weights) # Entropies of each word
    return allowed_words[np.argmax(ents)]

def min_expected_score_guess(allowed_words, possible_words, priors):
    expected_scores = get_expected_scores(allowed_words, possible_words, priors)
    return allowed_words[np.argmin(expected_scores)]

def approx_curve_guess(allowed_words, possible_words, pattern):
    if len(possible_words) == 1:
            return possible_words[0]
    expected_scores = get_expected_scores_using_approximation_curve(allowed_words, possible_words, pattern)
    return allowed_words[np.argmin(expected_scores)]

def solve_simulation(guess, answer, guesses, patterns, possibilities, priors, all_words, hard_mode=False, purely_maximize_information=False, expected_scores_heuristic=False, super_heuristic=False, use_approximation_curve=False):

    score = 1

    while guess != answer:

        possibilities = get_possible_words(guess, get_pattern(guess, answer), possibilities)

        if len(possibilities)==1:
            guess = answer
            continue

        pattern = get_pattern(guess, answer)
        guesses.append(guess)
        patterns.append(pattern)
        choices = prune_allowed_words(all_words, possibilities)

        if hard_mode:
            for guess, pattern in zip(guesses, patterns):
                choices = get_possible_words(guess, pattern, choices)

        if super_heuristic:
            if purely_maximize_information:
                guess = max_info_gain_guess(allowed_words = choices, possible_words = possibilities, priors = priors)
            elif expected_scores_heuristic:
                guess = min_expected_score_guess(allowed_words = choices, possible_words = possibilities, priors = priors)
        else:
            if purely_maximize_information:
                guess = max_info_gain_guess(allowed_words = choices, possible_words = possibilities, priors = priors)

            if expected_scores_heuristic:
                guess = min_expected_score_guess(allowed_words = choices, possible_words = possibilities, priors = priors)

            if use_approximation_curve:
                guess = approx_curve_guess(allowed_words = choices, possible_words = possibilities, pattern = pattern)

        score += 1

    return score

def brute_force_optimal_guess(all_words, possible_words, priors, n_top_picks=10, pattern=None, 
                            super_heuristic = False,
                            optimize_using_lower_bound=False, 
                            purely_maximize_information=False, 
                            use_approximation_curve=False, 
                            expected_scores_heuristic = False,
                            display_progress=False, 
                            hard_mode=False): #n_top_picks=10 also takes a long time, use this for the end_game
    
    
    if len(possible_words) == 1: # If there's only one possible word, it's the answer
        return possible_words[0]

    # For max information gain heursitic only:

    if purely_maximize_information:
        expected_scores = get_entropy_scores(all_words, possible_words, priors) 
        top_choices = [all_words[i] for i in np.argsort(expected_scores)[::-1][:n_top_picks]] 
        # top_entropies = [expected_scores[i] for i in np.argsort(expected_scores)[::-1][:n_top_picks]] 
        total_max_info_score = []

    ## TODO: @Sid: shuffle elements of top choices that have same entropy. 

    # For expected scores from Grant's formula - IMPORTANT - Our initialization of top k words is different than his:

    if expected_scores_heuristic:
        expected_scores = get_expected_scores(all_words, possible_words, priors) 
        top_choices = [all_words[i] for i in np.argsort(expected_scores)[:n_top_picks]] 
        # top_entropies = [expected_scores[i] for i in np.argsort(expected_scores)[:n_top_picks]]
        total_min_expected_score = []

    # For the super heuristic:

    if super_heuristic:
        expected_scores = get_expected_scores(all_words, possible_words, priors) 
        top_choices = [all_words[i] for i in np.argsort(expected_scores)[:n_top_picks]]
        total_max_info_score = []
        total_min_expected_score = []

    # for expected scores from approximation curve:

    if use_approximation_curve:
        expected_scores = get_expected_scores_using_approximation_curve(all_words, possible_words, pattern) 
        # TODO: @Sid: shuffle elements of top choices that have same expected score.
        top_choices = [all_words[i] for i in np.argsort(expected_scores)[:n_top_picks]] 
        total_approx_curve_score = []

    total_max_info_score = []
    total_min_expected_score = [] 
    total_approx_curve_score = []
    total_super_heuristic_score = []

    if display_progress:
        iterable = ProgressDisplay(
            top_choices,
            desc=f"Possibilities: {len(possible_words)}",
            leave=False
        )
    else:
        iterable = top_choices

    for next_guess in iterable:

        max_info_scores = []
        min_expected_scores = [] 
        approx_curve_scores = []
        super_heuristic_scores = []
        
        for answer in possible_words:
            guesses = []
            patterns = []

            possibilities = list(possible_words)
            guess = next_guess

            if purely_maximize_information:
                info_score = solve_simulation(guess, answer, guesses, patterns, possibilities, priors, all_words, hard_mode=hard_mode, purely_maximize_information=purely_maximize_information, expected_scores_heuristic=False, super_heuristic=False, use_approximation_curve=False)
                max_info_scores.append(info_score)

            if expected_scores_heuristic:
                expected_score = solve_simulation(guess, answer, guesses, patterns, possibilities, priors, all_words, hard_mode=hard_mode, purely_maximize_information=False, expected_scores_heuristic=expected_scores_heuristic, super_heuristic=False, use_approximation_curve=False)
                min_expected_scores.append(expected_score)

            if use_approximation_curve:
                approx_curve_score = solve_simulation(guess, answer, guesses, patterns, possibilities, priors, all_words, hard_mode=hard_mode, purely_maximize_information=False, expected_scores_heuristic=False, super_heuristic=False, use_approximation_curve=use_approximation_curve)
                approx_curve_scores.append(approx_curve_score)

            if super_heuristic:
                info_score = solve_simulation(guess, answer, guesses, patterns, possibilities, priors, all_words, hard_mode=hard_mode, purely_maximize_information=True, expected_scores_heuristic=False, super_heuristic=super_heuristic, use_approximation_curve=False)
                expected_score = solve_simulation(guess, answer, guesses, patterns, possibilities, priors, all_words, hard_mode=hard_mode, purely_maximize_information=False, expected_scores_heuristic=True, super_heuristic=super_heuristic, use_approximation_curve=False)
                super_heuristic_scores.append(min(info_score, expected_score))

        total_max_info_score.append(np.sum(max_info_scores)+1) if purely_maximize_information or super_heuristic else None
        total_min_expected_score.append(np.sum(min_expected_scores)+1) if expected_scores_heuristic or super_heuristic else None
        total_approx_curve_score.append(np.sum(total_approx_curve_score)+1)  if use_approximation_curve else None
        total_super_heuristic_score.append(np.sum(super_heuristic_scores)+1) if super_heuristic else None

    max_info_score_indices = np.where(total_max_info_score == np.amin(total_max_info_score))[0] if purely_maximize_information or super_heuristic else None
    min_expected_score_indices = np.where(total_min_expected_score == np.amin(total_min_expected_score))[0] if expected_scores_heuristic or super_heuristic else None
    approx_curve_score_indices = np.where(total_approx_curve_score == np.amin(total_approx_curve_score))[0] if use_approximation_curve else None
    super_heuristic_indices = np.where(total_super_heuristic_score == np.amin(total_super_heuristic_score))[0] if super_heuristic else None

    ## TODO: @AB: try ascending order instead of the default descending alphabetical order. I expect performance to improve.

    if purely_maximize_information:
        return top_choices[random.choice(max_info_score_indices)]
    elif expected_scores_heuristic:
        return top_choices[random.choice(min_expected_score_indices)]
    elif use_approximation_curve:
        return top_choices[random.choice(approx_curve_score_indices)]
    else: # superheuristic case
        return top_choices[random.choice(super_heuristic_indices)]

if __name__ == "__main__":

    all_words = get_word_list()
    wordle_answers = get_word_list(short=True)
    priors = get_true_wordle_prior()
    # priors = get_frequency_based_priors()

    # result = brute_force_optimal_guess(all_words, wordle_answers, priors, n_top_picks=10, display_progress=True) returns 'Salet' with both types of priors

    # result = optimal_guess(all_words, wordle_answers, priors, look_two_ahead=False, optimize_using_lower_bound=False, purely_maximize_information=False) # returns 'Soare' with both types of priors
    # result = optimal_guess(all_words, wordle_answers, priors, look_two_ahead=False, optimize_using_lower_bound=False, purely_maximize_information=True) # returns 'Soare' with both types of priors
    # result = optimal_guess(all_words, wordle_answers, priors, look_two_ahead=False, optimize_using_lower_bound=True, purely_maximize_information=False) # returs 'Trace' with both types of priors
    # result = optimal_guess(all_words, wordle_answers, priors, look_two_ahead=False, optimize_using_lower_bound=True, purely_maximize_information=True) # returns 'Soare' with both types of priors

    # result = optimal_guess(all_words, wordle_answers, priors, look_two_ahead=True, optimize_using_lower_bound=False, purely_maximize_information=False) # returns 'slate' and 'trace' with both types of priors
    # result = optimal_guess(all_words, wordle_answers, priors, look_two_ahead=True, optimize_using_lower_bound=False, purely_maximize_information=True) # returns 'Soare' with both types of priors
    # result = optimal_guess(all_words, wordle_answers, priors, look_two_ahead=True, optimize_using_lower_bound=True, purely_maximize_information=False) # returs 'Trace' with both types of priors
    # result = optimal_guess(all_words, wordle_answers, priors, look_two_ahead=True, optimize_using_lower_bound=True, purely_maximize_information=True) # returns 'Soare' with both types of priors

    # result = optimal_guess(all_words, wordle_answers, priors, look_two_ahead=True, look_three_ahead=False ,optimize_using_lower_bound=False, purely_maximize_information=False) # returns 'slate' and 'trace' with both types of priors
    # result = optimal_guess(all_words, wordle_answers, priors, look_two_ahead=False, look_three_ahead=True ,optimize_using_lower_bound=False, purely_maximize_information=False) # returns 'slate' and 'trace' with both types of priors
    
    result = optimal_guess(all_words, wordle_answers, priors, purely_maximize_information=True)

    print(result)
