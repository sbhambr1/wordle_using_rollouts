import itertools as it
import json
import logging as log
import os
import sys

from cv2 import exp

sys.path.append(".")
#sys.path.append(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'color_patterns')))
import random
import time
import warnings

import manimlib.utils as mu
import numpy as np
import pandas as pd
from algorithms.entropy import *
from algorithms.second_guesses import *
from helper_functions.color_patterns import *
from helper_functions.get_data import *
from manimlib import *
from tqdm import tqdm as ProgressDisplay

from solver.solvers import *

warnings.filterwarnings("ignore")

# Run simulated wordle games

def simulate_games(first_guess=None,
                   priors=None,
                   look_two_ahead=False,
                   optimize_using_lower_bound=False,
                   second_guess_map=None,
                   exclude_seen_words=False,
                   test_set=None,
                   shuffle=False,
                   hard_mode=False,
                   super_heuristic=False,
                   purely_maximize_information=False,
                   use_approximation_curve=False,
                   expected_scores_heuristic = False,
                   brute_force_optimize=False,
                   rollout_begin_at=3,
                   rollout_top_k=10,
                   test_mode=False,
                   results_file=None,
                   next_guess_map_file=None,
                   quiet=True,
                   track_failures=False,
                   num_times_word_in_top_k=0,
                   num_times_word_finally_selected=0
                   ):

    all_words = get_word_list(short=False)
    short_word_list = get_word_list(short=True)

    # if track_failures:
    tracking_dict = {}

    if first_guess is None: 
        first_guess = optimal_guess(
            all_words, all_words, priors
            # **choice_config
        )

    if priors is None:
        # priors = get_frequency_based_priors()
        priors = get_true_wordle_prior()

    if test_set is None:
        test_set = short_word_list  ## set test set as actual Wordle answer mystery list

    if shuffle:
        print("shuffled")
        random.shuffle(test_set)

    seen = set()

    # Function for choosing the next guess, with a dict to cache
    # and reuse results that are seen multiple times in the sim
    next_guess_map = {}

    def get_next_guess(guesses, patterns, possibilities, pattern):
        phash = "".join(
            str(g) + "".join(map(str, pattern_to_int_list(p)))
            for g, p in zip(guesses, patterns) 
        ) 
        
        choices = prune_allowed_words(all_words, possibilities)
        
        ## replacing next_guess_map[phash] with computed_guess
        if hard_mode:
            for guess, pattern in zip(guesses, patterns):
                choices = get_possible_words(guess, pattern, choices)
        if brute_force_optimize:
            computed_guess = brute_force_optimal_guess(
                choices, possibilities, priors,
                n_top_picks=rollout_top_k,
            )
            guess=computed_guess
        else:
            computed_guess = optimal_guess(
                choices, possibilities, priors, pattern,
                look_two_ahead=look_two_ahead,
                look_three_ahead=False,
                purely_maximize_information=purely_maximize_information,
                optimize_using_lower_bound=optimize_using_lower_bound,
                use_approximation_curve=use_approximation_curve
            )
            guess=computed_guess
        return guess

    # Go through each answer in the test set, play the game,
    # and keep track of the stats.

    scores = np.zeros(0, dtype=int)
    game_results = []
    mystery_list_lengths = []
    
    for answer in ProgressDisplay(test_set, leave=False, desc=" Trying all wordle answers"):
        
        guesses = []
        patterns = []
        possibility_counts = []
        possibilities = list(filter(lambda w: priors[w] > 0, all_words))  ## here you are defining the priors over possible answers

        if exclude_seen_words:
            possibilities = list(filter(lambda w: w not in seen, possibilities))
        # answer = "bound" ##checking
        possibility_counts.append(len(possibilities))
        score = 1
        guess = first_guess
        while guess != answer:
            pattern = get_pattern(guess, answer)
            guesses.append(guess)
            patterns.append(pattern)
            possibilities = get_possible_words(guess, pattern, possibilities)
            possibility_counts.append(len(possibilities))
            score += 1
            if len(possibilities) == 1:
                guess = possibilities[0]
            else:
                if score >= rollout_begin_at:
                    # do bruteforce optimization
                    phash = "".join(
                    str(g) + "".join(map(str, pattern_to_int_list(p)))
                    for g, p in zip(guesses, patterns))
                    
                    # if phash not in next_guess_map:
                    choices = prune_allowed_words(all_words, possibilities)
                    
                    if hard_mode:
                        for guess, pattern in zip(guesses, patterns):
                            choices = get_possible_words(guess, pattern, choices)

                    results = one_step_lookahead_minimization(guess_words=choices,
                                                                    mystery_words=possibilities,
                                                                    priors=priors,
                                                                    heuristic='max_info_gain', #min_expected_scores, max_info_gain, most_rapid_decrease, greatest_exp_prob
                                                                    top_picks=rollout_top_k,
                                                                    pattern=pattern,
                                                                    hard_mode=hard_mode,
                                                                    num_times_word_in_top_k=0,
                                                                    num_times_word_finally_selected=0)
                    
                    computed_guess, top_k_counter, final_selection_counter = results[0], results[1], results[2]
                    num_times_word_in_top_k += top_k_counter
                    num_times_word_finally_selected += final_selection_counter

                    # computed_guess = brute_force_optimal_guess(
                    # choices, possibilities, priors,
                    # n_top_picks=rollout_top_k, 
                    # pattern=pattern,
                    # super_heuristic=super_heuristic,
                    # optimize_using_lower_bound=optimize_using_lower_bound,
                    # purely_maximize_information=purely_maximize_information,
                    # use_approximation_curve=use_approximation_curve,
                    # expected_scores_heuristic=expected_scores_heuristic,
                    # hard_mode=hard_mode)

                    # Experiments for using ONLY base heuristics.

                    # computed_guess = min_expected_score_guess(choices, possibilities, priors)

                    # computed_guess = max_info_gain_guess(choices, possibilities, priors)

                    # computed_guess = most_rapid_decrease_guess(choices, possibilities, priors)

                    # computed_guess = greatest_exp_prob_guess(choices, possibilities, priors)

                    guess=computed_guess
                    # guess = next_guess_map[phash]
                else:
                    computed_guess = get_next_guess(guesses, patterns, possibilities, pattern)
                    guess=computed_guess
        guesses.append(guess)

        if track_failures:
            if score>6:
                tracking_dict[answer] = guesses

        scores = np.append(scores, [score])
        score_dist = [
            int((scores == i).sum())
            for i in range(1, scores.max() + 1)
        ]
        total_guesses = scores.sum()
        average = scores.mean()
        seen.add(answer)
        
        # game_results.append(dict(
        #     score=int(score),
        #     answer=answer,
        #     guesses=guesses,
        #     patterns=list(map(int, patterns)),
        #     reductions=possibility_counts,
        # ))

        mystery_list_lengths.append(possibility_counts)

    final_result = dict(
        score_distribution=score_dist,
        total_guesses=int(total_guesses),
        average_score=float(scores.mean()),
        # game_results=game_results,
        mystery_list_lengths=mystery_list_lengths,
    )

    return final_result, tracking_dict, num_times_word_in_top_k, num_times_word_finally_selected

if __name__ == "__main__":
    start_time = time.time()

    first_guesses = ["salet", "soare"] #testing

    # first_guesses = ['salet', 'reast', 'crate', 'trace', 'slate', 'trape', 'slane', 'prate', 'crane', 'carle', 'train',
    #                 'raise', 'clout', 'nymph'] #existing works

    # first_guesses = ['saletnf', 'soaremx']

    # first_guesses = ['scarpe', 'adsale']

    saving_results_to_csv = False

    for first_guess in first_guesses:

        print(first_guess)
        
        results, tracking_failure, num_times_word_in_top_k, num_times_word_finally_selected = simulate_games(
            first_guess=first_guess,
            priors=None,
            rollout_begin_at=2,
            rollout_top_k=10,
            hard_mode=False,
            test_mode=False,
            track_failures=True,
            num_times_word_in_top_k=0,
            num_times_word_finally_selected=0
        )
        
        print(results["score_distribution"], results["total_guesses"], results["average_score"], num_times_word_in_top_k, num_times_word_finally_selected)

        if saving_results_to_csv:

            score_distributions = []
            total_guesses = []
            average_scores = []
            game_results = []
            mystery_list_lengths = []

            score_distributions.append(results["score_distribution"])
            total_guesses.append(results["total_guesses"])
            average_scores.append(results["average_score"])
            mystery_list_lengths.append(results["mystery_list_lengths"])

            max_info_gain_rollout_results_dict =  dict(score_distribution=score_distributions, total_guesses=total_guesses, average_score=average_scores, mystery_list_lengths=mystery_list_lengths)
            max_info_gain_rollout_results_df = pd.DataFrame(max_info_gain_rollout_results_dict, columns=["score_distribution", "total_guesses", "average_score", "mystery_list_lengths"])
            max_info_gain_rollout_results_df.to_csv("max_info_gain_rollout_results.csv", index=False)

        if tracking_failure is not None:

            print("Failure case guesses: ")
            print(tracking_failure)

    print("--- %s seconds ---" % (time.time() - start_time))

