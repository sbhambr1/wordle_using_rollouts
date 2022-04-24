import itertools as it
import json
import logging as log
import os
import sys

sys.path.append(".")
#sys.path.append(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'color_patterns')))
import random
import time
import warnings

import manimlib.utils as mu
import numpy as np
from algorithms.entropy import *
from algorithms.second_guesses import *
from helper_functions.color_patterns import *
from helper_functions.get_data import *
from manimlib import *
from tqdm import tqdm as ProgressDisplay

from solver.solvers import *

warnings.filterwarnings("ignore")

# Run simulated wordle games

def get_two_step_score_lower_bound(first_guess, allowed_words, possible_words): # not being used right now, and is a utility function for additional analysis
    """
    Useful to prove what the minimum possible average score could be
    for a given initial guess
    """
    N = len(possible_words)
    buckets = get_word_buckets(first_guess, possible_words)
    min_score = 0
    for bucket in buckets:
        if len(bucket) == 0:
            continue
        lower_bounds = get_score_lower_bounds(allowed_words, bucket)
        min_score += (len(bucket) / N) * lower_bounds.min()
    p = (1 / len(possible_words)) * (first_guess in possible_words) 
    return p + (1 - p) * (1 + min_score)


def find_top_scorers(n_top_candidates=100, quiet=True, file_ext="", **kwargs): # not being used right now, but seems like a utility function for additional analysis
    # Run find_best_two_step_entropy first
    file = os.path.join(mu.directories.get_directories()["data"], "wordle", "best_double_entropies.json")
    with open(file) as fp:
        double_ents = json.load(fp)

    answers = get_word_list(short=True)
    priors = get_true_wordle_prior()
    guess_to_score = {}
    guess_to_dist = {}

    for row in ProgressDisplay(double_ents[:n_top_candidates]):
        first_guess = row[0]
        result, decision_map = simulate_games(
            first_guess, priors=priors,
            optimize_for_uniform_distribution=True,
            quiet=quiet,
            **kwargs,
        )
        average = result["average_score"]
        total = int(np.round(average * len(answers)))
        guess_to_score[first_guess] = total
        guess_to_dist[first_guess] = result["score_distribution"]

    top_scorers = sorted(list(guess_to_score.keys()), key=lambda w: guess_to_score[w])
    result = [[w, guess_to_score[w], guess_to_dist[w]] for w in top_scorers]

    file = os.path.join(
        mu.directories.get_directories()["data"], "wordle",
        "best_scores" + file_ext + ".json",
    )
    with open(file, 'w') as fp:
        json.dump(result, fp)

    return result


def find_best_two_step_entropy(): # not being used right now, but seems like a utility function for additional analysis
    words = get_word_list()
    answers = get_word_list(short=True)
    priors = get_true_wordle_prior()

    ents = get_entropies(words, answers, get_weights(answers, priors))
    sorted_indices = np.argsort(ents)
    top_candidates = np.array(words)[sorted_indices[:-250:-1]]
    top_ents = ents[sorted_indices[:-250:-1]]

    ent_file = os.path.join(mu.directories.get_directories()["data"], "wordle", "best_entropies.json")
    with open(ent_file, 'w') as fp:
        json.dump([[tc, te] for tc, te in zip(top_candidates, top_ents)], fp)

    ents2 = get_average_second_step_entropies(
        top_candidates, words, answers, priors,
    )

    total_ents = top_ents + ents2
    sorted_indices2 = np.argsort(total_ents)

    double_ents = [
        [top_candidates[i], top_ents[i], ents2[i]]
        for i in sorted_indices2[::-1]
    ]

    ent2_file = os.path.join(mu.directories.get_directories()["data"], "wordle", "best_double_entropies.json")
    with open(ent2_file, 'w') as fp:
        json.dump(double_ents, fp)

    return double_ents


def find_smallest_second_guess_buckets(n_top_picks=100): # not being used right now, but seems like a utility function for additional analysis
    all_words = get_word_list()
    possibilities = get_word_list(short=True)
    priors = get_true_wordle_prior()
    weights = get_weights(possibilities, priors)

    dists = get_pattern_distributions(all_words, possibilities, weights)
    sorted_indices = np.argsort((dists**2).sum(1))

    top_indices = sorted_indices[:n_top_picks]
    top_picks = np.array(all_words)[top_indices]
    top_dists = dists[top_indices]
    # Figure out the average number of matching words there will
    # be after two steps of game play
    avg_ts_buckets = []
    for first_guess, dist in ProgressDisplay(list(zip(top_picks, top_dists))):
        buckets = get_word_buckets(first_guess, possibilities)
        avg_ts_bucket = 0
        for p, bucket in zip(dist, buckets):
            weights = get_weights(bucket, priors)
            sub_dists = get_pattern_distributions(all_words, bucket, weights)
            min_ts_bucket = len(bucket) * (sub_dists**2).sum(1).min()
            avg_ts_bucket += p * min_ts_bucket
        avg_ts_buckets.append(avg_ts_bucket)

    result = []
    for j in np.argsort(avg_ts_buckets):
        i = top_indices[j]
        result.append((
            # Word
            all_words[i],
            # Average bucket size after first guess
            len(possibilities) * (dists[i]**2).sum(),
            # Average bucket size after second, with optimal
            # play.
            avg_ts_buckets[j],
        ))
    return result


def get_optimal_second_guess_map(first_guess, n_top_picks=10, regenerate=False): # not being used right now, but finds answer in a different way probably?
    with open(SECOND_GUESS_MAP_FILE) as fp:
        all_sgms = json.load(fp)

    if first_guess in all_sgms and not regenerate:
        return all_sgms[first_guess]

    log.info("\n".join([
        f"Generating optimal second guess map for {first_guess}.",
        "This involves brute forcing many simulations",
        "so can take a little while."
    ]))

    sgm = [""] * 3**5
    all_words = get_word_list()
    wordle_answers = get_word_list(short=True)
    priors = get_true_wordle_prior()

    buckets = get_word_buckets(first_guess, wordle_answers)
    for pattern, bucket in ProgressDisplay(list(enumerate(buckets)), leave=False):
        sgm[pattern] = brute_force_optimal_guess(
            all_words, bucket, priors,
            n_top_picks=n_top_picks,
            display_progress=True
        )

    # Save to file
    with open(SECOND_GUESS_MAP_FILE) as fp:
        all_sgms = json.load(fp)
    all_sgms[first_guess] = sgm
    with open(SECOND_GUESS_MAP_FILE, 'w') as fp:
        json.dump(all_sgms, fp)

    return sgm


def gather_entropy_to_score_data(first_guess="crane", priors=None): # not being used right now, but finds answer in a different way
    words = get_word_list()
    answers = get_word_list(short=True)
    if priors is None:
        priors = get_true_wordle_prior()

    # List of entropy/score pairs
    ent_score_pairs = []

    for answer in ProgressDisplay(answers):
        score = 1
        possibilities = list(filter(lambda w: priors[w] > 0, words))
        guess = first_guess
        guesses = []
        entropies = []
        while True:
            guesses.append(guess)
            weights = get_weights(possibilities, priors)
            entropies.append(entropy_of_distributions(weights))
            if guess == answer:
                break
            possibilities = get_possible_words(
                guess, get_pattern(guess, answer), possibilities
            )
            guess = optimal_guess(words, possibilities, priors)
            score += 1

        for sc, ent in zip(it.count(1), reversed(entropies)):
            ent_score_pairs.append((ent, sc))

    with open(ENT_SCORE_PAIRS_FILE, 'w') as fp:
        json.dump(ent_score_pairs, fp)

    return ent_score_pairs


def simulate_games(first_guess=None,
                   priors=None,
                   look_two_ahead=False,
                   optimize_for_uniform_distribution=False,
                   second_guess_map=None,
                   exclude_seen_words=False,
                   test_set=None,
                   shuffle=False,
                   hard_mode=False,
                   purely_maximize_information=False,
                   brute_force_optimize=False,
                   brute_force_depth=10,
                   results_file=None,
                   next_guess_map_file=None,
                   quiet=True,
                   ):
    all_words = get_word_list(short=False)
    short_word_list = get_word_list(short=True)

    if first_guess is None: 
        first_guess = optimal_guess(
            all_words, all_words, priors
            # **choice_config
        )

    if priors is None:
        # priors = get_frequency_based_priors()
        priors = get_true_wordle_prior()

    if test_set is None:
        test_set = short_word_list

    if shuffle:
        random.shuffle(test_set)

    seen = set()

    # Function for choosing the next guess, with a dict to cache
    # and reuse results that are seen multiple times in the sim
    next_guess_map = {}

    def get_next_guess(guesses, patterns, possibilities):
        phash = "".join(
            str(g) + "".join(map(str, pattern_to_int_list(p)))
            for g, p in zip(guesses, patterns) 
        ) 
        if second_guess_map is not None and len(patterns) == 1: 
            next_guess_map[phash] = second_guess_map[patterns[0]] 
        if phash not in next_guess_map: 
            choices = prune_allowed_words(all_words, possibilities)
            # print('inside get_next_guess 3b1b line 277')
            # print(len(choices))
            if hard_mode:
                for guess, pattern in zip(guesses, patterns):
                    choices = get_possible_words(guess, pattern, choices)
            if brute_force_optimize:
                next_guess_map[phash] = brute_force_optimal_guess(
                    choices, possibilities, priors,
                    n_top_picks=brute_force_depth,
                )
            else:
                next_guess_map[phash] = optimal_guess(
                    choices, possibilities, priors,
                    look_two_ahead=look_two_ahead,
                    look_three_ahead=False,
                    purely_maximize_information=True,
                    optimize_for_uniform_distribution=optimize_for_uniform_distribution,
                )
        return next_guess_map[phash]

    # Go through each answer in the test set, play the game,
    # and keep track of the stats.
    scores = np.zeros(0, dtype=int)
    game_results = []
    for answer in ProgressDisplay(test_set, leave=False, desc=" Trying all wordle answers"):
        guesses = []
        patterns = []
        possibility_counts = []
        possibilities = list(filter(lambda w: priors[w] > 0, all_words))

        if exclude_seen_words:
            possibilities = list(filter(lambda w: w not in seen, possibilities))

        score = 1
        guess = first_guess
        while guess != answer:
            pattern = get_pattern(guess, answer)
            guesses.append(guess)
            patterns.append(pattern)
            possibilities = get_possible_words(guess, pattern, possibilities)
            possibility_counts.append(len(possibilities))
            score += 1
            if score >= 3:
                # do bruteforce optimization
                phash = "".join(
                str(g) + "".join(map(str, pattern_to_int_list(p)))
                for g, p in zip(guesses, patterns))
                if second_guess_map is not None and len(patterns) == 1: 
                    next_guess_map[phash] = second_guess_map[patterns[0]] 
                if phash not in next_guess_map:
                    choices = prune_allowed_words(all_words, possibilities)
                    # print('inside rollout 3b1b line 327')
                    # print(len(choices))
                    next_guess_map[phash] = brute_force_optimal_guess(
                    choices, possibilities, priors,
                    n_top_picks=brute_force_depth)
                guess = next_guess_map[phash]
            else:
                guess = get_next_guess(guesses, patterns, possibilities)

        # Accumulate stats
        scores = np.append(scores, [score])
        score_dist = [
            int((scores == i).sum())
            for i in range(1, scores.max() + 1)
        ]
        total_guesses = scores.sum()
        average = scores.mean()
        seen.add(answer)

        game_results.append(dict(
            score=int(score),
            answer=answer,
            guesses=guesses,
            patterns=list(map(int, patterns)),
            reductions=possibility_counts,
        ))
        # Print outcome
        # if not quiet:
        #     message = "\n".join([
        #         "",
        #         f"Score: {score}",
        #         f"Answer: {answer}",
        #         f"Guesses: {guesses}",
        #         f"Reductions: {possibility_counts}",
        #         *patterns_to_string((*patterns, 3**5 - 1)).split("\n"),
        #         *" " * (6 - len(patterns)),
        #         f"Distribution: {score_dist}",
        #         f"Total guesses: {total_guesses}",
        #         f"Average: {average}",
        #         *" " * 2,
        #     ])
        #     if answer is not test_set[0]:
        #         # Move cursor back up to the top of the message
        #         n = len(message.split("\n")) + 1
        #         print(("\033[F\033[K") * n)
        #     else:
        #         print("\r\033[K\n")
        #     print(message)

    final_result = dict(
        score_distribution=score_dist,
        total_guesses=int(total_guesses),
        average_score=float(scores.mean()),
        game_results=game_results,
    )

    # Save results
    # for obj, file in [(final_result, results_file), (next_guess_map, next_guess_map_file)]:
    #     if file:
    #         path = os.path.join(DATA_DIR, "simulation_results", file)
    #         with open(path, 'w') as fp:
    #             json.dump(obj, fp)

    return final_result, next_guess_map




if __name__ == "__main__":
    start_time = time.time()
    first_guesses = ["salet", "soare", "trace", "slate", "crane", "dealt", "carse"]
    for first_guess in first_guesses:
        print(first_guess)
        results, decision_map = simulate_games(
            first_guess=first_guess,
            priors=None,
            look_two_ahead=False,
            optimize_for_uniform_distribution=False,
            brute_force_depth=10,
            hard_mode=True,
        )
        print(results["score_distribution"], results["total_guesses"], results["average_score"])

    # for first_guess in first_guesses:
    #     print(first_guess)
    #     results, decision_map = simulate_games(
    #         first_guess=first_guess,
    #         priors=None,
    #         look_two_ahead=False,
    #         optimize_for_uniform_distribution=False,
    #         hard_mode=True,
    #     )
    #     print(results["score_distribution"], results["total_guesses"], results["average_score"])

    print("--- %s seconds ---" % (time.time() - start_time))

