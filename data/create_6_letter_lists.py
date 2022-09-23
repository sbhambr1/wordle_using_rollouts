import random

random.seed(0)

def get_words(filename):
    '''Reads a file and returns a list of words.'''
    
    with open(filename, 'r') as f:
        return [line.strip() for line in f.readlines()]

def save_words(filename, words):
    '''Saves a list of words to a file.'''
    
    with open(filename, 'w') as f:
        for word in words:
            f.write(word)
            f.write('\n')

def get_guess_words_list(source_list):
    '''Creates a guess word list of 6-letter words equal in length to 5-letter guess word list.'''
    
    count = 12972
    guess_words_list = []
    
    while len(guess_words_list)!=count:
        guess_words_list.append(random.choice(source_list))

    return guess_words_list

def get_mystery_words_list(guess_words_list):
    '''Creates a mystery word list of 6-letter words equal in length to 5-letter mystery word list.'''
    
    count = 2315
    mystery_words_list = []
    
    while len(mystery_words_list)!=count:
        mystery_words_list.append(random.choice(guess_words_list))

    return mystery_words_list

def main():
    source_list = sorted(get_words('data/norvig_6letter_source.txt'))
    guess_words_list = sorted(get_guess_words_list(source_list))
    mystery_words_list = sorted(get_mystery_words_list(guess_words_list))

    save_words('data/actual_guess_words_6letter.txt', guess_words_list)
    save_words('data/actual_mystery_words_6letter.txt', mystery_words_list)

    print('Length of 6-letter guess word list: ', len(guess_words_list))
    print('Length of 6-letter mystery word list: ', len(mystery_words_list))
    
if __name__ == '__main__':
    main()