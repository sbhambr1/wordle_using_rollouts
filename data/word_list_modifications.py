import random

random.seed(10)

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

def flag_guess_words(guess_words, mystery_words):
    '''Returns a hashmap of words in guess_words that are in mystery_words.'''

    hashmap = {}
    for word in guess_words:
        if word in mystery_words:
            hashmap[word] = True
        else:
            hashmap[word] = False
    return hashmap

def check_modified_words(guess_words, mystery_words):
    '''Checks if the modified mystery words are same in modified guess words list.'''

    count = 0
    for word in guess_words:
        if word in mystery_words:
            count += 1
    return count

def main():
    guess_words = get_words('data/modified_guess_words_6letter.txt')
    mystery_words = get_words('data/modified_mystery_words_6letter.txt')
    hashmap = flag_guess_words(guess_words, mystery_words)

    modified_guess_words = []
    modified_mystery_words = []
    allowed_chars = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

    for word in guess_words: 
        modified_word = word + random.choice(allowed_chars)
        modified_guess_words.append(str(modified_word))
        if hashmap[word]:
            modified_mystery_words.append(str(modified_word))
    
    save_words('data/modified_guess_words_7letter.txt', modified_guess_words)
    save_words('data/modified_mystery_words_7letter.txt', modified_mystery_words)

    print(check_modified_words(modified_guess_words, modified_mystery_words))

if __name__ == '__main__':
    main()
