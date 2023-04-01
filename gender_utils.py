import spacy

# Load the SpaCy English model
nlp = spacy.load("en_core_web_lg")

masculine_keywords = [
    "man", "men", "boy", "boys", "male", "males", "gentleman", "gentlemen",
    "guy", "guys", "lad", "lads", "youth", "youths", "son", "sons",
    "father", "fathers", "uncle", "uncles", "nephew", "nephews",
    "brother", "brothers", "grandfather", "grandfathers", "grandson", "grandsons",
    "husband", "husbands", "boyfriend", "boyfriends", "king", "kings",
    "prince", "princes", "duke", "dukes", "lord", "lords", "sir", "sirs",
    "actor", "actors", "waiter", "waiters", "salesman", "salesmen",
    "policeman", "policemen", "fireman", "firemen"
]

masculine_pronouns = [
    "he", "him", "his"
]

masculine_adjectives = [
    "masculine", "manly", "virile", "macho", "paternal"
]

feminine_keywords = [
    "woman", "women", "girl", "girls", "female", "females", "lady", "ladies",
    "gal", "gals", "lass", "lasses", "daughter", "daughters", "mother", "mothers",
    "aunt", "aunts", "niece", "nieces", "sister", "sisters", "grandmother", "grandmothers",
    "granddaughter", "granddaughters", "wife", "wives", "girlfriend", "girlfriends",
    "queen", "queens", "princess", "princesses", "duchess", "duchesses",
    "lady", "ladies", "madam", "madams", "actress", "actresses", "waitress", "waitresses",
    "saleswoman", "saleswomen", "policewoman", "policewomen", "firewoman", "firewomen"
]

feminine_pronouns = [
    "she", "her", "hers"
]

feminine_adjectives = [
    "feminine", "womanly", "girly", "ladylike", "maternal"
]


# Define your list of gender-related words
masculine_words = {
    "keywords": masculine_keywords,
    "pronouns": masculine_pronouns,
    "adjectives": masculine_adjectives
}

feminine_words = {
    "keywords": feminine_keywords,
    "pronouns": feminine_pronouns,
    "adjectives": feminine_adjectives
}

# Define a function to identify gender-related words in the text
def internal_identify_gender_words(text, gender_words):
    doc = nlp(text)
    found_gender_words = []
    
    for token in doc:
        if token.text.lower() in gender_words["keywords"] or \
           token.text.lower() in gender_words["pronouns"] or \
           token.text.lower() in gender_words["adjectives"]:
            found_gender_words.append(token.text)
    
    return found_gender_words != []

def identify_gender_words(text):
    is_masculine = internal_identify_gender_words(text, masculine_words)
    is_feminine = internal_identify_gender_words(text, feminine_words)

    if is_masculine and not is_feminine:
        return 1
    elif is_feminine and not is_masculine:
        return 0
    else:
        return 2

# caption = "A man and a woman."
# gender_info = identify_gender_words(caption)
# print(gender_info)
