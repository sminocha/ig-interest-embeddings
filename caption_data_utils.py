from googletrans import Translator
import emoji

# constants
language = 'en'


googleTranslator = Translator()

def remove_emojis(caption):
    """Remove emojis from the caption"""
    return emoji.get_emoji_regexp().sub(u'', caption)

def remove_tags(caption):
    """Remove tags '#___' from caption"""
    words = caption.split(' ')
    filtered_words = list(filter(lambda w: len(w) > 0 and w[0] != '#', words))
    return ' '.join(filtered_words)


def remove_mentions(caption):
    """Remove mentions '@___' from caption"""
    words = caption.split(' ')
    filtered_words = list(filter(lambda w: len(w) > 0 and w[0] != '@', words))
    return ' '.join(filtered_words)

'''
Source language isn't given in our data, and many captions are from non
english-speaking influencers.

Translate a user's 17 captions from source language to english (uses single
method call and single HTTP session). Return as list of translations.
'''
def translateToEnglish(captionsList, language):
    en_translations = googleTranslator.translate(captionsList, dest=language)
    en_translations = [translation.text for translation in translations]
    return en_translations

def getDetection(caption):
    """output googletrans detection object or objects (in case input is a list)"""
    return googleTranslator.detect(caption)

def checkNoForeignCaptions(captions, language):
    """return true if all captions are english, false if there is a foreign caption"""
    detectionObjects = getDetection(captions)
    whichAreEnglish = [obj.lang==language for obj in detectionObjects]
    return all(whichAreEnglish)


"""
Will want to remove emojis, tags, and mentions before translating
Test caption: "معاناة عشناها في مباراة امس .. ولكن الملكي دائما يتفائل بالفوز في ظل حضور المدريدية المتعصبة ماما زليخة ❤️✌🏻️😂 #HalaMadrid .من عدسة المبدع ‏ @m7md_khr"
"""
