from googletrans import Translator
import emoji

# constants
language = 'en'


googleTranslator = Translator()

def remove_emojis(caption):
    """Remove emojis from the caption"""
    return emoji.get_emoji_regexp().sub(u'', str(caption))

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

def remove_unnecessary(caption):
    """Remove emojis, tags, and mentions from caption"""
    noEmojis = remove_emojis(caption)
    alsoNoTags = remove_tags(noEmojis)
    return remove_mentions(alsoNoTags)

'''
Source language isn't given in our data, and many captions are from non
english-speaking influencers.

Translate a user's 17 captions from source language to english (uses single
method call and single HTTP session). Return as list of translations.
'''
def translateListToEnglish(captionsList, language):
    en_translations = googleTranslator.translate(captionsList, dest=language)
    en_translations = [translation.text for translation in translations]
    return en_translations

def translateOne(caption):
    """Translate caption into english"""
    en_translation = googleTranslator.translate(caption, 'en')
    return en_translation.text

def removeThenTranslate(caption):
    """translate caption to english then remove emojis, tags, and mentions"""
    trimmed = remove_unnecessary(caption)
    return translateOne(trimmed)

def getDetection(caption):
    """output googletrans detection object or objects (in case input is a list)"""
    return googleTranslator.detect(caption)

def isEnglish(caption):
    """return true if caption in english, false otherwise"""
    detectionObject = getDetection(caption)
    return detectionObject.lang == 'en'

def checkNoForeignCaptions(captions, language):
    """return true if all captions are english, false if there is a foreign caption"""
    detectionObjects = getDetection(captions)
    whichAreEnglish = [obj.lang==language for obj in detectionObjects]
    return all(whichAreEnglish)


"""
Will want to remove emojis, tags, and mentions before translating
Test caption: "معاناة عشناها في مباراة امس .. ولكن الملكي دائما يتفائل بالفوز في ظل حضور المدريدية المتعصبة ماما زليخة ❤️✌🏻️😂 #HalaMadrid .من عدسة المبدع ‏ @m7md_khr"
"""
