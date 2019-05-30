from googletrans import Translator
import emoji

translator = Translator()

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

def translate(caption):
    """Translate caption into english if necessary"""
    en_translation = translator.translate(caption)
    return en_translation.text


"""
Will want to remove emojis, tags, and mentions before translating
Test caption: "معاناة عشناها في مباراة امس .. ولكن الملكي دائما يتفائل بالفوز في ظل حضور المدريدية المتعصبة ماما زليخة ❤️✌🏻️😂 #HalaMadrid .من عدسة المبدع ‏ @m7md_khr"
"""
