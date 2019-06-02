from googletrans import Translator
import caption_data_utils as caption

# constants
test_en = ['Hi gorgeous!', 'what is your name', 'where are you from?']
test_span = ['hola me llamo charles', 'como te llamas', 'de donde eres']


# get user nodes, using name
# pandas data frame
# todo for processing captions and images.
# pd.read_csv(pass in path ), then grab column that has all captions
# url image or url image profile, can go run image_data_utils on that link to download that photo.
# process all things for singl euser, then add to new row in new csv. new


'''
Source language isn't given in our data, and many captions are from non
english-speaking influencers.

Translate a user's 17 captions from source language to english (uses single
method call and single HTTP session). Return as list of translations.
'''
def translate_caption(prevCaptions):
     googleTranslator = Translator()
     translations = googleTranslator.translate(prevCaptions, dest='en')
     translations = [translation.text for translation in translations]
     return translations



engCaptions = translate_caption(test_span)


detections = caption.getDetection(test_span)
captionLanguages = [caption.lang=='en' for caption in detections]
print(captionLanguages)
print(engCaptions)
