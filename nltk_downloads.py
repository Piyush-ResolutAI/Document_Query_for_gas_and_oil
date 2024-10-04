import nltk
try:
    print("Installing")
    nltk.download('punkt_tab')
    nltk.download('averaged_perceptron_tagger_eng')
except:
    pass
