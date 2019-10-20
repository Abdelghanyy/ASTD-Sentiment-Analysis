from gensim.models.fasttext import FastText
from gensim.test.utils import common_texts
import re
import string
from nltk.util import ngrams
emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)
punc_pattern = re.compile("[–" + string.punctuation + "]+", flags=re.UNICODE)


def reading_bankingData(filename):
    dataset=[]
    labels=[]
    file = open(filename, 'r', encoding='utf-8')
    for line in file:
        dataset.append(line)
    return dataset,labels
def reading_asdt(filename):
    # A list that contains ASTD lables
    labels = []
    # A list that contains ASTD Tweets
    dataset = []
    # A list that contains ASTD labels
    new_labels = []

    file = open(filename, 'r', encoding='utf-8')
    i = 0
    for line in file:
        labels.append(line[-4:])
        labels[i] = labels[i][0:3]
        if (labels[i] == "RAL"):
            dataset.append(line[0:len(line) - 8])
        else:
            dataset.append(line[0:len(line) - 5])

        i = i + 1
    return dataset, labels


emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)
punc_pattern = re.compile("[–" + string.punctuation + "]+", flags=re.UNICODE)


def clean_sentence(line, lang='en', verbose=0):
    line = line.replace('\n', ' ').replace('\r', '')
    line = line.replace('\\n', ' ').replace('\\r', '')
    line = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', line)
    line = re.sub('(?<=^|(?<=[^a-zA-Z0-9-_\.]))@([A-Za-z]+[A-Za-z0-9]+)', ' ', line)
    line = re.sub('[أآاإ]', 'ا', line)
    line = line.replace("ة", "ه ")
    line = re.sub('[ئ]', 'ىء', line)
    line = re.sub('[ؤ]', 'وء', line)
    line = re.sub('[ًٌٍَُِّْ]', ' ', line)
    if line != "" and line[-1] == 'ي':
        line = line[:-1] + 'ى'
    line = line.replace("_", " ")
    line = line.replace("#", " ").replace("@", "")
    line = line.replace('⇘', ' ')
    line = line.replace('╔ ', ' ')
    line = line.replace(':', ' ')
    line = line.replace('ً', ' ')
    line = line.replace('<', ' ')
    line = line.replace('>', ' ')
    line = re.sub('\s+', ' ', line)  # remove spaces
    line = re.sub('\.+', ' ', line)  # multiple dots
    line = re.sub('\.', ' ', line)  # multiple dots
    line = re.sub(r'[،:?؟@#$%^&*_~\'-;",()<>!.؛]\s*', ' ', line)
    line = re.sub(' +', ' ', line)
    line = re.sub(r'(\w)\1+', r'\1', line)  # yes
    line = emoji_pattern.sub(r'', line)  # no emoji
    line = re.sub('"', ' ', line)
    line = re.sub('\'', ' ', line)
    line = re.sub('’', ' ', line)
    line = re.sub('،', ' ', line)
    line = re.sub('“', ' ', line)
    line = re.sub('”', ' ', line)
    line = re.sub('”', ' ', line)
    line = punc_pattern.sub(r'', line)  # no punctuation
    if lang == 'ar':
        cleaned_line = re.sub(u'[\x00-\x7F\x80-\xFF\u0100-\u017F\u0180-\u024F\u1E00-\u1EFF]', u' ', line).strip()

    if verbose:
        print(line)

    if len(line) > 2:
        return line
    else:
        return None

def clean_arabic_sentence(line):
    cleaned_text = clean_sentence(line, 'ar')
    if cleaned_text is None:
        return None
    return cleaned_text

def normalizeArabic(text):
    text = re.sub('[أآاإ]', 'ا', text)
    text = text.replace("ة", "ه")
    text = re.sub('[ئ]', 'ىء', text)
    text = re.sub('[ؤ]', 'وء', text)
    text = re.sub('[ًٌٍَُِّْ]', ' ', text)
    if text and text[-1] == 'ي':
        text = text.replace(text, text[:-1] + 'ى', len(text) - 1)
        text = text[:-1] + 'ى'
        return text.strip()
    else:
        return text

def split_sentence_to_ngrams(sentence, ngram_range=(1, 1)):
    min_n, max_n = ngram_range
    tokens=[]
    if max_n != 1:
        tokens = []
        for n in range(min_n, max_n + 1):
            grams = ngrams(sentence.split(), n)
            for gram in grams:
                result = ""
                for g in gram:
                    result += g + " "
                tokens.append(result.strip())
    else:
        tokens=sentence.split(" ")
    return tokens
def prep (filepath):
    if filepath.startswith("Tweets"):
            dataset,labels=reading_asdt(filepath)
    else:
        dataset,labels=reading_bankingData(filepath)


    new_dataset=[]
    for line in dataset:
        cleaned_sentence=clean_arabic_sentence(line)
        normalized_sentence=normalizeArabic(cleaned_sentence)
        words=split_sentence_to_ngrams(normalized_sentence,ngram_range=(1,2))
        new_dataset.append(words)
    return new_dataset,labels


# dataset,labels=reading_bankingData("all_data_batches.txt")
# cleanned=clean_arabic_sentence(dataset[0])
# normalized=normalizeArabic(cleanned)
# print(split_sentence(normalized))


dataset,labels=prep("all_data_batches.txt")

print(dataset[:5])
model = FastText(size=4, window=3, min_count=1)
model.build_vocab(sentences=dataset)
model.train(sentences=common_texts, total_examples=len(dataset), epochs=10)
token="حساب جاري"

if token in model.wv:
    most_similar = model.wv.most_similar( token, topn=10 )
    for term, score in most_similar:
        if term != token:
            print(term, score)
