import numpy as np
import nltk
from collections import Counter
import math
import json

"""
We need corpus term frequencies: for each word, count how often it occurs
We need the corpus document frequencies: for each word, count of documents it occurs in
Then calculate normalized document frequencies:
total_df = sum of all document frequencies values
for every word in the corpus: skip words that occur less than 10 times
normalized_df = document frequency / total_df
Also the negative sample table which I have not fully understood
"""


class CorpusSampler:

    def __init__(self, normalized_df):
        table_len = 1e8  # actually 1e8
        table = np.zeros(int(table_len), dtype=np.uint32)

        p, i = 0, 0
        for index, word in enumerate(normalized_df):
            p += float(normalized_df[word])
            while i < table_len and float(i) / table_len < p:
                table[i] = index
                i += 1
        self.table = table
        self.vocab = list(normalized_df.keys())

    def sample(self, sample_size):
        indices = np.random.randint(low=0, high=len(self.table), size=sample_size)
        return [self.vocab[self.table[i]] for i in indices]


def sample_length():
    length = 0
    while length <= 0:
        length = np.random.poisson(3)
    return length


def get_normalized_dfs():
    path = "/home/katja/singularity/python-images/transformer_pretraining/python/output/dataset_stats/prop/normalized_dfs.json"
    with open(path) as file:
        normalized_dfs = json.load(file)
    return normalized_dfs


def get_general_corpus_info():
    path = "/home/katja/singularity/python-images/transformer_pretraining/python/output/dataset_stats/prop/general_info.json"
    with open(path) as file:
        info = json.load(file)
    total_corpus_words = info.get("total_number_of_words")
    avg_doc_word_num = info.get("avg_doc_len")
    return total_corpus_words, avg_doc_word_num


def get_corpus_tfs():
    path = "/home/katja/singularity/python-images/transformer_pretraining/python/output/dataset_stats/prop/corpus_tfs.json"
    with open(path) as file:
        corpus_tfs = json.load(file)
    return corpus_tfs


def get_word_kept_probability(word, corpus_tf, total_words):
    # randomly discard frequent words and discard rare words
    t = 1e-4
    word_count = corpus_tf[word] if word in corpus_tf else 0
    if word_count < 100:
        return None
    word_frequency = word_count / total_words
    prob = math.sqrt(t / word_frequency) + (t / word_frequency)
    return prob


def get_document_info(document):
    """
    Term frequencies, number of words and vocab of a document
    Normalized df: dict {word: normalized_df}
    """
    document_words = [word for word in nltk.word_tokenize(document) if word.isalpha()]
    document_len = len(document_words)
    counts = Counter(document_words)
    # tfs = {key: (value / document_len) for key, value in counts.items()}
    tfs = counts
    vocab = list(set(document_words))
    return tfs, document_len, vocab


def generate_word_sets(document, total_corpus_words, avg_doc_word_num, corpus_tf, normalized_df,
                       negative_sampler: CorpusSampler):
    # total corpus words: not unique, but the amount of words in the corpus
    document = document.lower()
    doc_tfs, doc_len, doc_vocab = get_document_info(document)

    rop_length = sample_length()

    doc_vocab_score = {}
    for word in doc_vocab:
        tf = doc_tfs[word] if word in doc_tfs else 0
        df = normalized_df[word] if word in normalized_df else 0
        doc_vocab_score[word] = (tf + avg_doc_word_num * df) / (doc_len + avg_doc_word_num)

    corpus_sample_score = {}
    sample_num = 500 - len(doc_vocab) if 500 > len(doc_vocab) else len(doc_vocab) + 100
    corpus_sample = negative_sampler.sample(sample_num)
    for word in corpus_sample:
        if word in doc_vocab:
            continue
        df = normalized_df[word] if word in normalized_df else 0
        corpus_sample_score[word] = (0 + avg_doc_word_num * df) / (doc_len + avg_doc_word_num)

    total_scores = {}
    total_scores.update(doc_vocab_score)
    total_scores.update(corpus_sample_score)
    total_scores = {k: v for k, v in sorted(total_scores.items(), key=lambda item: item[1], reverse=True)}
    prob = [v for k, v in total_scores.items()]

    total_prob = sum(prob)
    normalized_prob = [p / total_prob for p in prob]

    normalized_sample_prob = {k: normalized_prob[i] for i, (k, v) in enumerate(total_scores.items())}

    word_sets = []
    scored_word_sets = []
    for k in range(2):
        representative_words = []
        while len(representative_words) < rop_length:
            word = np.random.choice(list(normalized_sample_prob), size=1, p=normalized_prob)[0]
            word_kept_prob = get_word_kept_probability(word, corpus_tf, total_corpus_words)
            if word_kept_prob is None or word_kept_prob < np.random.rand() or word in representative_words:
                continue
            else:
                representative_words.append(word)
        word_set_score = sum([math.log(total_scores[word]) for word in representative_words])
        scored_word_sets.append((' '.join(representative_words), word_set_score))
    word_sets.append(scored_word_sets)
    return word_sets




def get_doc_word_pairs(index, text_corpus, epoch_num):
    item = text_corpus[index]
    document = item["text"]
    word_set_dict = item["word_sets"]
    epoch_num = epoch_num % 10
    word_set_dict = json.loads(word_set_dict)

    word_set_list = word_set_dict.get(str(epoch_num))
    word_sets = (word_set_list[0], word_set_list[1])
    label = word_set_list[2]

    return document, word_sets, label


if __name__ == "__main__":

    doc_1 = """The best way to do this is just play with the numbers – take this, replace the “n” with the number of draws, “m” with the number of reminaing “economy cards” (obviously not including another Contacts – things that help with one or two clicks), and “N” with the remaining deck size, and develop your intuition about the odds of a total whiff (x=0 on the PDF table of values). But I’d be remiss to talk about this without bringing up the grand poobah – Liberated Account. What Stimhack is to Scenario Two, Liberated Account is to Scenario Three. Liberated Account gives you +10 AND one click to burn, even IF you want to run hygienically. So worst come to worst, you can click for credit and get +11/+12. But where it really shines is with one money Event. The only options we had so far where you clicked for cred with your spare time – Casts and Kati – were both dismal. Being able to play money events is what helps prevent these from creating big pockets of weakness. Liberated Account, though, is not only strong, BUT, because it finishes out, it has that one “free click” for a Money event. Or hell, maybe a Virus counter on Crypsis or a Tinkering…you get the idea. Liberated Account has the high profit of the other options WITH the flexibility of a spare click. It’s super crazy on this timescale. So here’s a lesson. Kati, Daily Casts, Contacts – none of these are BAD cards. But if you need the money FOR something, or are in a dance with the corp desperate not to leave a gap, Armitage and ESPECIALLY Liberated Account do a lot better than they get credit for. Remember, though, that all of this analysis assumes the Corp can do something about it. If the Corp is broke with an unrezzed remote, you’re in Scenario 1, and don’t need to worry about stinking vulnerability. Corps, this is why you must be careful what you’re rezzing and when. """
    doc_2 = """UPDATE: Drone video over spillway added. Collapse of emergency spillway expected, evacuation ordered Department of Water Resources officials say they expect the emergency spillway at Oroville Dam to fail, and say residents should evacuate northward. The emergency spillway suffered erosion and could fail, according to DWR. If that happens, the water behind that barrier will comedown the hill and down the river. Flow through the broken main spillway was increased to 100,000 cubic feet per second in an effort to lower the water level in the lake more rapidly. The Butte County Sheriff’s Office reports helicopters will be depositing rock-filled containers to strengthen the potential failure point. Bud Englund, a public information officer for the incident, said downtown Oroville and low-lying areas, including residents along the Feather River from Oroville to Gridley, are being evacuated. Reporter Andre Byik said Caltrans and the California Highway Patrol have converted the southbound lanes of Highway 70 into northbound lanes to expedite the evacuation. Traffic there is still nearly gridlocked. An evacuation center has been set up at the Silver Dollar Fairgrounds in Chico. Black Butte Lake west of Orland has also opened up the Buckhorn Campground to evacuees. Emergency operations centers as far south of Sacramento have been notified, Englund said. Evacuation orders have also been made in Yuba and Sutter counties. From ChicoER.com My local newspaper publishes a scathing editorial of DWR idiocy and mismanagement Live video here: https://www.facebook.com/KCRA3/videos/10155026580966514/ UPDATE: DWR issued this statement.n their track record so far…not sure its all that reassuring. OROVILLE DAM, Calif. – The Department of Water Resources has provided an explanation as to why the mandatory immediate evacuations in Oroville and areas downstream are occurring. The concern is that erosion at the head of the emergency/auxiliary spillway issued evacuation orders for residents. The concern is that erosion at the head of the emergency spillway threatens to undermine the concrete weir and allow large, uncontrolled releases of water from Lake Oroville. Those potential flows could exceed the capacity of downstream channels. """

    total_corpus_words, avg_doc_word_num = get_general_corpus_info()
    corpus_tf = get_corpus_tfs()
    normalized_dfs = get_normalized_dfs()
    negative_sampler = CorpusSampler(normalized_dfs)

    print("Generating word sets")

    for i in range(10):
        word_sets = generate_word_sets(doc_2, total_corpus_words, avg_doc_word_num, corpus_tf, normalized_dfs,
                                       negative_sampler)
        print(word_sets)
