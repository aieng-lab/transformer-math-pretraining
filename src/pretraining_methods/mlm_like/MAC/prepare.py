"""
MAC - MLM as Correction

- Replacing 15% of input (tokens or words ???) --> words
- 40% unigrams
- 30% 2-grams
- 20% 3-grams
- 10% 4-grams

- instead of MASK token: replace with similar words (80%), random words (10%) original words (10%)
Predict original tokens



Implementation:
choose replacement strategy on the word level
Ignore punctuation when doing the replacements (as in the paper they used whole word tokenization)
During replacement: prohibit words that produce more tokens than the original words
When the replaced word has fewer tokens than the original, fill up with [PAD] (but maybe an entirely new token should be
introduced? Like [None]) --> [unused1]
But adding a new token will change the embedding size of BERT and the same token needs to be added to the tokenizer in
the fine-tuning stage
There was a paper somewhere that introduced a NONE token for detecting deletions, maybe do this
"""

from src.helpers.general_helpers import path_exists, create_path_if_not_exists
from src.helpers.Timer import Timer

from transformers import BertTokenizerFast
import random
import re
import os
import json
from nltk.corpus import words
from random import sample
import nltk


# todo remove this, put it somewhere where it does not have to be loaded for every sample


def get_sim_word_dict():
    nltk.download("words")
    path = os.path.expanduser(f'~/gensim-data/precomputed')
    file_name = "similar_words_small_model.json"
    with open(os.path.join(path, file_name)) as file:
        sim_word_dict = json.load(file)
    return sim_word_dict


def replace_word(tokens, word_positions, strategy, tokenizer, sim_word_dict, word_list):
    special_chars = r"[^\w\s\d]"
    word_token_len = len(word_positions)
    word_tokens = [tokens[i] for i in word_positions]
    word = tokenizer.convert_tokens_to_string(word_tokens)

    synonym = None

    if strategy == "synonym":

        try:
            similar_words = sim_word_dict.get(word)
            if similar_words is None:
                raise KeyError("No similar word could be found")
            candidates = []
            for sim_word in similar_words:
                tokenized = tokenizer.tokenize(sim_word)
                if len(tokenized) <= word_token_len and tokenized != tokenizer.unk_token and not re.match(special_chars,
                                                                                                          sim_word):
                    candidates.append(tokenized)
            if not candidates:
                raise KeyError("No similar word could be found")
            candidates = candidates[:5]
            synonym = random.choice(candidates)
            if len(synonym) < word_token_len:
                synonym = synonym + (["[unused1]"] * (word_token_len - len(synonym)))
        except KeyError:
            word = word.lower()
            try:
                similar_words = sim_word_dict.get(word)
                if similar_words is None:
                    raise KeyError("No similar word could be found")
                candidates = []
                for sim_word in similar_words:
                    tokenized = tokenizer.tokenize(sim_word)
                    if len(tokenized) <= word_token_len and tokenized != tokenizer.unk_token and not re.match(
                            special_chars,
                            sim_word):
                        candidates.append(tokenized)
                if not candidates:
                    raise KeyError("No similar word could be found")
                candidates = candidates[:5]
                synonym = random.choice(candidates)
                if len(synonym) < word_token_len:
                    synonym = synonym + (["[unused1]"] * (word_token_len - len(synonym)))
            except KeyError:
                synonym = None

        if synonym is not None:
            # synonym has been found
            labels = word_tokens
            label_positions = word_positions

            for i, index in enumerate(word_positions):
                tokens[index] = synonym[i]

            return tokens, labels, label_positions

    if strategy == "random" or (strategy == "synonym" and synonym is None):
        random_word = random.choice(word_list)
        random_word = " ".join(random_word.split("_"))
        tokenized = tokenizer.tokenize(random_word)
        i = 0
        while len(tokenized) > word_token_len:
            random_word = random.choice(word_list)
            random_word = " ".join(random_word.split("_"))
            if re.match(special_chars, random_word):
                i += 1
                continue
            tokenized = tokenizer.tokenize(random_word)
            i += 1
            if i > 20:
                random_word = None
                break
        if random_word is None:
            # use a random word from the input text instead (there will be short words like "in", "the", etc.)
            word_list = list(sim_word_dict.keys())
            random.shuffle(word_list)
            for random_word in word_list:
                random_word = " ".join(random_word.split("_"))
                if re.match(special_chars, random_word) or random_word == word:
                    continue
                tokenized = tokenizer.tokenize(random_word)
                if len(tokenized) <= word_token_len:
                    break

            if len(tokenized) > word_token_len:
                tokenized = word_tokens  # fall back to original word

        if len(tokenized) < word_token_len:
            tokenized = tokenized + (["[unused1]"] * (word_token_len - len(tokenized)))

        labels = word_tokens
        label_positions = word_positions

        for i, index in enumerate(word_positions):
            tokens[index] = tokenized[i]

        return tokens, labels, label_positions

    if strategy == "no":
        labels = word_tokens
        label_positions = word_positions
        return tokens, labels, label_positions


def corrupt_tokens(tokens, tokenizer: BertTokenizerFast, vocabulary: list, seq_len, sim_words_dict, rand_word_list):
    # seq len is the number of words in this case
    special_tokens = tokenizer.all_special_tokens
    replacement_dict = {}
    label_dict = {}
    word_len = 0
    for token in tokens:  # todo: maybe skip special tokens as well?
        if not token.startswith("##"):
            word_len += 1

    special_chars = r"[^\w\s\d]"
    masked_words = 0
    masked_positions = []
    spans = []
    labels = []
    label_positions = []

    span_finding_iterations = 0  # todo: for debugging
    while (masked_words + 1) / word_len <= 0.15 and span_finding_iterations < 600:

        span_finding_iterations += 1

        current_masked = masked_words
        current_positions = []

        index = random.randrange(len(tokens))
        token = tokens[index]

        # Masking should always start at beginning of word
        while (token.startswith("##") or re.match(special_chars,
                                                  token)) and index >= 1 and token not in special_tokens and index not in masked_positions:

            index = index - 1
            token = tokens[index]
            if re.match(special_chars, token) and not token.startswith("##"):
                token = f"##{token}"

        # start token is found
        if (index in masked_positions or tokens[index] in special_tokens) or (
                index > 0 and (index - 1 in masked_positions or tokens[index - 1] in special_tokens)):
            # keep at least one token between spans (and special tokens)
            continue

        prob = random.random()
        # decide how many words should be masked

        if prob > 0.9:
            num_words = 4
        elif prob > 0.7:
            num_words = 3
        elif prob > 0.4:
            num_words = 2
        else:
            num_words = 1

        full_words = 0
        overlap = False

        while full_words < num_words and (current_masked / word_len) < 0.15 and index < len(tokens) - 1:
            index += 1
            if index in masked_positions or (index < len(tokens) - 1 and index in masked_positions):
                # keep at least one token between spans
                overlap = True
                break
            token = tokens[index]
            if token in special_tokens:
                overlap = True
                break
            prev_token = tokens[index - 1]
            if not prev_token.startswith("##"):
                # about to start new word
                if (current_masked + 1) / word_len > 0.15:
                    break
            current_positions.append(index - 1)
            if not token.startswith("##"):
                full_words += 1
                current_masked += 1
                if full_words == num_words:
                    break

        if overlap:
            continue

        # possibly last token might belong to a word that was already begun but not finished
        if index == len(tokens) - 1:
            token = tokens[index]
            if token.startswith("##"):
                current_positions.append(index)

        # decide replacement strategy (per word or per span???) --> do it like SpanBERT and choose per span (n-gram)
        replace = None
        for id in current_positions:
            if not tokens[id].startswith("##"):
                prob = random.random()
                if prob <= 0.8:
                    replace = "synonym"
                elif prob <= 0.9:
                    replace = "random"
                else:
                    replace = "no"
            replacement_dict[id] = replace

        masked_positions.extend(current_positions)
        spans.append(current_positions)
        masked_words = current_masked

    # replacement positions are determined

    for span in spans:
        current_word_positions = []
        new_label_positions = []

        for index in span:
            replacement = replacement_dict.get(index)
            token = tokens[index]
            if current_word_positions and not token.startswith("##"):
                # beginning of new word
                tokens, new_labels, new_label_positions = replace_word(tokens, current_word_positions, replacement,
                                                                       tokenizer, sim_words_dict, rand_word_list)
                for i, position in enumerate(new_label_positions):
                    label_dict[position] = new_labels[i]
                labels.extend(new_labels)
                label_positions.extend(new_label_positions)
                current_word_positions = []
            if re.match(special_chars, token):
                # make sure this is actually a special character, not a suffix token
                stripped_token = token.lstrip("##")
                if re.match(special_chars, stripped_token):
                    if replacement in ("synonym", "random"):
                        # ignore punctuation in replacement of synonyms and random words
                        label_dict[index] = token
                        current_word_positions = []
                        continue
            current_word_positions.append(index)

        if current_word_positions:
            tokens, new_labels, new_label_positions = replace_word(tokens, current_word_positions, replacement,
                                                                   tokenizer, sim_words_dict, rand_word_list)

        for i, position in enumerate(new_label_positions):
            label_dict[position] = new_labels[i]

    ordered_labels = []
    ordered_label_positions = []

    for index, token in enumerate(tokens):
        if index in label_dict:
            ordered_label_positions.append(index)
            ordered_labels.append(label_dict.get(index))

    return tokens, ordered_labels, ordered_label_positions


if __name__ == "__main__":

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")

    vocab = list(tokenizer.get_vocab().keys())

    # text = """Cole Harbour District High School was founded in 1982 by the amalgamation of Cole Harbour High School, founded in 1979, and Gordon Bell High School. The newer building for Cole Harbour High was used as the new location for Cole Harbour District High, and Gordon Bell High was repurposed as the Gordon Bell Building, a general-purpose adult education centre. The Gordon Bell Building served as the Grade 10 building until 1995 when the Cole Harbour District high school was split into two separate schools. Cole Harbour High and Auburn High. The Bell building was also used as a temporary location for the students of Halifax West High School during the 2001-2002 school year, when its own building was unusable due to health concerns. As of January 2015 the Gorden Bell building has been demolished. Cole Harbour District High is seen in the hit series Trailer Park Boys. It is used as the location to film the scenes that take place in the fictional Dartmouth Regional Vocational School (DRVS). CHDHS's unofficial rival school is Auburn Drive High School, with whom they play a football game against every Thanksgiving weekend called the Turkey Bowl. In the 2014-2015 school year, renovations were completed on the school adding a new gym and a skilled trades centre while converting the old gymnasium into a cafetorium. The skilled trades centre also includes a yoga studio and a student council office. """
    text = """Newman says that the song was inspired by his own lighthearted reflection on the Los Angeles music scene of the late 1960s. As with most Newman songs, he assumes a character; in this song the narrator is a sheltered and extraordinarily straitlaced young man, who recounts what is presumably his first "wild" party in the big city, is shocked and appalled by marijuana smoking, whiskey drinking, and loud music, and – in the chorus of the song – recalls that his "Mama told [him] not to come". The first recording of "Mama Told Me Not to Come" was cut by Eric Burdon & The Animals. A scheduled single-release of September 1966 was withdrawn,[1] but the song was eventually included on their 1967 album Eric Is Here. Newman's own turn at his song was released on the 1970 album 12 Songs, and was characterized by Newman's mid-tempo piano accompaniment, as well as Ry Cooder's slide guitar part, both of which give the song the feel of a bluesy Ray Charles-style rhythm and blues number. Also in 1970, Three Dog Night released a longer, rock 'n roll and funk-inspired version (titled "Mama Told Me (Not to Come)") on It Ain't Easy, featuring Cory Wells singing lead in an almost humorous vocal style,[4] Jimmy Greenspoon playing a Wurlitzer electronic piano, Michael Allsup playing guitar, and Donna Summer on backing vocals, though uncredited.[citation needed]"""
    # text = """Brianne Alexandra Jenner (born May 4, 1991) is a Canadian professional ice hockey player and a member of Canada's national women's hockey team, currently affiliated with the Toronto chapter of the Professional Women's Hockey Players Association (PWHPA). She made her debut for Canada at the 2010 Four Nations Cup and won a gold medal. She was also a member of the Cornell Big Red women's ice hockey program. In high school, Jenner was the Appleby College hockey team captain. Jenner played junior hockey in the Provincial Women's Hockey League (PWHL) with the Stoney Creek Sabres. She was also the captain of Team Ontario Red at the 2008 National Women's Under-18 Championship. She scored the game-winning goal in double overtime of the gold medal game. On October 29 and 30, 2010, Jenner played a role in both victories for the Cornell Big Red ice hockey team. On October 29, she had three assists at Quinnipiac. The following day, she scored a pair of goals and added an assist at Princeton. During three games from February 7 to February 11, 2012, Jenner led her team with eight points. Versus nationally ranked Mercyhurst, Jenner had a goal and an assist in a February 7 victory over Mercyhurst. In a 5–0 shutout win over the Brown Bears (on February 10), Jenner garnered two assists from two goals. On February 11, Jenner scored the game-winning goal versus the Yale Bulldogs that clinched the ECAC Hockey regular-season championship. In addition, she scored another goal, earning her 30th assist of the season."""
    # text = """Steven Harold Conroy (19 December 1956 – 4 May 2021) was an English footballer who played as a goalkeeper who spent the bulk of his career playing for Sheffield United. Born in Chesterfield, North East Derbyshire, England, Conroy had appeared as a schoolboy for his local town before signing as an apprentice for Sheffield United in 1972. Graduating to the first team squad in 1974, Conroy turned professional, but was usually employed as cover for the first choice keeper, and so did not make his league debut until 23 August 1977 in a 2–0 home victory over Hull City.[1] Establishing himself in the first team in 1978, Conroy became first choice keeper until breaking his arm in an Anglo-Scottish Cup game against St Mirren in December 1979.[1] His injury sidelined Conroy for over a year and he did not return to playing until the 1980–81 season and was part of the team that was relegated to Division Four for the first time in the club's history. Conroy continued to be dogged by injuries and was eventually released by United in January 1983, signing for near neighbours Rotherham United on non-contract terms the following February.[1] Released at the end of the season, Conroy then signed for Rochdale where he spent a further 18 months battling with injuries before returning to Rotherham for a final twelve-month spell in 1985.[1] Following his retirement from playing, Conroy spent a time as part of the Sheffield United coaching staff. Steve Conroy died at the age of 64 on 4 May 2021"""

    tokens = tokenizer.tokenize(text)

    original_tokens = tokens

    sim_words_dict = get_sim_word_dict()

    timer = Timer()
    timer.start()

    tokens, labels, label_positions = corrupt_tokens(tokens, tokenizer, vocab, 512, sim_words_dict)

    timer.stop()
    timer.print_elapsed()

    print("done")
    print(f"{len(labels)} / {len(tokens)}")

    print("\n\n")
    labels = [-100 if index not in label_positions else labels.pop(0) for index, token in enumerate(tokens)]
    print(f"{'Original':<25}{'Corrputed':<25}{'Labels':25}")
    for i in range(len(tokens)):
        print(f"{original_tokens[i]:<25}{tokens[i]:<25}{labels[i]}")
    print("\n\n")
