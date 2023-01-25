import random
from transformers import BertTokenizerFast


def get_sentence_pair(index, text_corpus):
    sentence_a = text_corpus[index]
    prob = random.random()
    if prob > 0.5:
        is_next = True
        sentence_b = text_corpus[index + 1]
    else:
        is_next = False
        sentence_b = text_corpus[random.randrange(len(text_corpus))]
        while sentence_b == sentence_a or sentence_b == text_corpus[index + 1]:
            sentence_b = text_corpus[random.randrange(len(text_corpus))]

    return sentence_a, sentence_b, is_next


def add_special_tokens(first_sent_tokens, second_sent_tokens, tokenizer: BertTokenizerFast):
    first_sentence = [tokenizer.cls_token] + first_sent_tokens + [tokenizer.sep_token]
    second_sentence = second_sent_tokens + [tokenizer.sep_token]
    return first_sentence, second_sentence



if __name__ == "__main__":

    text_corp = ["This is the first sentence.", "And this is the second sentence.", "After the second comes the third sentence.",
                 "And after that the fourth ome.", "Here we have the fifth sentence.", "And finally the sixth one."]
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")

    sent_a, sent_b, is_next = get_sentence_pair(2, text_corp)
    print(sent_a)
    print(sent_b)
    print(is_next)
    sent_a = tokenizer.tokenize(sent_a)
    sent_b = tokenizer.tokenize(sent_b)
    special_a, special_b = add_special_tokens(sent_a, sent_b, tokenizer)
    print(special_a)
    print(special_b)
