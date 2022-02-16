from itertools import product
import math
import nltk
from Code.DataAugment.text_utils import tokenize, is_punctuation
from Code.common_utils import read_table_file
import argparse
import glob
import os
from tqdm import tqdm

SOS = "<s> "
EOS = "</s>"
UNK = "<UNK>"


def add_sentence_tokens(sentences, n):
    """Wrap each sentence in SOS and EOS tokens.
    For n >= 2, n-1 SOS tokens are added, otherwise only one is added.
    Args:
        sentences (list of str): the sentences to wrap.
        n (int): order of the n-gram model which will use these sentences.
    Returns:
        List of sentences with SOS and EOS tokens wrapped around them.
    """
    sos = SOS * (n - 1) if n > 1 else SOS
    return ['{}{} {}'.format(sos, s, EOS) for s in sentences]


def replace_singletons(tokens, threshold=1):
    """Replace tokens which appear under threshold in the corpus with <UNK>.
    
    Args:
        tokens (list of str): the tokens comprising the corpus.
        threshold (int): frequency limit
    Returns:
        The same list of tokens with each singleton replaced by <UNK>.
    
    """
    vocab = nltk.FreqDist(tokens)
    return [token if vocab[token] > threshold else UNK for token in tokens]


def preprocess(sentences, n, threshold=1):
    """Add SOS/EOS/UNK tokens to given sentences and tokenize.
    Args:
        sentences (list of str): the sentences to preprocess.
        n (int): order of the n-gram model which will use these sentences.
        threshold (int): frequency limit
    Returns:
        The preprocessed sentences, tokenized by words.
    """
    sentences = add_sentence_tokens(sentences, n)
    tokens = ' '.join(sentences).split(' ')
    tokens = replace_singletons(tokens, threshold)
    return tokens


def load_data(filename):
    with open(filename, encoding='utf-8') as f:
        sents = []
        raw_sents = []
        for line in f:
            tokens = tokenize(line)
            tokens = [x for x in tokens if not is_punctuation(x)]
            if len(tokens) == 0:
                continue
            sents.append(' '.join(tokens))
            raw_sents.append(line.strip())
    return sents, raw_sents


class LanguageModel(object):
    """An n-gram language model trained on a given corpus.
    
    For a given n and given training corpus, constructs an n-gram language
    model for the corpus by:
    1. preprocessing the corpus (adding SOS/EOS/UNK tokens)
    2. calculating (smoothed) probabilities for each n-gram
    Also contains methods for calculating the perplexity of the model
    against another corpus, and for generating sentences.
    Args:
        train_data (list of str): list of sentences comprising the training corpus.
        n (int): the order of language model to build (i.e. 1 for unigram, 2 for bigram, etc.).
        laplace (int): lambda multiplier to use for laplace smoothing (default 1 for add-1 smoothing).
    """

    def __init__(self, train_data, n, laplace=1):
        self.n = n
        self.laplace = laplace
        self.tokens = preprocess(train_data, n)
        self.vocab = nltk.FreqDist(self.tokens)
        self.model = self._create_model()
        self.masks = list(reversed(list(product((0, 1), repeat=n))))

    def _smooth(self):
        """Apply Laplace smoothing to n-gram frequency distribution.
        
        Here, n_grams refers to the n-grams of the tokens in the training corpus,
        while m_grams refers to the first (n-1) tokens of each n-gram.
        Returns:
            dict: Mapping of each n-gram (tuple of str) to its Laplace-smoothed 
            probability (float).
        """
        vocab_size = len(self.vocab)

        n_grams = nltk.ngrams(self.tokens, self.n)
        n_vocab = nltk.FreqDist(n_grams)

        m_grams = nltk.ngrams(self.tokens, self.n - 1)
        m_vocab = nltk.FreqDist(m_grams)

        def smoothed_count(n_gram, n_count):
            m_gram = n_gram[:-1]
            m_count = m_vocab[m_gram]
            return (n_count + self.laplace) / (m_count + self.laplace * vocab_size)

        return {n_gram: smoothed_count(n_gram, count) for n_gram, count in n_vocab.items()}

    def _create_model(self):
        """Create a probability distribution for the vocabulary of the training corpus.
        
        If building a unigram model, the probabilities are simple relative frequencies
        of each token with the entire corpus.
        Otherwise, the probabilities are Laplace-smoothed relative frequencies.
        Returns:
            A dict mapping each n-gram (tuple of str) to its probability (float).
        """
        if self.n == 1:
            num_tokens = len(self.tokens)
            return {(unigram,): count / num_tokens for unigram, count in self.vocab.items()}
        else:
            return self._smooth()

    def _convert_oov(self, ngram):
        """Convert, if necessary, a given n-gram to one which is known by the model.
        Starting with the unmodified ngram, check each possible permutation of the n-gram
        with each index of the n-gram containing either the original token or <UNK>. Stop
        when the model contains an entry for that permutation.
        This is achieved by creating a 'bitmask' for the n-gram tuple, and swapping out
        each flagged token for <UNK>. Thus, in the worst case, this function checks 2^n
        possible n-grams before returning.
        Returns:
            The n-gram with <UNK> tokens in certain positions such that the model
            contains an entry for it.
        """
        mask = lambda ngram, bitmask: tuple((token if flag == 1 else "<UNK>" for token, flag in zip(ngram, bitmask)))

        ngram = (ngram,) if type(ngram) is str else ngram
        for possible_known in [mask(ngram, bitmask) for bitmask in self.masks]:
            if possible_known in self.model:
                return possible_known

    def perplexity(self, test_data):
        """Calculate the perplexity of the model against a given test corpus.

        Args:
            test_data (list of str): sentences comprising the training corpus.
        Returns:
            The perplexity of the model as a float.

        """
        test_data = "".join(tokenize(test_data))
        test_tokens = preprocess(test_data, self.n, 0)
        test_ngrams = list(nltk.ngrams(test_tokens, self.n))
        N = len(test_tokens)

        known_ngrams = list(self._convert_oov(ngram) for ngram in test_ngrams)
        probabilities = [self.model[ngram] for ngram in known_ngrams]

        return math.exp((-1 / N) * sum(map(math.log, probabilities)))

    def _best_candidate(self, prev, i, without=[]):
        """Choose the most likely next token given the previous (n-1) tokens.
        If selecting the first word of the sentence (after the SOS tokens),
        the i'th best candidate will be selected, to create variety.
        If no candidates are found, the EOS token is returned with probability 1.
        Args:
            prev (tuple of str): the previous n-1 tokens of the sentence.
            i (int): which candidate to select if not the most probable one.
            without (list of str): tokens to exclude from the candidates list.
        Returns:
            A tuple with the next most probable token and its corresponding probability.
        """
        blacklist = ["<UNK>"] + without
        candidates = ((ngram[-1], prob) for ngram, prob in self.model.items() if ngram[:-1] == prev)
        candidates = filter(lambda candidate: candidate[0] not in blacklist, candidates)
        candidates = sorted(candidates, key=lambda candidate: candidate[1], reverse=True)
        if len(candidates) == 0:
            return ("</s>", 1)
        else:
            return candidates[0 if prev != () and prev[-1] != "<s>" else i]

    def generate_sentences(self, num, min_len=12, max_len=24):
        """Generate num random sentences using the language model.
        Sentences always begin with the SOS token and end with the EOS token.
        While unigram model sentences will only exclude the UNK token, n>1 models
        will also exclude all other words already in the sentence.
        Args:
            num (int): the number of sentences to generate.
            min_len (int): minimum allowed sentence length.
            max_len (int): maximum allowed sentence length.
        Yields:
            A tuple with the generated sentence and the combined probability
            (in log-space) of all of its n-grams.
        """
        for i in range(num):
            sent, prob = ["<s>"] * max(1, self.n - 1), 1
            while sent[-1] != "</s>":
                prev = () if self.n == 1 else tuple(sent[-(self.n - 1):])
                blacklist = sent + (["</s>"] if len(sent) < min_len else [])
                next_token, next_prob = self._best_candidate(prev, i, without=blacklist)
                sent.append(next_token)
                prob *= next_prob

                if len(sent) >= max_len:
                    sent.append("</s>")

            yield ' '.join(sent), -1 / math.log(prob)


if __name__ == '__main__':
    train_filename = "Data/sighan.labels.txt"
    test_filename = "csc_evaluation/data/basedata/simplified/test2015.txt"
    test_filename = "csc_evaluation/data/customized_data/odw.txt"
    # test_dir = r'D:\repos\Corpus\new2016zh\texts\train'
    # save_dir = r'D:\repos\Corpus\new2016zh\ppl\train'
    n = 3
    laplace = 0.01
    ppl_threshold = 19
    data = [x[0] for x in read_table_file(test_filename, [2])]

    train, _ = load_data(train_filename)
    print("Loading {}-gram model...".format(n))
    lm = LanguageModel(train, n, laplace=laplace)
    print("Vocabulary size: {}".format(len(lm.vocab)))

    ppl_count = 0
    for sent in data:
        temp = lm.perplexity(sent)
        ppl_count += temp
    print(f"ppl is : {ppl_count / len(data)}")

    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    # for test_file in glob.glob(os.path.join(test_dir, '*.txt')):
    #     if os.path.exists(os.path.join(save_dir, os.path.basename(test_file))):
    #         continue
    #     test, raw_sents = load_data(test_file)
    #     print('Handle {}, total sentences: {}'.format(test_file, len(test)))
    #     with open(os.path.join(save_dir, os.path.basename(test_file)), 'w', encoding='utf-8') as f:
    #         for sent, raw_sent in tqdm(zip(test, raw_sents)):
    #             perplexity = lm.perplexity(sent)
    #             if perplexity > ppl_threshold:
    #                 continue
    #             f.write('{:.3f}\t{}\n'.format(perplexity, raw_sent))
