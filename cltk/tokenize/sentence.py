# -*- coding: utf-8 -*-
"""Tokenize sentences."""

__author__ = 'Kyle P. Johnson <kyle@kyle-p-johnson.com>'
__license__ = 'MIT License. See LICENSE.'


from cltk.utils.file_operations import open_pickle
from nltk.tokenize.punkt import PunktLanguageVars
from nltk.tokenize.punkt import PunktSentenceTokenizer
from nltk.tokenize.punkt import PunktTrainer
import os
import pickle

try:
    # For Python 3.0 and later
    from urllib.request import urlopen
except ImportError:
    # Fall back to Python 2's urllib2
    from urllib2 import urlopen



PUNCTUATION = {'greek':
                   {'external': ('.', ';'),
                    'internal': (',', '·'),
                    'file': 'greek.pickle', },
               'latin':
                   {'external': ('.', '?', ':'),
                    'internal': (',', ';'),
                    'file': 'latin.pickle', }}


class TokenizeSentence():  # pylint: disable=R0903
    """Tokenize sentences for the language given as argument, e.g.,
    ``TokenizeSentence('greek')``.
    """

    def __init__(self, language, train=False):
        """Lower incoming language name and assemble variables.
        :type language: str
        :param language : Language for sentence tokenization.
        """
        self.language = language.lower()
        if (train and language == 'latin'):
            self._train_latin()
        self.internal_punctuation, self.external_punctuation, self.tokenizer_path = \
            self._setup_language_variables(self.language)


    def _train_latin(self):
        """Uses the logic from github.com/cltk/latin_training_set_sentence_cltk/
        to train the new classifier and save it as a pickle in the appropriate place.
        """
        training_url = urlopen('https://raw.githubusercontent.com/cltk/latin_training_set_sentence_cltk/cd5a4fc033bd14d20ed4dc17774dec95f354eeb5/training_sentences.txt')
        train_data = training_url.read()
        language_punkt_vars = PunktLanguageVars
        language_punkt_vars.sent_end_chars = ('.', '?', ':')
        language_punkt_vars.internal_punctuation = (',', ';')
        trainer = PunktTrainer(train_data, language_punkt_vars)
        pickle_file = PUNCTUATION[self.language]['file']
        rel_path = os.path.join('~/cltk_data',
                                self.language,
                                'model/' + self.language + '_models_cltk/tokenizers/sentence')  # pylint: disable=C0301
        path = os.path.expanduser(rel_path)
        tokenizer_path = os.path.join(path, pickle_file)

        with open(tokenizer_path, 'wb') as open_pickle_file:
            pickle.dump(trainer, open_pickle_file)


    def _setup_language_variables(self, lang):
        """Check for language availability and presence of tokenizer file,
        then read punctuation characters for language and build tokenizer file
        path.
        :param lang: The language argument given to the class.
        :type lang: str
        :rtype (str, str, str)
        """
        assert lang in PUNCTUATION.keys(), \
            'Sentence tokenizer not available for {0} language.'.format(lang)
        internal_punctuation = PUNCTUATION[lang]['internal']
        external_punctuation = PUNCTUATION[lang]['external']
        file = PUNCTUATION[lang]['file']
        rel_path = os.path.join('~/cltk_data',
                                lang,
                                'model/' + lang + '_models_cltk/tokenizers/sentence')  # pylint: disable=C0301
        path = os.path.expanduser(rel_path)
        tokenizer_path = os.path.join(path, file)
        assert os.path.isfile(tokenizer_path), \
            'CLTK linguistics data not found for language {0}, {1} is not a file'.format(lang,tokenizer_path)
        return internal_punctuation, external_punctuation, tokenizer_path

    def _setup_tokenizer(self, tokenizer):
        """Add tokenizer and punctuation variables.
        :type tokenizer: object
        :param tokenizer : Unpickled tokenizer object.
        :rtype : object
        """
        language_punkt_vars = PunktLanguageVars
        language_punkt_vars.sent_end_chars = self.external_punctuation
        language_punkt_vars.internal_punctuation = self.internal_punctuation
        tokenizer.INCLUDE_ALL_COLLOCS = True
        tokenizer.INCLUDE_ABBREV_COLLOCS = True
        params = tokenizer.get_params()
        return PunktSentenceTokenizer(params)

    def tokenize_sentences(self, untokenized_string):
        """Tokenize sentences by reading trained tokenizer and invoking
        ``PunktSentenceTokenizer()``.
        :type untokenized_string: str
        :param untokenized_string: A string containing one of more sentences.
        :rtype : list of strings
        """
        # load tokenizer
        assert isinstance(untokenized_string, str), \
            'Incoming argument must be a string.'
        tokenizer = open_pickle(self.tokenizer_path)
        tokenizer = self._setup_tokenizer(tokenizer)

        # mk list of tokenized sentences
        tokenized_sentences = []
        for sentence in tokenizer.sentences_from_text(untokenized_string, realign_boundaries=True):  # pylint: disable=C0301
            tokenized_sentences.append(sentence)
        return tokenized_sentences
