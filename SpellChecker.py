from collections import defaultdict, Counter
import random
import math
import re


class SpellChecker:
    """
    Implements a context-sensitive spell checker using the Noisy Channel framework.

    The spell checker relies on a language model and an error distribution model to suggest corrections for
    misspelled words based on their surrounding context. The correction process is context-sensitive, meaning
    that the suggested correction is tailored to fit the context in which the misspelled word appears.
    """
    def __init__(self, lm=None):
        """Initializing a spell checker object with a language model as an
        instance  variable.

        Args:
            lm (Language_Model) : a language model object. Defaults to None.
        """
        self.lm = lm
        self.error_tables = None
        self.error_tables_with_proba = None

    def add_language_model(self, lm):
        """Adds the specified language model as an instance variable.
            (Replaces an older LM dictionary if set)

            Args:
                lm: a Spell_Checker.Language_Model object
        """
        self.lm = lm

    def add_error_tables(self, error_tables):
        """Adds the specified dictionary of error tables as an instance variable.
            (Replaces an older value dictionary if set)

            Args:
            error_tables (dict): a dictionary of error tables in the format
            of the provided confusion matrices:
            https://www.dropbox.com/s/ic40soda29emt4a/spelling_confusion_matrices.py?dl=0
        """
        self.error_tables = error_tables

    def evaluate_text(self, text):
        """Returns the log-likelihood of the specified text given the language
            model in use. Smoothing should be applied on texts containing OOV words
    
           Args:
               text (str): Text to evaluate.
    
           Returns:
               float: The float should reflect the (log) probability.
        """
        if self.lm is None:
            return 0
        else:
            return self.lm.evaluate_text(text)

    def spell_check(self, text, alpha):
        """Returns the most probable fix for the specified text. Use a simple
            noisy channel model if the number of tokens in the specified text is
            smaller than the length (n) of the language model.

            Steps:
                1. Preprocess the input text by normalizing it and splitting it into tokens.
                2. Generate and process candidate words for each token in the text, considering both the error model
                 probabilities and the probability of keeping a word as is (given by alpha).
                3. Calculate the probabilities for each candidate sentence based on the language model and the
                processed candidate words.
                4. Select and return the most probable candidate sentence based on the language model's evaluation.

            Args:
                text (str): the text to spell check.
                alpha (float): the probability of keeping a lexical word as is.

            Returns:
                str: A modified string (or a copy of the original if no corrections are made.)
        """
        self._update_error_table_with_proba(self.error_tables)
        text = normalize_text(text)
        tokens = text.split()
        word_candidates_proba = defaultdict(dict)

        for token in tokens:
            candidates_after_edits = self._generate_candidates(token)
            word_candidates_proba[token] = self._normalize_proba_to_alpha(candidates_after_edits, alpha)
            word_candidates_proba[token][token] = alpha

        word_candidates_proba = self._normalize_proba(word_candidates_proba)

        sentences_with_proba = self._get_sentences_with_proba(tokens, word_candidates_proba, self.lm.contexts_dict,
                                                              self.lm.vocab_distribution, self.lm.vocab_size)

        candidate_sentences = set(max(sentences, key=sentences.get) for sentences in sentences_with_proba.values())

        return max(candidate_sentences, key=lambda sentence: self.evaluate_text(sentence))

    def _get_word_probability(self, candidate, context, contexts_dict, vocab_distribution, vocab_size):
        """Calculate the probability of a candidate word given its context.

        Args:
            candidate (str): The candidate word for which the probability is being calculated.
            context (str): The context in which the candidate word appears.
            contexts_dict (dict): A dictionary containing the language model's contexts and their
                                  corresponding word probabilities.
            vocab_distribution (dict): A dictionary containing the distribution of words in the
                                      vocabulary.
            vocab_size (int): The size of the vocabulary.

        Returns:
            float: The probability of the candidate word given its context.
        """
        if context in contexts_dict:
            return contexts_dict[context].get(candidate, 0) / sum(contexts_dict[context].values())
        else:
            return vocab_distribution[candidate] / vocab_size

    def _get_sentences_with_proba(self, tokens, word_candidates_proba, contexts_dict, vocab_distribution, vocab_size):
        """Compute the probability of each candidate sentence based on the word candidates and language model.

        Args:
            tokens (list): A list of tokens from the input text.
            word_candidates_proba (dict): A dictionary containing the probability of each candidate word
                                          for each token in the input text.
            contexts_dict (dict): A dictionary containing the language model's contexts and their
                                  corresponding word probabilities.
            vocab_distribution (dict): A dictionary containing the distribution of words in the
                                      vocabulary.
            vocab_size (int): The size of the vocabulary.

        Returns:
            dict: A dictionary containing the probability of each candidate sentence for each token
                  in the input text.
        """
        sentences_with_proba = {token: {} for token in tokens}

        for index, token in enumerate(tokens):
            start_context = max(0, index - self.lm.n)
            context = ' '.join(tokens[start_context:index])

            for candidate, proba in word_candidates_proba[token].items():
                word_proba = self._get_word_probability(candidate, context, contexts_dict, vocab_distribution,
                                                        vocab_size)
                new_tokens = tokens[:index] + [candidate] + tokens[index + 1:]
                full_sentence = ' '.join(new_tokens)
                sentences_with_proba[token][full_sentence] = proba * word_proba

        return sentences_with_proba

    def _generate_candidates(self, word):
        """Generate in-vocabulary candidate words within 1 and 2 edit distances from the input word, along with their
         probabilities.

        Args:
            word (str): The input word for which candidate words are generated.

        Returns:
            dict: A dictionary containing the candidate words and their corresponding probabilities in the form:
                  {word: {'corr_1': proba_1, ...}}
        """
        edits1 = self._generate_candidates_with_proba_one_edit(word, self.error_tables_with_proba)
        edits2 = self._generate_candidates_with_proba_two_edits(edits1)

        candidates_after_second_edit = {k: edits1[k] if k in edits1 else edits2[k] for k in set(edits1) | set(edits2)}

        candidates_in_vocab = {candidate: proba for candidate, proba in candidates_after_second_edit.items()
                               if candidate in self.lm.vocab_distribution}

        return candidates_in_vocab

    def _normalize_proba(self, word_candidates_proba):
        """Normalize the probabilities of candidate words for each word in the input dictionary.

        Args:
            word_candidates_proba (dict): A dictionary containing input words and their candidate words
                                          with probabilities, in the format:
                                          {word_1: {candidate_1: proba_1, ...}, ...}

        Returns:
            dict: A normalized dictionary of word candidates and their probabilities, with the same structure
                  as the input dictionary.
        """
        normalized_dict = {}

        for word, candidates in word_candidates_proba.items():
            total_prob = sum(candidates.values())
            normalized_candidates = {candidate: prob / total_prob for candidate, prob in candidates.items()}
            normalized_dict[word] = normalized_candidates

        return normalized_dict

    def _normalize_proba_to_alpha(self, word_candidates_proba, alpha):
        """Normalize the probabilities of candidate words to 1 - alpha.

        Args:
            word_candidates_proba (dict): A dictionary containing candidate words and their
                                          probabilities, in the format:
                                          {candidate_1: proba_1, ...}
            alpha (float): The probability of keeping a lexical word as is.

        Returns:
            dict: A normalized dictionary of candidate words and their probabilities, with the
                  same structure as the input dictionary.
        """
        proba_sum = sum(word_candidates_proba.values())
        normalized_word_candidates_proba = {k: (v / proba_sum) * (1 - alpha) for k, v in word_candidates_proba.items()}
        num_decimals = 10
        keys = list(normalized_word_candidates_proba.keys())

        for i, k in enumerate(keys[:-1]):
            normalized_word_candidates_proba[k] = round(normalized_word_candidates_proba[k], num_decimals)

        normalized_word_candidates_proba[keys[-1]] = round(1 - alpha - sum(normalized_word_candidates_proba[k]
                                                                           for k in keys[:-1]), num_decimals)

        return normalized_word_candidates_proba

    def _update_error_table_with_proba(self, error_tables):
        """Compute smoothed probabilities for each error in the error tables.

        This function calculates the smoothed probability for each error in the input error tables using
        Laplace smoothing. For example, for deletion[tz]: proba = (del[tz] + 1) / (count(tz) + |alphabet size = 26|).
        Then it updates the variable `error_tables_with_proba` with the calculated probabilities.

        Args:
            error_tables (dict): A dictionary containing the input error tables.

        Returns:
            None
        """
        error_tables_with_proba = error_tables.copy()

        self._update_deletion_proba(error_tables, error_tables_with_proba)
        self._update_insertion_proba(error_tables, error_tables_with_proba)
        self._update_substitution_proba(error_tables, error_tables_with_proba)
        self._update_transposition_proba(error_tables, error_tables_with_proba)

        self.error_tables_with_proba = error_tables_with_proba

    def _update_deletion_proba(self, error_tables, error_tables_with_proba):
        """Update the deletion probabilities in the confusion matrix.

        Args:
            error_tables (dict): A dictionary containing the input error tables with counts.
            error_tables_with_proba (dict): A dictionary containing the input error tables with probabilities.

        Returns:
            None
        """
        alphabet_size = 26

        for err_corr, value in error_tables['deletion'].items():
            if err_corr[0] == '#':
                count = self.lm.chars_distribution[err_corr[-1]]
            else:
                count = self.lm.chars_distribution[err_corr]
            error_tables_with_proba['deletion'][err_corr] = (value + 1) / (count + alphabet_size)

    def _update_insertion_proba(self, error_tables, error_tables_with_proba):
        """Update the insertion probabilities in the confusion matrix.

        Args:
            error_tables (dict): A dictionary containing the input error tables with counts.
            error_tables_with_proba (dict): A dictionary containing the input error tables with probabilities.

        Returns:
            None
        """
        alphabet_size = 26

        for (err, corr), value in error_tables['insertion'].items():
            if err == '#':
                count = self.lm.chars_distribution[corr]
            else:
                count = self.lm.chars_distribution[err]
            error_tables_with_proba['insertion'][err+corr] = (value + 1) / (count + alphabet_size)

    def _update_substitution_proba(self, error_tables, error_tables_with_proba):
        """Update the substitution probabilities in the confusion matrix.

        Args:
            error_tables (dict): A dictionary containing the input error tables with counts.
            error_tables_with_proba (dict): A dictionary containing the input error tables with probabilities.

        Returns:
            None
        """
        alphabet_size = 26

        for (err, corr), value in error_tables['substitution'].items():
            count = self.lm.chars_distribution[corr]
            error_tables_with_proba['substitution'][err+corr] = (value + 1) / (count + alphabet_size)

    def _update_transposition_proba(self, error_tables, error_tables_with_proba):
        """Update the transposition probabilities in the confusion matrix.

        Args:
            error_tables (dict): A dictionary containing the input error tables with counts.
            error_tables_with_proba (dict): A dictionary containing the input error tables with probabilities.

        Returns:
            None
        """
        alphabet_size = 26

        for err_corr, value in error_tables['transposition'].items():
            count = self.lm.chars_distribution[err_corr]
            error_tables_with_proba['transposition'][err_corr] = (value + 1) / (count + alphabet_size)

    def _generate_candidates_with_proba_one_edit(self, word, error_tables_with_proba):
        """Generate all possible candidates and their probabilities for a given word within one edit distance.

        Args:
            word (str): The input word.
            error_tables_with_proba (dict): A dictionary containing the error distribution table with calculated
            probabilities.

        Returns:
            dict: A dictionary of all possible candidates for a word and their associated probabilities.
        """
        candidates_proba = {}

        self._deletion(word, error_tables_with_proba, candidates_proba)
        self._insertion(word, error_tables_with_proba, candidates_proba)
        self._substitution(word, error_tables_with_proba, candidates_proba)
        self._transposition(word, error_tables_with_proba, candidates_proba)

        return candidates_proba

    def _generate_candidates_with_proba_two_edits(self, candidates_with_proba):
        """Generate all possible candidates and their probabilities for a given set of candidates within two edit
         distances.

        Args:
            candidates_with_proba (dict): A dictionary of candidate words with their probabilities after the first edit.

        Returns:
            dict: A dictionary of candidates that appear in the corpus after the second edit, along with their
             associated probabilities.
        """
        candidates_proba_second_edit = {}

        for candidate, proba in candidates_with_proba.items():
            candidates_proba = self._generate_candidates_with_proba_one_edit(candidate, self.error_tables_with_proba)

            for candidate_after_edit, proba_ in candidates_proba.items():
                candidates_proba_second_edit[candidate_after_edit] = proba_ * proba

        return candidates_proba_second_edit

    def _deletion(self, word, error_tables, correction_proba):
        """Generate all possible candidates for a given word within one edit distance using deletion operations.

        Args:
            word (str): The input word.
            error_tables (dict): A dictionary containing error distribution tables.
            correction_proba (dict): A dictionary to be updated with candidates and their probabilities.

        Returns:
            None
        """
        for (x, y), value in error_tables["deletion"].items():
            candidates = [(y + word, value)] if x == '#' else [(word[:i] + x + y + word[i + 1:], value) for i in
                                                               range(len(word)) if word[i] == x]
            correction_proba.update(candidates)

    def _insertion(self, word, error_tables, correction_proba):
        """Generate all possible candidates for a given word within one edit distance using insertion operations.

        Args:
            word (str): The input word.
            error_tables (dict): A dictionary containing error distribution tables.
            correction_proba (dict): A dictionary to be updated with candidates and their probabilities.

        Returns:
            None
        """
        for (x, y), value in error_tables["insertion"].items():
            candidates = [(word[1:], value)] if x == '#' and word.startswith(y) else [
                (word[:i] + x + word[i + 2:], correction_proba.get(word[:i] + x + word[i + 2:], value)) for i in
                range(len(word) - 1) if word[i:i + 2] == x + y]

            correction_proba.update(candidates)

    def _substitution(self, word, error_tables, correction_proba):
        """Generate all possible candidates for a given word within one edit distance using substitution operations.

        Args:
            word (str): The input word.
            error_tables (dict): A dictionary containing error distribution tables.
            correction_proba (dict): A dictionary to be updated with candidates and their probabilities.

        Returns:
            None
        """
        for (x, y), value in error_tables["substitution"].items():
            candidates = [(word[:i] + y + word[i + 1:], correction_proba.get(word[:i] + y + word[i + 1:], value)) for i
                          in range(len(word)) if word[i] == x]

            correction_proba.update(candidates)

    def _transposition(self, word, error_tables, correction_proba):
        """Generate all possible candidates for a given word within one edit distance using transposition operations.

        Args:
            word (str): The input word.
            error_tables (dict): A dictionary containing error distribution tables.
            correction_proba (dict): A dictionary to be updated with candidates and their probabilities.

        Returns:
            None
        """
        for (x, y), value in error_tables["transposition"].items():
            candidates = [
                (word[:i] + y + x + word[i + 2:], correction_proba.get(word[:i] + y + x + word[i + 2:], value)) for i in
                range(len(word) - 1) if word[i:i + 2] == x + y]

            correction_proba.update(candidates)

    #####################################################################
    #                   Inner class                                     #
    #####################################################################

    class LanguageModel:
        """
        Implements a Markov Language Model that is able to learn from a given text, and supports both language
        generation and the evaluation of a given string. This model can operate at both the word level and the
        character level, offering flexibility in the types of text inputs it can handle.
        """
        def __init__(self, n=3, chars=False):
            """Initializing a language model object.

                Args:
                    n (int): the length of the markov unit (the n of the n-gram). Defaults to 3.
                    chars (bool): True iff the model consists of ngrams of characters rather than word tokens.
                                  Defaults to False
            """
            self.n = n
            self.model_dict = None
            self.chars = chars
            # dict in form of n - 1 gram and the counts of the next possible nth gram for space and complexity
            # reasons, as depicted in assignment 0.
            self.contexts_dict = None
            self.vocab_size = 0
            self.vocab_distribution = None
            self.chars_distribution = None

        def build_model(self, text):
            """Populates the instance variable model_dict.

                Args:
                    text (str): the text to construct the model from.
            """
            text = normalize_text(text)

            if self.chars:
                text = self._convert_to_space_separated_chars(text)

            tokens = text.split()
            self.vocab_size = len(tokens)
            self.vocab_distribution = Counter(tokens)
            self.model_dict = defaultdict(int)
            contexts = defaultdict(int)
            self.contexts_dict = defaultdict(lambda: defaultdict(int))
            n = self.n

            for i in range(len(tokens) - n + 1):
                self.model_dict[' '.join(tokens[i:i + n])] += 1
                contexts[' '.join(tokens[i:i + n - 1])] += 1
                self.contexts_dict[' '.join(tokens[i:i + n - 1])][tokens[i + n - 1]] += 1

            self._update_chars_distribution()

        def get_model_dictionary(self):
            """Returns the dictionary class object.

            Returns:
                dict: model dictionary in the form {ngram:count}
            """
            return self.model_dict

        def get_model_window_size(self):
            """Returns the size of the context window (the n in "n-gram").

            Returns:
                int: ngram window size.
            """
            return self.n

        def generate(self, context=None, n=20):
            """
            Returns a string of the specified length, generated by applying the language model
            to the specified seed context. If no context is specified the context should be sampled
            from the models' contexts distribution. Generation should stop before the n'th word if the
            contexts are exhausted. If the length of the specified context exceeds (or equal to)
            the specified n, the method should return a prefix of length n of the specified context.

            Args:
                context (str): a seed context to start the generated string from. Defaults to None
                n (int): the length of the string to be generated.

            Return:
                str: The generated text.
            """
            if context is None:
                context = self._sample_word_from_dist(self.vocab_distribution)
            else:
                context = normalize_text(context)
                if self.chars:
                    context = self._convert_to_space_separated_chars(context)

                context_tokens = context.split()
                if len(context_tokens) >= n:
                    return ' '.join(context_tokens[:n])

            generated = context.split()

            while len(generated) < n:
                context = ' '.join(generated[-self.n + 1:])
                # if the context exists in the dict sample the most probable context to appear after that

                if len(self.contexts_dict[context]) > 0:
                    chosen_context = [self._sample_word_from_dist(self.contexts_dict[context])]
                else:
                    # choose randomly the next context
                    chosen_context = random.choice(list(self.contexts_dict.keys())).split()

                generated.extend(chosen_context)

            return ' '.join(generated[:n])

        def evaluate_text(self, text):
            """Returns the log-likelihood of the specified text to be a product of the model.
               Laplace smoothing should be applied if necessary.

               Args:
                   text (str): Text to evaluate.

               Returns:
                   float: The float should reflect the (log) probability.
            """
            text = normalize_text(text)

            if self.chars:
                text = self._convert_to_space_separated_chars(text)

            tokens = text.split()
            log_likelihood = 0

            for i in range(len(tokens) - self.n + 1):
                ngram = ' '.join(tokens[i:i + self.n])
                log_likelihood += math.log(self.smooth(ngram))

            return log_likelihood

        def smooth(self, ngram):
            """Returns the smoothed (Laplace) probability of the specified ngram.

                Args:
                    ngram (str): the ngram to have its probability smoothed

                Returns:
                    float: The smoothed probability.
            """
            ngram = normalize_text(ngram)
            if self.chars:
                ngram = self._convert_to_space_separated_chars(ngram)

            count = self.model_dict[ngram]
            num_of_ngrams_in_corpus = sum(list(self.model_dict.values()))
            num_of_unique_ngrams_in_corpus = len(self.model_dict)

            # return the laplace smoothing based on the formula: (c+1)/(N+V)
            return (count + 1) / (num_of_ngrams_in_corpus + num_of_unique_ngrams_in_corpus)

        def _get_vocab_size(self):
            """Returns the (non-unique) word count in the text.

                Returns:
                    int: word count.
            """
            return self.vocab_size

        def _get_vocab_distribution(self):
            """Returns the word distribution in the text.

                Returns:
                     dict: in the format {word:count}.
            """
            return self.vocab_distribution

        def _sample_word_from_dist(self, dist_dict):
            """Returns a random word sampled from a given word distribution.
                Args:
                    dist_dict (dict): A dictionary in the form {word:count}

                Returns :
                    str: A random word.
            """
            words = list(dist_dict.keys())
            counts = list(dist_dict.values())
            chosen_word = random.choices(words, weights=counts)[0]

            return chosen_word

        def _convert_to_space_separated_chars(self, text):
            """Convert a string of characters into a space-separated string.

            Args:
                text (str): The input text containing characters.

            Returns:
                str: A space-separated string of characters.
            """
            char_list = [char for char in text.replace(' ', '')]
            text = ' '.join(char_list)

            return text

        def _update_chars_distribution(self):
            """
            An assisting function that updates the character and two-character combination statistics in the text.

            Returns:
                None
            """
            alphabet = 'abcdefghijklmnopqrstuvwxyz'

            keys = [a for a in alphabet]
            keys.extend([a + b for b in alphabet for a in alphabet])
            char_count_dict = defaultdict(int, {key: 0 for key in keys})

            for comb in char_count_dict.keys():
                char_count_dict[comb] = self._count_appearnces(comb, self.vocab_distribution)

            self.chars_distribution = char_count_dict

        def _count_appearnces(self, substring, vocab):
            """
            Returns the total count of a given substring in the vocabulary.

            Args:
                substring (str): The substring for which the count is to be computed.
                vocab (dict): A dictionary containing word and count pairs, in the format {word: count}.

            Returns:
                int: The total count of the specified substring in the vocabulary.
            """
            counter = 0

            for word, count in vocab.items():
                counter += (word.count(substring)) * count

            return counter


def normalize_text(text):
    """Returns a normalized version of the specified string.
      You can add default parameters as you like (they should have default values!)
      You should explain your decisions in the header of the function.

      Args:
        text (str): the text to normalize

      Returns:
        str: the normalized text.
    """
    text = text.lower()
    # remove url and email patters
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    email_pattern = re.compile(r'\S+@\S+\.\S+')
    text = url_pattern.sub('', text)
    text = email_pattern.sub('', text)
    # remove numbers
    text = re.sub(r'\d+', '', text)
    pat_re = re.compile("([.?!\;`,\—\-'\’\"\“])")
    # padding special chars with a white space
    text = pat_re.sub(" \\1 ", text)
    # removing sequences white spaces
    text = re.sub('\s+', ' ', text)
    # removing punctuation and special chars
    text = pat_re.sub('', text)
    return ' '.join(re.findall(r'\w+', text))
