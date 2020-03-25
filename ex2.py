
import re
from collections import defaultdict
import math
from nltk.util import ngrams
from nltk import sent_tokenize
from collections import Counter


class Spell_Checker:
    """The class implements a context sensitive spell checker. The corrections
        are done in the Noisy Channel framework, based on a language model and
        an error distribution model.
    """

    def __init__(self,lm=None):
        """Initializing a spell checker object with a language model as an
        instance variable.

        Args:
            lm: a language model object. Defaults to None
        """
        self.error_types=('substitution','transposition','insertion','deletion')
        self.error_distribution = {t:defaultdict(float) for t in self.error_types}
        
        #create Language Model from inner class
        if not lm:
            self.lm = Ngram_Language_Model()
        
        if lm:
            self.lm = lm
            self.vocab = get_vocab_from_ngram(lm.get_model())

            
    
    def build_model(self, text, n = 3):
        """Returns a language model object built on the specified text.

            Args:
                text (str): the text to construct the model from.

            Returns:
                A language model object
        """
        
        try:
            self.lm.build_model(text,n)
        except:
            raise
        
        self.vocab = Counter(words(text))

        return self.lm
      
        

    def add_language_model(self,lm):
        """Adds the specified language model as an instance variable.
            (Replaces an older LM disctionary if set)

            Args:
                lm: a language model object
        """
        self.lm = lm
        try:
            self.vocab = get_vocab_from_ngram(lm.get_model())
        except:
            raise
        

    def learn_error_distribution(self, errors_file):
        """Returns a dictionary {str:dict} where str is in:
            <'deletion', 'insertion', 'transposition', 'substitution'> and the
            inner dict {tupple: float} represents the confution matrix of the
            specific errors, where tupple is (err, corr) and the float is the
            probability of such an error.
            Examples of such tupples are ('t', 's'), for deletion of a 't'
            after an 's', insertion of a 't' after an 's' and substitution
            of 's' by a 't'; and example of a transposition tupple is ('ac','ca').
            In the case of insersion, the tuppe (i,j) reads as "i was mistakingly
            added after j". In the case of deletion, the tupple reads as
            "i was mistakingly ommitted after j"

            Notes:
                1. The error distributions could be represented in more efficient ways.
                    We ask you to keep it simple and straight forward for clarity.
                2. Ultimately, one can use only 'deletion' and 'insertion' and have
                    'substitution' and 'transposition' derived. Again,  we use all
                    four types explicitly in order to keep things simple.
            Args:
                errors_file (str): full path to the errors file. File format, TSV:
                                    <error>    <correct>


            Returns:
                A dictionary of error distributions by error type (dict).
        """
        error_text=read_file(errors_file)
        list_err_corr=clean_error_text([tuple(tokenize(pair)) for pair in error_text.split('\n')])
        text_flat=' '.join([t[0]+' '+t[1] for t in list_err_corr])
        
        #calc counts of error
        for err , corr in list_err_corr:
            for error_type,err_letters in find_error_pair(err,corr):
                self.error_distribution[error_type][err_letters]+=1
        
        #counter all characters for normilze
        chrs = [c for c in text_flat]
        counter_chars = Counter(chrs)
        counter_chars += Counter([''.join(ca) for ca in list(ngrams(chrs,2))])
        
        #calc distribution
        self.error_distribution = normilize_distribution(self.error_distribution,counter_chars)
        
        return self.error_distribution
        
        

    def add_error_tables(self, error_tables):
        """ Adds the speficied dictionary of error tables as an instance variable.
            (Replaces an older value disctionary if set)

            Args:
                error_tables (dict): a dictionary of error tables in the format
                returned by  learn_error_distribution()
        """
        self.error_distribution = error_tables


    def evaluate(self,text):
        """Returns the log-likelihod of the specified text given the language
            model in use. Smoothing is applied on texts containing OOV words

           Args:
               text (str): Text to evaluate.

           Returns:
               Float. The float should reflect the (log) probability.
        """
        return self.lm.evaluate(text)
        
    def spell_check(self, text, alpha = 0.95):
        """ Returns the most probable fix for the specified text. Use a simple
            noisy channel model is the number of tokens in the specified text is
            smaller than the length (n) of the language model.
            
            Check if there is a number of sentences for separate by a period. 
            Then fix each sentence. afterwards return all sentences concat with period.

            Args:
                text (str): the text to spell check.
                alpha (float): the probability of keeping a lexical word as is.

            Return:
                A modified string (or a copy of the original if no corrections are made.)
        """
        sentences = normalize_text_sentences(text)
        return ' . '.join([self.fix_sentence(sent,alpha) for sent in sentences])
    
        
    def fix_sentence(self,text,alpha):
        '''
        Returns the sentence with the maximum log likelihood. There is two error type possible:
            1. One of the 'text' words are not in the vocabulary
            2. All the 'text' words in the vocabulary but the context of the sentence is incorect.
            find the candidate that known from the vocabulary and create a complete sentence and evalute.
            save the sentence and its value (log likelihood) in a dictionary {'sentence':log-likelihood}
            
            1. The first error type the function search all known words in the vocabulary with edit distance 1.
            and change with the bad word and evalute the sentence for all permutation.
            2. The second need to check for each word in the sentecne is candidates and then calculate by evalute 
            all the possible permutation.
        Args:
            text (str): text to evalute error and correct if necessary
            alpha (float): the probabilty that the worng word are correct and just not contains in the vocabulary
        Return:
            A modified string (or a copy of the original if no corrections are made.)
        '''
        sentence_dict={}
        sum_of_priors = 0
        tokens=tokenize(text)
        all_words_in_vocab = self.words_in_vocab(tokens)
        sentence_dict[' '.join(tokens)] = math.log(alpha) + self.evaluate(' '.join(tokens))
        for i,token in enumerate(tokens):
            if self.known([token]) and not all_words_in_vocab:
                continue
            else:
                for cand in self.candidates(token):
                    new_tokens=tokens.copy()
                    new_tokens[i]=cand
                    err_tuple_li = find_error_pair(token,cand)
                    prob_xw = self.max_error_tuple(err_tuple_li)
                    prob_xw = (1/len(self.vocab)) if not prob_xw else prob_xw
                    sum_of_priors+=prob_xw
                    sentence_dict[' '.join(new_tokens)] = math.log(1-alpha) + math.log(prob_xw) + self.evaluate(' '.join(new_tokens))
        
        #normilize the prior characters error by 1-alpha
        if sum_of_priors:
            for key in sentence_dict:
                if key == text:
                    continue
                else:
                    sentence_dict[key] -= math.log(sum_of_priors)
        return max(sentence_dict, key=sentence_dict.get)
        
    def max_error_tuple(self,err_tuple_li):
        '''
        returns the max probabilty for multiple error for the same error type and candidate.
        for example: error: 'botle' correct: 'bottle' -> ('deletion',('t','o')),('deletion',('t','t'))
            and return the maximum probability between both errors. 
            Return the probability between ('t','t') and ('t','o').
        Args:
            err_tuple_li (list) contains tuples of error in the foramt of error_distribution
        Returns:
            float max probability for the same candidate.
        '''
        max_prob = 0
        for err_tuple in err_tuple_li:
            prob_xw = self.error_distribution.get(err_tuple[0],{}).get(err_tuple[1],0)
            max_prob=max(max_prob,prob_xw)
        return max_prob
    
    def words_in_vocab(self,words):
        '''
        returns true if all words exists in vocabulary
        Arg: 
            words (str or list): list of str of string with spaces 
        return:
            True/False
        '''
        if type(words) is list:
            return all([w in self.vocab for w in words])
        else:
            return all([w in self.vocab for w in words.split(' ')])
    
    
    def known(self,words): 
        '''
        Returns all the subset of words that exists in the vocabulary
        Args:
            words (list): list of str
        Return:
            list after filter if words exists in the vocabulary
        '''
        words_li=[]
        for w in words:
            if ' ' in w and self.words_in_vocab(w):
                words_li.append(w)
            elif w in self.vocab:
                 words_li.append(w)
        return words_li
    
    
    def candidates(self,word):
        '''
        Returns a set contain all the words that in edit distance 1 from 'word' 
            and exists in the vocabulary
        Arg:
            word (str): string for edit distance 1
        return:
            set of string with edit distance 1 from word
        '''
        "Generate possible spelling corrections for word."
        return set(self.known(edits1(word))).difference(set([word]))


class Ngram_Language_Model:
    """The class implements a Markov Language Model that learns a model from a given text.
        It supoprts language generation and the evaluation of a given string.
        The class can be applied on both word level and caracter level.
    """

    
    def __init__(self, chars=False):
        """Initializing a language model object.
        Arges:
            chars (bool): True iff the model consists of ngrams of characters rather then word tokens.
                          Defaults to False
        """
        self.chars = chars
        self.lm_dict = defaultdict(lambda: defaultdict(float))
        

    def build_model(self , text, n=3):  #should be called build_model
        """populates a dictionary counting all ngrams in the specified text.

            Args:
                text (str): the text to construct the model from.
                n (int): the length of the markov unit (the n of the n-gram). Defaults to 3.
                model in form of dictionary of dictionaries {tuple():dict}.
                In the tuple there is ngram-1 tuple(ngram,ngram,....) and in the inner dictonary
                there is the last ngram. for example: 
                    [tuple(gram0,gram1)][gram2]=val
                
        """
        self.n = n
        self.vocab = Counter(words(text))

        tokens=tokenize(text)
        for gram in list(ngrams(tokens,self.n)):
            self.lm_dict[tuple(gram[:-1])][gram[-1]]+=1
    
    
    def get_model(self):
        """Returns the model as a dictionary of dictionaries in the form {tuple():dict}.
            In the tuple there is ngram-1: tuple(ngram,ngram,....) and in the inner dictionary
            there is the last ngram. for example:  [tuple(gram0,gram1)][gram2]=val
        """
        return self.lm_dict
    
    
    def evaluate(self,text):
        """Returns the log-likelihod of the specified text to be generated by the model.
           Laplace smoothing should be applied if necessary.

           Args:
               text (str): Text to ebaluate.

           Returns:
               Float. The float should reflect the (log) probability.
        """
        tokens=tokenize(text)
        if len(tokens)<self.n:
            return self.evaluate_n_minus1(tuple(tokens))
        
        ngrams_list = list(ngrams(tokens,self.n))
        value=0

        for i, ng in enumerate(ngrams_list):
            gram2,gram1=tuple(ng[:-1]),ng[-1]
#            if not i:
#                value = self.evaluate_n_minus1(gram2)
            denominator = sum(self.lm_dict[gram2].values())
            numerator = self.lm_dict[gram2][gram1]  
            if numerator == 0 or denominator == 0:
                prob = self.smooth(numerator , denominator)
            else:
                prob = numerator / denominator
            value += math.log(prob)
        return value

    
    def evaluate_n_minus1(self,grams):
        '''
        Returns evaluation of prior and ngram-1 tokens. find partial keys that represent the start of ngram
        and find the prior and the conditional probability for ngram-1. return log-likelihood
        Args:
            grams (tuple): ngrams-1 of tokens
        Return:
            Float. The float should reflect the (log) probability of the ngram-1 tokens 
        '''
        value=0
        for i in range(0,len(grams)):
            total = 0
            counter = 0
            total = sum(self.vocab.values())
            counter = self.vocab.get(tuple(grams)[i],0)
        
            if total == 0 or counter == 0:
                prob = self.smooth(counter , total)
            else:
                prob = counter / total
            value += math.log(prob)
        return value
        
    
    def smooth(self, numerator , denominator):
        """Returns the smoothed (Laplace) probability of the specified ngram.
        
            Args:
                ngram (str): the ngram to have it's probability smoothed
        
            Returns:
                float. The smoothed probability.
        """  
        return (numerator+1)/(denominator+len(self.vocab))
         
  
def who_am_i(): #this is not a class method
    """Returns a ductionary with your name, id number and email. keys=['name', 'id','email']
        Make sure you return your own info!
    """
    return {'name': 'Mor Kertis', 'id': '300830692', 'email': 'morker@post.bgu.ac.il'}



def read_file(file_path):
    '''
    Returns strings of the file after reading the file
    :Args: 
        file_path (str): path of file 
    :Return: 
        strings of the file
    '''
    try:
        with open(file_path, 'r',encoding="utf8") as file:
            text = file.read()
        return text
    except:
        raise
        
def normalize_text_sentences(text,pad_punc='!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~',remove_punc='!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'):
    """Returns a normalized list of sentences based on the specifiy string.
        explain: 
         1. lower case of all characters. It will remain less characters that is the same in understanding the sentance
         2. Add spaces between punctuation simbols for better separting between words. but it can change by parametrs(for hashtag # for example)
         3. Remove punctuation exlude '.'. Those punctuation may not contribute to the sentance. but it can change by parametrs(for hashtag # for example)
         4. Remove extra spaces in a row and leave only one space between tokens
         5. Multi dots in a row switch to 'dots'(str only in word normilze and not characters normilize)
       Args:
           text (str): the text to normalize
           pad_punc(str): characters for creating a space before and after the characters (working only when normilize word and not characters)
                           default: all string punctuation
           remove_punc(str): characters to remove from the text
                           default: all string punctuation without '.'
           chars(boolean): True - normilize characters. False - normalize words
                           default: False
       Returns:
           List. of the normalized text.
    """
    normalize_text_list=[]
    for sent in list(sent_tokenize(text)):
        normalize_text_list.append(normalize_text(sent,pad_punc=pad_punc,remove_punc=remove_punc))
    return normalize_text_list


def normalize_text(text,pad_punc='!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~',remove_punc='!"#$%&\'()*+,-/:;<=>?@[\\]^_`{|}~',remove_number='[0-9]',chars=False):
    """Returns a normalized string based on the specifiy string.
        explain: 
         1. lower case of all characters. It will remain less characters that is the same in understanding the sentance
         2. Add spaces between punctuation simbols for better separting between words. but it can change by parametrs(for hashtag # for example)
         3. Remove punctuation exlude '.'. Those punctuation may not contribute to the sentance. but it can change by parametrs(for hashtag # for example)
         4. Remove extra spaces in a row and leave only one space between tokens
         5. Multi dots in a row switch to 'dots'(str only in word normilze and not characters normilize).
         6. Remove single characters if chars is False except 'a' and 'i'  
       Args:
           text (str): the text to normalize
           pad_punc(str): characters for creating a space before and after the characters (working only when normilize word and not characters)
                           default: all string punctuation
           remove_punc(str): characters to remove from the text
                           default: all string punctuation without '.'
           chars(boolean): True - normilize characters. False - normalize words
                           default: False
       Returns:
           string. the normalized text.
    """
    punc_spaces = re.compile('([%s])' % re.escape(pad_punc))
    punc = re.compile('[%s]' % re.escape(remove_punc))
    text = text.lower()
    if chars:
        text = re.sub(punc,'',text)
    else:
        text = re.sub('\.{3,}',' dots',text)
        text = re.sub(punc_spaces, r' \1 ', text)
        text = re.sub(remove_number,'',text)
        text = re.sub(punc,'',text)
        text = re.sub(r'\b((?![ai])[a-z])\b','',text)
        text = re.sub('\s{2,}', ' ', text)
        text = re.sub('\n', ' ', text)
        text = re.sub('\t', ' ', text)
        text=text.strip()
        
    return text


def tokenize(text,split_str='\s',chars=False):
    """
    Return list of tokens(text saparate) based on specific string and tokenize level
    Args:
        text(str): The text for tokenize
        chars(boolean): True - tokenize in character level . False - tokenize by space
                        default - False
        split_str(str): the sign/character the text is split
                        default - '\s'
    Returns:
        list contain a strings
    """
    if not chars:
        text=re.split(split_str,text)
    return [token for token in text if token not in [""]] 


def find_error_pair(err,corr):
    '''
    Returns error type from the str <'deletion', 'insertion', 'transposition', 'substitution'>
        and the error and correct letter in the form of: [(error_type,tuple(err_character,corr_character))].
        for example ('insertion',('a','b')) - insertion of 'a' before 'b'. 
        ***The full example is in learn_error_distribution method***
    Args:
        err (str): error string
        corr (str): correct string
    Returns:
        list of tuples in the form of [(error_type,(err_character,corr_character))]
    '''
    
    err_ln=len(err)
    corr_ln=len(corr)
    
    #check substitution or transposition
    if err_ln == corr_ln:
        error_latters=[(e,c) for e,c in zip(err,corr) if e !=c]
        if len(error_latters)==0:
            return [] # the same string
        if len(error_latters) == 1:
            return [('substitution',error_latters[0])]
        else:
            error_latters=tuple(''.join(tup) for tup in error_latters)
            return [('transposition',error_latters)]
    
    #insertion
    if err_ln>corr_ln:
        insertion_li=[]
        last=''
        for i,e in enumerate(err):
            if corr_ln<=i or e!=corr[i]:
                insertion=(e,last)
                insertion_li.append(('insertion',insertion))
                break
            last=e
        if insertion[0]==insertion[1]: # check repeated letters error
            start,end=re.search(''.join(insertion),err).span()
            if not start:
                insertion=('',err[end-1])
            else:
                insertion=(err[end-1],err[start-1])

            insertion_li.append(('insertion',insertion))
        return insertion_li
    
    #deletion
    if err_ln<corr_ln:
        deletion_li=[]
        last=''
        for i,c in enumerate(corr):
            if i>=err_ln or c!=err[i]:
                deletion=(c,last)
                deletion_li.append(('deletion',deletion))
                break
            last=c
        if deletion[0]==deletion[1]: # check repeated letters error
            start,end = re.search(''.join(deletion),corr).span()
            if not start:
                deletion=(corr[end-1],'')
            else:
                deletion=(corr[end-1],corr[start-1])
            deletion_li.append(('deletion',deletion))
        return deletion_li


def clean_error_text(list_tuple):
    '''
    Returns list of tuple contains error string and correct string in the form of [(error,correct),...]
        before filter some words and fix tuple size by:
            1. filter words when error and correct are the same word
            2. error and corrcet contain three words in the tuple - error stay the same and correct concat to one token
        afterwards all tuple will be in size of two.
    Args:
        list_tuple (list(tuple)): list of tuples contain str  [(error,correct),...]
    Returns:
        list of tuple after filter and fix tuple size
    '''
    err_list=[]
    for tuple_str in list_tuple:
        if not tuple_str:
            continue
        
        if len(tuple_str)==2 and tuple_str[0]==tuple_str[1]:
            continue
        
        if len(tuple_str)>2:
            tuple_str=(normalize_text(tuple_str[0]),normalize_text(' '.join(tuple_str[1:])))
        err_list.append(tuple_str)  
    return err_list


def normilize_distribution(error_distribution,counter_chars):
    '''
    Returns error distributions after normilize the value
    Args:
        error_distribution (dictionary): {str:dict} where str is in:
            <'deletion', 'insertion', 'transposition', 'substitution'> and the
            inner dict {tupple: float} represents the confution matrix of the
            specific errors, where tupple is (err, corr) and the float is the
            probability of such an error.
        counter_chars (dictionary): dictionary in the form of {str:float}. contain all letters as keys 
            and their value for normilize
    Returns:
        error distributions after normilize (dictionary): {str:dict}
    '''
    error_type_list=list(error_distribution.keys())
    for error_type in error_type_list:
        for tuple_error,count_error in error_distribution[error_type].items():
            search = str_to_nromilize(error_type , tuple_error)
            count_norm=counter_chars.get(search,0)+1
            error_distribution[error_type][tuple_error]=count_error/count_norm
    return error_distribution


def str_to_nromilize(error_type,tuple_error):
    '''
    Returns the characters for normilize error distributions
    Args:
        tuple_error (tuple): form of (err, corr) where err is error characters and corr is correct characters
        error_type (str): str is one of <'deletion', 'insertion', 'transposition', 'substitution'>
    Returns:
        the characters need for normilize the error tuple for error type
    '''
    if error_type=='deletion':
        search=''.join(tuple_error[::-1])
        
    elif  error_type=='insertion':
        search = tuple_error[1]
        
    elif error_type=='transposition':
        search = tuple_error[0]
        
    elif error_type=='substitution':
        search = tuple_error[0]
        
    return search
    

def words(text): 
    '''
    Returns list of all string in the text
    Args:
        text (str): string of text
    Returns: 
        List of words form the text
    '''
    return re.findall(r'\w+', text.lower())


def ngram_with_sentence(text,n):
    '''
    Returns list of tuple ngrams. In the form of [(gram0,gram1,gram2),....]
        after padding by string '<s>','</s>' between each sentence in the text
    Args:
        text (str): string of text
        n (int): size of ngrams
    Returns:
        list of tuple ngrams
    '''
    tokens=tokenize(text)
    return list(ngrams(tokens,n=n))
     

def edits1(word):
    '''
    Returns set of all words in one edit distance away from 'word'.
        check all types of error that include: <'deletion', 'insertion', 'transposition', 'substitution'>
    Args:
        word (str): string for creating edit distance
    Return set of words in one edit distance
    '''
    letters    = 'abcdefghijklmnopqrstuvwxyz '
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)


def get_vocab_from_ngram(dict_ngram):
    '''
    Returns vocabulary of words from ngrams language model.
    check if the language model in format dictionary key string {str:value} 
        or dictionary key tuple and key string {tuple:{str:value}}
    and return a dictionary {str:value} where str is a single word and the value is the count.
    Args:
        dict_ngram (dictionary) language model dictionary in two formats:
                string {str:value} or tuple string {tuple:{str:value}}
    Return:
        vocabulary in format of dictionary {str:value}
    '''
    key = next(iter(dict_ngram.keys()))
    
    if type(key) is str:
        n = len(key.split(' '))
        vocab = get_vocab_str(dict_ngram,n)
   
    else:  # my format {tuple:{str:value}} 
        n = len(key)+1
        vocab = get_vocab_tuple(dict_ngram,n)
    
    return vocab


def get_vocab_tuple(dict_tuple,n):
    '''
    Returns vocabulary of words from ngrams language model in format of key tuple key string {tuple:{str:value}}
    and return a dictionary {str:value} where str is a single word and the value is the count.
    Args:
        dict_tuple (dictionary) language model in format dictionary key tuple string {tuple:{str:value}}
    Return:
        vocabulary in format of dictionary {str:value}
    '''
    vocab_list=[]
    
    for key1,val1 in dict_tuple.items():
        for key2,val2 in val1.items():
            if key1:
                vocab_list.extend(list(key1)*int(val2))
            vocab_list.extend([key2]*int(val2))
    
    vocab = Counter(vocab_list)
    
    for key in vocab:
        vocab[key]=math.ceil(vocab[key] / n)
    return vocab

        
def get_vocab_str(dict_str,n):
    '''
    Returns vocabulary of words from ngrams language model in format of key string {str:value}
    and return a dictionary {str:value} where str is a single word and the value is the count.
    Args:
        dict_tuple (dictionary) language model in format dictionary key string {str:value}
    Return:
        vocabulary in format of dictionary {str:value}
    '''
    vocab_list=[]
    vocab = Counter(vocab_list)
    for key,val in dict_str.items():
        gram = key.split(' ')
        vocab_list.extend(gram*int(val))
        
    vocab = Counter(vocab_list)
    
    for key in vocab:
        vocab[key]=math.ceil(vocab[key] / n)
    return vocab



