import jieba
import numpy as np
import string
import subprocess
import re
# jieba.set_dictionary('data/DICT_CK+jieba_lower')
# jieba.add_word('龐燮傍謝', freq=10, tag='xx')
ANS = ['a', 'b', 'c', 'd', 'e']
WINDOW = 10
VEC_SIZE = 100



class TokenizerException(Exception):
    '''Have to specify which tokenizer to use, jieba or guofoo '''

class PandasRowException(Exception):
    '''Failed to parse a row in pandas table'''

def normalize_vec(vec):
    mag = ((vec * vec).sum()) ** 0.5
    return vec / mag


class Question(object):

    def __init__(self, row):
        try:
            self.no = row.no
            self.context = row.content.strip().lower()
            self.options = [x.strip().lower() for x in list(row[['a','b','c','d','e']].values)]
            self.ans = row.ans
        except Exception as e:
            print(e)
            raise PandasRowException

class TokenQuestion(Question):

    def __init__(self, row, tokenizer = None, *args, **kwargs):
        if not tokenizer or type(tokenizer) != str:
            raise TokenizerException
        super().__init__(row)

        self.tokenizer = tokenizer
        self.tokenize()


    def tokenize(self):
        context = self.context.strip().lower()
        if self.tokenizer == 'jieba':
            context = context.replace('︽⊙＿⊙︽', '龐燮傍謝')
            wlist = list(jieba.cut(context))
            qidx = []
            for i, w in enumerate(wlist):
                if w == '龐燮傍謝':
                    wlist[i] = '*'
                    qidx.append(i)
            self.wlist = wlist
            self.qidx = qidx
        elif self.tokenizer == 'guofoo':
            print('[tokenizer]oh... you forgot something')
        elif self.tokenizer == 'uni':
            context = context.replace('︽⊙＿⊙︽', '*')
            wlist = re.sub(r'\s+',' ', re.sub(r'([\u4E00-\u9FFF\*])', r' \1 ', context)).strip().split(' ')
            qidx = []
            for i, w in enumerate(wlist):
                if w == '*': qidx.append(i)
            self.wlist = wlist
            self.qidx = qidx


class Solver(object):
    UNSOLVED = -2
    UNKNOWN = -1
    '''
    This is an interface for models which claim to find the answer of the cloze context.
    '''
    def __init__(self):
        self.options_prob = {}
        self.prediction = Solver.UNSOLVED

    def solve(self, question, *args, **kwargs):
        raise NotImplementedError( "Subclass should have implemented solve method" )


class LanguageModelSolver(Solver):
    def __init__(self, path, prefix):
        import kenlm
        # self.model = kenlm.Model('lm/test.arpa')
        pass

    def solve:
        pass

class Word2VecSolver(Solver):

    def __init__(self, path, prefix):
        import gensim
        from smart_open import smart_open

        # unicode error
        # self.model = gensim.models.Word2Vec.load_word2vec_format(path + prefix + '-syn0.bin', binary = True)
        
        self.model = gensim.models.Word2Vec.load_word2vec_format(path + prefix + '-syn0.bin', binary = True, unicode_errors = 'ignore')
        self.vocab_size, self.vector_size = self.model.syn0.shape
        syn1neg = np.zeros((self.vocab_size, self.vector_size), dtype = np.float32)

        binary_len = np.dtype(np.float32).itemsize * self.vector_size
        with smart_open(path + prefix + '-syn1neg.bin') as fin:
            for i in range(self.vocab_size):
                weights = np.fromstring(fin.read(binary_len), dtype=np.float32)
                syn1neg[i] = weights
        self.syn1neg = syn1neg

        super().__init__()

    def build_senlist(self, tokenized_question, window = WINDOW):
        q = tokenized_question
        temp = q.wlist[:]
        est_sen = []
        sen_len = len(q.wlist)
        for i in q.qidx:
            head = max(i - window, 0)
            tail = min(i + window, sen_len)
            est_sen.append(q.wlist[head : i] + q.wlist[i + 1 : tail])
        return est_sen

    def ngram_option(self, options, uni_gram = False):
        ret = []
        if uni_gram:
            for opt in options:
                ret.append(re.sub(r'\s+',' ', re.sub(r'([\u4E00-\u9FFF])', r' \1 ', opt)).strip().split(' ')) 
        else:
            for opt in options:
                ret.append([opt]) 
        return ret

    def solve(self, tokenized_question, uni_gram = False, key = None):
        q = tokenized_question
        num_opt = len(q.options)

        if not key:
            w2v_senlist = self.build_senlist(q)
            gram_option = self.ngram_option(q.options, uni_gram)
            num_sen = float(len(w2v_senlist))
            if num_sen > 0.:
                option_vec_idx = []
                for opt in gram_option:
                    li = []
                    for w in opt: 
                        if w in self.model: li.append(self.model.vocab[w].index)
                        else: li.append(-1)
                    option_vec_idx.append(li)
                score = [0.] * num_opt

                
                for wlist in w2v_senlist:
                    arr = np.zeros(self.vector_size)
                    for w in wlist:
                        if w in self.model and w != u'*': arr += self.model[w]

                    for i, opt_i in enumerate(option_vec_idx):
                        s, k = 0.0, 0.0
                        for j, w in enumerate(opt_i):
                            if w >= 0:
                                temp = arr.copy()
                                others = [x for _i, x in enumerate(opt_i) if _i != j]
                                for v in others:
                                    if v >= 0: 
                                        temp += self.model.syn0[v]
                                s += np.dot(normalize_vec(temp), normalize_vec(self.syn1neg[w]))
                                k += 1

                        score[i] += float(s) / float(max(k, 1))

                score = [s/num_sen for s in score]                
                self.options_prob = {x: y for x, y in zip(string.ascii_lowercase[:num_opt], score)}
                self.prediction = string.ascii_lowercase[score.index(max(score))]
            else:
                self.prediction = Solver.UNSOLVED


        else:
            if key in self.model:
                arr = np.zeros(num_opt)
                hidd_vec = normalize_vec(self.model[key])
                for i in range(num_opt):
                    if q.options[i] in self.model:
                        w_idx = self.model.vocab[q.options[i]].index
                        arr[i] = np.dot(hidd_vec, (self.syn1neg[w_idx]))
                self.options_prob = {x: y for x, y in zip(string.ascii_lowercase[:num_opt], list(arr))}
                self.prediction = string.ascii_lowercase[arr.argmax()]
            else:
                self.prediction = Solver.UNKNOWN
        return self.prediction


class RetrievalSolver(Solver):

    def __init__(self, context_path, max_query_len = 10):
        self.context_path = context_path
        self.max_query_len = max_query_len
        super().__init__()

    def solve(self, question):
        self.prediction = Solver.UNSOLVED
        q = question
        num_opt = len(q.options)
        cleaned = re.sub(r'([\.\^\$\|\*\+\?\\\{\}\[\]\(\)])', r'\\\1', re.sub(r'"(.)', r'\1', q.context))
        seg_str = cleaned.split('︽⊙＿⊙︽')
        if len(seg_str) < 2: return self.prediction
        seg_len = [len(s) for s in seg_str]
        seg_num = len(seg_str)
        long_idx = seg_len.index(max(seg_len))

        long_str = seg_str[long_idx]

        if long_idx > 0:
            if seg_len[long_idx] > self.max_query_len: long_str = long_str[:self.max_query_len]
            pre_str = seg_str[long_idx - 1]
            if seg_len[long_idx - 1] > self.max_query_len: pre_str = pre_str[-self.max_query_len:]
            post_str = long_str
        else:
            if seg_len[long_idx] > self.max_query_len: long_str = long_str[-self.max_query_len:]
            post_str = seg_str[1]
            if seg_len[1] > self.max_query_len: post_str = post_str[:self.max_query_len]
            pre_str = long_str
            
            
        cmd = "grep -F '{long_str}' {file} | grep -Po '(?<={pre_str}).+(?={post_str})'".format(file = self.context_path, pre_str=pre_str, post_str=post_str, long_str=long_str)
        key = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode('utf-8').strip().split('\n')
        self.key = sorted(key, key=key.count,reverse=True)[0]
        self.prediction = Solver.UNKNOWN
        if(len(self.key) > 0):
            for i in range(num_opt):
                if q.options[i] in self.key or self.key in q.options[i]:
                    self.prediction = string.ascii_lowercase[i]
                    self.options_prob = {x: 0 for x in string.ascii_lowercase[:num_opt]}
                    self.options_prob[self.prediction] = 1
                    break


        return self.prediction



import html
from urllib.parse import quote
from urllib import request
class CrawlerSolver(Solver):
    def __init__(self, max_query_len = 10, query_num = 1):
        super().__init__()
        self.max_query_len = max_query_len
        self.query_num = query_num

    def crawl(self, query):
        raw_html = self.google_crawl(query).lower()
        clean_html = self.clean_html(raw_html)
        return clean_html

    def fast_search(self):
        html_pool = ''
        for q in self.rand_query:
            html_pool += self.crawl(q)
            time.sleep(0.5)
        # TODO 
        # for x in self.anslist:
        #     if html_pool.find(x) > 0:
        #         return (self.anslist.index(x), x)

    def google_crawl(self, query):
        link = "https://www.google.com/search?q=" + quote(query)  + '&ie=utf8&oe=utf8' # + '&lr=lang_zh-TW'
        req = request.Request(link, headers = {'User-Agent' : "Chrome Browser"})
        raw = html.unescape(request.urlopen(req).read().decode('utf-8'))
        return raw
    
    def clean_html(self, raw_html):
        clean_html = re.sub(re.compile(r'(<br?>)|(</br?>)|\n|\r|\s'), '', raw_html)
        return clean_html

    def solve(self, tokenized_question):
        self.set_rand_query(tokenized_question)
        pass
    
    def set_rand_query(self, tokenized_question):
        self.rand_query = []
        wlen = len(tokenized_question.wlist)
        wlist = tokenized_question.wlist
        qidx = tokenized_question.qidx
        for i in range(self.query_num):
            iq = random.choice(qidx)
            if iq > 0:
                ihead = random.choice(range(0, iq))
                if (iq - ihead) > self.max_query_len: 
                    ihead = iq - self.max_query_len
                    if ihead < 0: raise Exception('[CrawlerSolver:set_rand_query]index must be positive')
            else:
                ihead = 0
            
            if iq + 1 < wlen:
                iend = random.choice(range(iq + 1, wlen))
                if (iend - iq) > self.max_query_len:
                    iend = iq + self.max_query_len
                    if iend >= wlen: raise IndexError
            else:
                iend = wlen - 1
            self.rand_query.append(''.join(wlist[ihead : iend]))

    # def print_rand_query(self):
    #         for q in self.rand_query:
    #             print(q)
