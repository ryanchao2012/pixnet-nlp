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
        self.build_senlist()


    def tokenize(self):
        if self.tokenizer == 'jieba':
            context = self.context.strip().lower().replace('︽⊙＿⊙︽', '龐燮傍謝')
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
            pass


    def build_senlist(self, window = WINDOW):
        temp = self.wlist[:]
        est_sen = []
        sen_len = len(self.wlist)
        for i in self.qidx:
            head = max(i - window, 0)
            tail = min(i + window, sen_len)
            est_sen.append(self.wlist[head : i] + self.wlist[i + 1 : tail])
        self.w2v_senlist = est_sen
        # TODO:



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



class Word2VecSolver(Solver):

    def __init__(self, path, prefix):
        import gensim
        from smart_open import smart_open

        self.model = gensim.models.Word2Vec.load_word2vec_format(path + prefix + '-syn0.bin', binary = True)
        self.vocab_size, self.vector_size = self.model.syn0.shape
        syn1neg = np.zeros((self.vocab_size, self.vector_size), dtype = np.float32)

        binary_len = np.dtype(np.float32).itemsize * self.vector_size
        with smart_open(path + prefix + '-syn1neg.bin') as fin:
            for i in range(self.vocab_size):
                weights = np.fromstring(fin.read(binary_len), dtype=np.float32)
                syn1neg[i] = weights
        self.syn1neg = syn1neg

        super().__init__()

    def solve(self, tokenized_question, key = None):
        q = tokenized_question
        num_opt = len(q.options)
        if not key:
            num_sen = float(len(q.w2v_senlist))
            if num_sen > 0.:
                option_vec_idx = []
                for w in q.options:
                    if w in self.model: option_vec_idx.append(self.model.vocab[w].index)
                    else: option_vec_idx.append(-1)
                score = [0.] * num_opt
                for wlist in q.w2v_senlist:
                    arr = np.zeros(self.vector_size)
                    for w in wlist:
                        if w in self.model and w != u'*': arr += self.model[w]

                    for i in range(num_opt):
                        if option_vec_idx[i] >= 0: 
                            score[i] += np.dot(normalize_vec(arr), normalize_vec(self.syn1neg[option_vec_idx[i]]))
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

