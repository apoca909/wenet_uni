import numpy as np
from collections import defaultdict
from wenet.utils.common import log_add, get_hex


class Vocab():
    def __init__(self, dict_path):
        self.token2id = {}
        self.id2token = {}
        self.blank_idx = 0
        self.unk_idx = 1
        self.sep_idx = 2
        self.bos_idx = 3
        self.eos_idx = 4
        self.eot_idx = 5 # end of token |
        self.load_file_dict(dict_path)

    def load_file_dict(self, dict_path):
        for _, line in enumerate(open(dict_path, 'r', encoding='utf-8')):
            token, i = line.strip().split()
            i = int(i)
            self.id2token[i] = token
            self.token2id[token] = i
        self.dict_size = len(self.token2id)

    def __len__(self,):
        return self.dict_size

    def str2ids(self, line):
        ids = [self.token2id.get(c, self.unk_idx) for c in line]
        return ids

    def ids2str(self, ids):
        line = [self.id2token.get(c) for c in ids]
        return line

    def get_token(self, idx):
        return self.id2token.get(idx)

    def get_index(self, token):
        return self.token2id.get(token, self.unk_idx)

class Node:
    "class representing nodes in a prefix tree"

    def __init__(self):
        self.children = {}  # all child elements beginning with current prefix
        self.isWord = False  # does this prefix represent a word

    def __str__(self):
        s = ''
        for k in self.children.keys():
            s += k
        return 'isWord: ' + str(self.isWord) + '; children: ' + s


class LexiconTrie:
    "prefix tree"

    def __init__(self, pvocab, plexicon):
        self.root = Node()
        self.addvocab(pvocab)
        self.addlexicon(plexicon)

    def addvocab(self, pvocab):
        self.vocab = Vocab(dict_path=pvocab)
        self.numUniqueWords = self.vocab.dict_size

    def addlexicon(self, plexicon):
        words = [line.split('\t')[0]+'|' for line in open(plexicon, 'r', encoding='utf-8')]
        self.addWords(words)

    def addWord(self, text):
        "add word to prefix tree"
        node = self.root
        for i in range(len(text)):
            c = text[i]  # current char
            if c not in node.children:
                node.children[c] = Node()
            node = node.children[c]
            isLast = (i + 1 == len(text))
            if isLast:
                node.isWord = True

    def addWords(self, words):
        for w in words:
            self.addWord(w)

    def getNode(self, text):
        "get node representing given text"
        node = self.root
        for c in text:
            if c in node.children:
                node = node.children[c]
            else:
                return None
        return node

    def isWord(self, text):
        node = self.getNode(text)
        if node:
            return node.isWord
        return False

    def getNextChars(self, text):
        "get all characters which may directly follow given text"
        chars = []
        node = self.getNode(text)
        if node:
            for k in node.children.keys():
                chars.append(k)
        return chars

    def getNext2Chars(self, text):
        first_chs = self.getNextChars(text)
        next2Chars = []
        for c1 in first_chs:
            if c1 == '|':
                nextch = [c1]
            else:
                new_text = text + c1
                nextch = [c1 + c2 for c2 in self.getNextChars(new_text)]

            next2Chars.extend(nextch)

        return next2Chars

    def getNextWords(self, text):
        "get all words of which given text is a prefix (including the text itself, it is a word)"
        words = []
        node = self.getNode(text)
        if node:
            nodes = [node]
            prefixes = [text]
            while len(nodes) > 0:
                # put all children into list
                for k, v in nodes[0].children.items():
                    nodes.append(v)
                    prefixes.append(prefixes[0] + k)
                # is current node a word
                if nodes[0].isWord:
                    words.append(prefixes[0])
                # remove current node
                del nodes[0]
                del prefixes[0]
        return words

    def dump(self):
        nodes = [self.root]
        while len(nodes) > 0:
            # put all children into list
            for v in nodes[0].children.values():
                nodes.append(v)
            # dump current node
            print(nodes[0])

            # remove from list
            del nodes[0]

def get_prefix_words(prefix_str, wordend='|'):
    _str = ''.join(prefix_str)
    words = [w  for w in _str.split(wordend) if w]

    return words

def beam_search(ctc_probs, beam_size):
    maxlen = ctc_probs.size(0)
    cur_hyps = [(tuple(), (0.0, -float('inf')))]
    # 2. CTC beam search step by step
    for t in range(0, maxlen):
        logp = ctc_probs[t]  # (vocab_size,)
        # key: prefix, value (pb, pnb), default value(-inf, -inf)
        next_hyps = defaultdict(lambda: (-float('inf'), -float('inf')))
        # 2.1 First beam prune: select topk best
        top_k_logp, top_k_index = logp.topk(beam_size)  # (beam_size,)
        for s in top_k_index:
            s = s.item()
            ps = logp[s].item()
            for prefix, (pb, pnb) in cur_hyps:
                last = prefix[-1] if len(prefix) > 0 else None
                if s == 0:  # blank
                    n_pb, n_pnb = next_hyps[prefix]
                    n_pb = log_add([n_pb, pb + ps, pnb + ps])
                    next_hyps[prefix] = (n_pb, n_pnb)
                elif s == last:
                    #  Update *ss -> *s;
                    n_pb, n_pnb = next_hyps[prefix]
                    n_pnb = log_add([n_pnb, pnb + ps])
                    next_hyps[prefix] = (n_pb, n_pnb)
                    # Update *s-s -> *ss, - is for blank
                    n_prefix = prefix + (s,)
                    n_pb, n_pnb = next_hyps[n_prefix]
                    n_pnb = log_add([n_pnb, pb + ps])
                    next_hyps[n_prefix] = (n_pb, n_pnb)
                else:
                    n_prefix = prefix + (s,)
                    n_pb, n_pnb = next_hyps[n_prefix]
                    n_pnb = log_add([n_pnb, pb + ps, pnb + ps])
                    next_hyps[n_prefix] = (n_pb, n_pnb)

        # 2.2 Second beam prune
        next_hyps = sorted(next_hyps.items(),
                           key=lambda x: log_add(list(x[1])),
                           reverse=True)
        cur_hyps = next_hyps[:beam_size]
    hyps = [(y[0], log_add([y[1][0], y[1][1]])) for y in cur_hyps]
    return hyps

def lexicon_beam_search(ctc_probs, beam_size, lexicon):
    maxlen = ctc_probs.shape[0]
    # cur_hyps: (prefix, (blank_ending_score, none_blank_ending_score))
    cur_hyps = [(tuple(), (0.0, -float('inf')))]
    # 2. CTC beam search step by step
    use_filter = False
    for t in range(0, maxlen):
        # key: prefix, value (pb, pnb), default value(-inf, -inf)
        next_hyps = defaultdict(lambda: (-float('inf'), -float('inf')))
        # 2.1 First beam prune: select topk best
        top_k_index = np.argsort(ctc_probs[t])[::-1][0:beam_size]  # (beam_size,)
        #_, top_k_index = ctc_probs[t].topk(beam_size)
        top_k_index = top_k_index.cpu().numpy()
        for i, item in enumerate(cur_hyps):
            prefix, (pb, pnb) = item
            last = prefix[-1] if len(prefix) > 0 else None
            if use_filter and lexicon is not None:
                prefix_str = lexicon.vocab.ids2str(prefix)
                words = get_prefix_words(prefix_str) ##get last unicode char
                if len(words) == 0:
                    words = [""]
                nextChars = lexicon.getNext2Chars(words[-1])
                nextCharsIds = lexicon.vocab.str2ids(nextChars)

                if last is not None:
                    nextCharsIds = nextCharsIds + [0, 5, last]
                else:
                    nextCharsIds = nextCharsIds + [0, 5]
                topk = list(set(top_k_index) & set(nextCharsIds))
            else:
                topk = top_k_index

            for s in topk:
                ps = ctc_probs[t, s].item()
                if s == 0:  # blank
                    n_pb, n_pnb = next_hyps[prefix]
                    n_pb = log_add([n_pb, pb + ps, pnb + ps])
                    n_prefix = prefix
                    #next_hyps[n_prefix] = (n_pb, n_pnb)
                elif s == last:
                    #  Update *ss -> *s;
                    n_pb, n_pnb = next_hyps[prefix]
                    n_pnb = log_add([n_pnb, pnb + ps])
                    next_hyps[prefix] = (n_pb, n_pnb)
                    # Update *s-s -> *ss, - is for blank
                    n_prefix = prefix + (s,)
                    n_pb, n_pnb = next_hyps[n_prefix]
                    n_pnb = log_add([n_pnb, pb + ps])
                    #next_hyps[n_prefix] = (n_pb, n_pnb)
                else:
                    n_prefix = prefix + (s,)
                    n_pb, n_pnb = next_hyps[n_prefix]
                    n_pnb = log_add([n_pnb, pb + ps, pnb + ps])
                #print(n_prefix)
                next_hyps[n_prefix] = (n_pb, n_pnb)
        # 2.2 Second beam prune
        next_hyps = sorted(next_hyps.items(), key=lambda x: log_add(list(x[1])), reverse=True)

        if lexicon is not None:
            next_hyps_checked = []
            for _, item in enumerate(next_hyps):
                prefix = item[0]
                if len(prefix) > 0 and prefix[-1] == lexicon.vocab.eot_idx:
                    prefix_str = lexicon.vocab.ids2str(prefix)  # get last word
                    words = get_prefix_words(prefix_str)


                    if len(words) > 0 and lexicon.isWord(words[-1] + '|') is True:
                        next_hyps_checked.append(item)
                    else:
                        pass #drop
                else:
                    next_hyps_checked.append(item)
            next_hyps = next_hyps_checked

        cur_hyps = next_hyps[:beam_size]

    if lexicon is not None:
        next_hyps_checked = []
        for _, item in enumerate(cur_hyps):
            prefix = item[0]
            if len(prefix) > 0 and prefix[-1] == lexicon.vocab.eot_idx:
                prefix_str = lexicon.vocab.ids2str(prefix)  # get last word
                words = get_prefix_words(prefix_str)
                if len(words) > 0 and lexicon.isWord(words[-1] + '|') is True:
                    next_hyps_checked.append(item)
        cur_hyps = next_hyps_checked

    hyps = [(y[0], log_add([y[1][0], y[1][1]])) for y in cur_hyps]

    return hyps


if __name__ == '__main__':
    # lexicon = LexiconTrie( 'D:/workspace/wenet-main/res/unicode_vocab.txt',
    #                     'D:/workspace/wenet-main/res/lexicon.txt')
    #
    #
    # testMat = np.load('D:/workspace/wenet-main/a.npy')
    # testBW = 15
    # # print(testMat.shape)
    # res = lexicon_beam_search(testMat, testBW,  lexicon )
    # print(res)

    #print(bytes.fromhex('E5819CE6ADA2').decode('utf-8'))

    fw = open('D:/workspace/wenet-main/res/lexicon_zh.txt', 'w', encoding='utf-8')
    ss = set()
    for line in open('D:/workspace/wenet-main/res/lexicon.txt', 'r', encoding='utf-8'):
        lexicon, chars = line.strip().split('\t')

        lexicon = get_hex(lexicon, 'dec', cws_tag='|')
        chars = get_hex(chars, 'dec', cws_tag='|')
        if lexicon in ss:
            continue
        else:
            print(lexicon+'\t'+ ' '.join(chars).strip(), file=fw)

            ss.add(lexicon)

    for line in open('D:/workspace/wenet-main/res/lang_zh.txt', 'r', encoding='utf-8'):
        lexicon, _ = line.strip().split()
        chars = lexicon
        # lexicon = get_hex(lexicon, 'dec', cws_tag='|')
        # chars = get_hex(chars, 'dec', cws_tag='|')
        if lexicon in ss:
            continue
        else:
            print(lexicon + '\t' + ' '.join(chars).strip(), file=fw)

            ss.add(lexicon)

