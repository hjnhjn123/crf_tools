# coding=utf-8

from .np_tools import *
from fysom import Fysom
from collections import Counter


def seq_to_sen(seq):
    stop_set = set(u'。！？.!?')
    stop_set.update({u'……', u'…'})
    spliter_set = set(u'()（）【】“”')
    sen = []

    for item in seq:
        try:
            word, tag = item
            if word in spliter_set:
                continue
            sen.append((word, tag))
            if word in stop_set and tag == 'PU':
                yield sen
                sen = []
        except ValueError:
            print("ValueError", item)
    if sen:
        yield sen


def seq_to_events(seq):
    sens = list(seq_to_sen(seq))
    elements = []

    for sen in sens:
        seq = trans_tag(sen)
        fsm_1 = Seq_L1_FSM()
        fsm_1.read_seq(seq)
        elements.append(fsm_1.result)

    names = [[w for w, t in ele if t == 'NN'] for ele in elements]
    core_names = value_names(names)
    core_index = reverse_index(core_names)  # core_index: noun phrase -> name
    core_sent = {}  # {sen_num: [cores, ]}
    core_value_sent = Counter()
    for i, ns in enumerate(names):
        a = {core_index[n] for n in ns if n in core_index}  # 逐句计算句子中的cores
        if a:
            core_sent[i] = a  # 第i句中的cores的集合 （a）
            core_value_sent.update(a)
    sens_num = len(elements)
    core_value = {c: 1.0 * core_value_sent[c] * len(core_names[c]) / sens_num for c in core_value_sent}
    core_rank = sorted(core_value.items(), key=itemgetter(1), reverse=True)
    core3 = {c for c, i in filter(lambda item: len(item[0]) > 1, core_rank)[:3]}
    print("!!!" + " ".join(core3))
    result = []
    for ci in core_sent:
        cod = core3.intersection(core_sent[ci])
        if len(cod) <= 1: continue
        events = L2_elements_to_events(elements[ci], core3)

        if events:
            for e in events:
                result.append((sens[ci], e))
    print("result:", len(result))

    return result


def value_names(names):
    '''
    take noun phrase as super edge in supergraph, a keyword is a star core in a star.
    get as many as keywords.
    names ＝》 ［［ns, ns , ..],［］，］  ns => （noun, noun, ..) tuple represents a noun phrase.
    '''
    nouns = Counter()
    for ns in set(chain.from_iterable(names)):
        nouns.update(ns)
    cores = {a for a in nouns if nouns[a] > 1}
    # print "cores:", " ".join(c for c in cores)

    core_index = defaultdict(set)
    for ns in chain.from_iterable(names):
        a = cores.intersection(ns)
        for n in a:
            core_index[n].add(tuple(ns))
    # print "core_index:\n", "\n".join("==%s:%s"%(_n, ns_list_str(_s)) for _n, _s in core_index.iteritems())

    core_point = defaultdict(int)
    for k, ns_li in core_index.items():
        # core_point[k] += len(ns_li)
        for ns in ns_li:
            core_point[k] += len(ns) - 1  # most important approvement @March.
            a = cores.intersection(ns)
            if len(a) > 1: core_point[k] -= (len(a) - 1)
    # print "core_point:\n", "\n".join("---%s:%s"%(_k, _v) for _k, _v in core_point.iteritems())

    c_point = sorted(core_point.items(), key=itemgetter(1), reverse=False)
    used_ns = set()
    result = {}
    for n, point in c_point:
        if point <= 0: continue
        t = core_index[n].difference(used_ns)
        if t and len(t) > 1:
            result[n] = t
            used_ns.update(t)
    return result


def reverse_index(name_index):
    '''
    @param name_index: 关键词 －>  出现关键词的词组集合
    @return 词组－> 关键词
    '''
    index = defaultdict(lambda: None)
    for k, ns in name_index.items():
        for n in ns:
            index[n] = k
    return index


def L2_elements_to_events(seq, cores):
    # seq = parse_txt(sen)
    # print " ".join("%s:%s"%(w,tag) for w,tag in seq)

    # seq = trans_tag(seq)
    #
    # fsm = Seq_L1_FSM()
    # fsm.read_seq(seq)
    # fsm.print_result()

    fsm2 = Seq_L2_FSM()
    fsm2.read_seq(seq)
    # fsm2.print_result()

    fsm3 = Seq_L3_FSM(cores)
    fsm3.read_seq(fsm2.result)
    # fsm3.print_result()

    return fsm3.result


class Ab_Seq_FSM(object):
    def __init__(self, fsm_set):
        self.sett = fsm_set
        self.fsm = Fysom(fsm_set)
        self.result = []
        self.action_set = None

    def read_seq(self, seq):
        for word, tag in seq:
            if self.action_set and tag in self.action_set:
                e = [a for a in self.sett['events'] if a['name'] == tag][0]
                if self.fsm.current in set(e['src']):
                    self.fsm.trigger(tag, msg=word, tag=tag)
                    continue
            self.fsm.other()
            self.result.append((self.get_word_tuple(word), tag))

    def get_word_tuple(self, word):
        # if type(word) == unicode or type(word) == str:
        if type(word) == str:
            return (word,)
        # if type(word) == tuple and (type(word[0]) == unicode or type(word[0]) == str):
        if type(word) == tuple and type(word[0]) == str:
            return word
        print("WORD TUPLE ERROR:", word)
        return None

    def print_word(self, word):
        # if type(word) == unicode or type(word) == str: print
        if type(word) == str:
            print(word)
        elif type(word) == tuple and type(word[0]) == str:
            # elif type(word) == tuple and (type(word[0]) == unicode or type(word[0]) == str):
            print(" ".join(word))
        else:
            print("WORD TUPLE ERROR:", word)


def print_result(self):
    ss = ""
    for r, t in self.result:
        rr = ' '.join(self.get_word_tuple(r))
        ss += "%s:%s  " % (rr, t)
    print(ss)


class Seq_L1_FSM(Ab_Seq_FSM):
    def __init__(self):
        super(Seq_L1_FSM, self).__init__({
            'initial': 'INIT',
            'events': [
                {'name': 'noun', 'src': ['INIT', 'N', 'V'], 'dst': 'N'},
                {'name': 'verb', 'src': ['INIT', 'V', 'N'], 'dst': 'V'},
                {'name': 'other', 'src': ['INIT', 'N', 'V'], 'dst': 'INIT'}
            ],
        })
        self.fsm.onnoun = self.on_noun
        self.fsm.onleaveN = self.leave_noun
        self.fsm.onverb = self.on_verb
        self.fsm.onleaveV = self.leave_verb
        self.fsm.onother = self.on_other
        self.noun = self.verb = ()
        self.action_set = {'noun', 'verb'}

    def on_noun(self, e):
        word = e.msg
        self.noun += self.get_word_tuple(word)

    def on_verb(self, e):
        word = e.msg
        self.verb += self.get_word_tuple(word)

    def leave_noun(self, e):
        self.result.append((self.noun, 'NN'))
        self.noun = ()

    def leave_verb(self, e):
        self.result.append((self.verb, 'VV'))
        self.verb = ()

    def on_other(self, e):
        self.noun = self.verb = ()


class Seq_L2_FSM(Ab_Seq_FSM):
    '''for Noun Phrase.
     Init -> NN
     Init -> PN -> NN
     Init -> M -> NN
     Init -> CD -> M -> NN
     Init -> DT -> M -> NN
    '''

    def __init__(self):
        super(Seq_L2_FSM, self).__init__({
            'initial': 'INIT',
            'events': [
                {'name': 'NN', 'src': ['INIT', '_M_', '_PN_', '_DT_', '_CD_', 'VP'], 'dst': 'NP'},
                {'name': 'CD', 'src': ['INIT', '_DT_', 'VP'], 'dst': '_CD_'},
                {'name': 'DT', 'src': ['INIT', 'VP'], 'dst': '_DT_'},
                {'name': 'PN', 'src': ['INIT', 'VP'], 'dst': '_PN_'},
                {'name': 'M', 'src': ['INIT', '_CD_', '_DT_'], 'dst': '_M_'},
                {'name': 'VV', 'src': ['INIT', 'NP'], 'dst': 'VP'},
                {'name': 'AS', 'src': ['VP'], 'dst': 'VP'},
                {'name': 'other', 'src': ['INIT', '_M_', '_PN_', '_CD_', '_DT_', 'NP', 'VP'], 'dst': 'INIT'}
            ]
        })
        self.fsm.onNN = self.add_NP
        self.fsm.onCD = self.add_NP
        self.fsm.onDT = self.add_NP
        self.fsm.onPN = self.add_NP
        self.fsm.onM = self.add_NP
        self.fsm.onleaveNP = self.leave_NP
        self.fsm.onleaveVP = self.leave_VP
        self.fsm.onINIT = self.on_init
        self.fsm.onVV = self.add_VP
        self.fsm.onAS = self.add_VP
        self.NP = []
        self.VP = []
        self.action_set = {'NN', 'CD', 'DT', 'PN', 'M', 'VV', 'AS'}

    def add_NP(self, e):
        self.NP.append((self.get_word_tuple(e.msg), e.tag))

    def add_VP(self, e):
        self.VP.append((self.get_word_tuple(e.msg), e.tag))

    def leave_NP(self, e):
        r = ()
        for n, t in self.NP:
            r += n

        self.result.append((r, 'NP'))
        self.NP = []

    def leave_VP(self, e):
        r = ()
        for n, t in self.VP:
            r += n

        self.result.append((r, 'VP'))
        self.VP = []

    def on_init(self, e):
        if self.NP:
            self.result.extend(self.NP)
            self.NP = []
        if self.VP:
            self.result.extend(self.VP)
            self.VP = []


class Seq_L3_FSM(Ab_Seq_FSM):
    '''
    NP（Core）－ V － NP（Core）－ PU
    '''

    def __init__(self, cores):
        super(Seq_L3_FSM, self).__init__({
            'initial': 'INIT',
            'events': [
                {'name': "subject", 'src': ['INIT'], 'dst': 'SUB'},
                {'name': 'act', 'src': ['SUB'], 'dst': 'ACT'},
                {'name': 'object', 'src': ['ACT'], 'dst': 'OBJ'},
                {'name': 'event', 'src': ['OBJ'], 'dst': 'INIT'},
                {'name': 'reset', 'src': ['SUB', 'ACT'], 'dst': 'INIT'}
            ]
        })
        self.fsm.onsubject = self.add_event
        self.fsm.onact = self.add_event
        self.fsm.onobject = self.add_event
        self.fsm.onevent = self.on_event
        self.fsm.onreset = self.reset
        self.cores = cores
        self.event = ()

    def read_seq(self, seq):
        for word, tag in seq:
            if set(word).intersection(self.cores) and tag == "NP" and self.fsm.current == 'INIT':
                self.fsm.subject(msg=word, tag=tag)
            elif tag == "VP" and self.fsm.current == "SUB":
                self.fsm.act(msg=word, tag=tag)
            elif set(word).intersection(self.cores) and tag == "NP" and self.fsm.current == 'ACT':
                self.fsm.object(msg=word, tag=tag)
            elif tag == "PU":
                if self.fsm.current == 'OBJ':
                    self.fsm.event()
                elif self.fsm.current in {"SUB", "ACT"}:
                    self.fsm.reset()
            elif self.fsm.current in {'SUB', 'ACT', 'OBJ'}:
                self.event += self.get_word_tuple(word)

    def add_event(self, e):
        word = e.msg
        self.event += self.get_word_tuple(word)

    def on_event(self, e):
        self.result.append(self.event)
        self.event = ()

    def reset(self, e):
        self.event = ()

    def print_result(self):
        for event in self.result:
            print(" ".join(event))


L1_trans_dict = {'NT': 'noun', 'NR': 'noun', 'NN': 'noun', 'VA': 'verb', 'VC': 'verb', 'VE': 'verb', 'VV': 'verb'}


def trans_tag(seq, tag_dic=L1_trans_dict):
    for item in seq:
        try:
            word, tag = item
            if tag in tag_dic:
                yield word, tag_dic[tag]
            else:
                yield word, tag
        except ValueError:
            print("trans_tag", item)


def read_tagged_txt(txt):
    def split_w(wt):
        a = wt.rindex('#')
        return (wt[:a], wt[(a + 1):])

    with open(txt, 'r') as f:
        for line in f.readlines():
            yield [split_w(wt) for wt in line.split(u' ')]
            # yield [tuple(wt.split(u'#')) for wt in line.split(u' ')]


if __name__ == '__main__':
    print('start', now_str())
    gc.disable()

    result_set = []
    n = 0
    for seq in read_tagged_txt('./qq.txt.tagged'):
        n += 1
        print(n)
        r = seq_to_events(seq)
        result_set.extend(r)

        if n % 100 == 0: gc.collect()
        # raw_input()

    dump_to(result_set, 'events_result.pkl')

    print('end', now_str())
