

def get_2d_spans(text, tokenss):
    spanss = []
    cur_idx = 0
    for tokens in tokenss:
        spans = []
        for token in tokens:
            if text.find(token, cur_idx) < 0:
                print ('something wrong with processing span...')
                print ('{} {} {}'.format(token, cur_idx, text))
                raise Exception()
            cur_idx = text.find(token, cur_idx)
            spans.append((cur_idx, cur_idx + len(token)))
            cur_idx += len(token)
        spanss.append(spans)
    return spanss

def get_word_span(context, wordss, start, stop):
    spanss = get_2d_spans(context, wordss)
    idx = []
    for sent_idx, spans in enumerate(spanss):
        for word_idx, span in enumerate(spans):
            if stop > span[0] and start < span[1]:
                idx.append((sent_idx, word_idx))
    return idx[0], idx[-1]
