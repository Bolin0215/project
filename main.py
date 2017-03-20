from seq2seq import tagger
import os

os.environ['THEANO_FLAGS'] = "device=cpu"
if __name__ == "__main__":
    seq2seqTagger = tagger.Tagger()
    seq2seqTagger.main()
