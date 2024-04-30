# copied from https://github.com/epfLLM/Megatron-LLM/blob/main/megatron/tokenizer/tokenizer.py
# (only keeping _FalconTokenizer & _SentencePieceTokenizer)

# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Megatron tokenizers."""

from abc import ABC, abstractmethod


def build_tokenizer(args):
    """Initialize tokenizer."""
    if args.rank == 0:
        print("> building {} tokenizer ...".format(args.tokenizer_type), flush=True)

    if args.tokenizer_type not in {"SentencePieceTokenizer", "FalconTokenizer"}:
        assert args.vocab_file is not None

    # Select and instantiate the tokenizer.
    if args.tokenizer_type == "SentencePieceTokenizer":
        assert args.vocab_file is not None
        tokenizer = _SentencePieceTokenizer(
            args.vocab_file,
            vocab_extra_ids=args.vocab_extra_ids,
            vocab_extra_ids_list=args.vocab_extra_ids_list,
            new_tokens=args.new_tokens,
        )
    elif args.tokenizer_type == "FalconTokenizer":
        tokenizer = _FalconTokenizer(vocab_extra_ids_list=args.vocab_extra_ids_list, new_tokens=args.new_tokens)
    else:
        raise NotImplementedError("{} tokenizer is not " "implemented.".format(args.tokenizer_type))

    # Add vocab size.
    args.padded_vocab_size = _vocab_size_with_padding(tokenizer.vocab_size, args)

    return tokenizer


def _vocab_size_with_padding(orig_vocab_size, args):
    """Pad vocab size so it is divisible by model parallel size and
    still having GPU friendly size."""

    after = orig_vocab_size
    multiple = args.make_vocab_size_divisible_by * args.tensor_model_parallel_size
    while (after % multiple) != 0:
        after += 1
    if args.rank == 0:
        print(
            " > padded vocab (size: {}) with {} dummy tokens "
            "(new size: {})".format(orig_vocab_size, after - orig_vocab_size, after),
            flush=True,
        )
    return after


class AbstractTokenizer(ABC):
    """Abstract class for tokenizer."""

    def __init__(self, name):
        self.name = name
        super().__init__()

    @property
    @abstractmethod
    def vocab_size(self):
        pass

    @property
    @abstractmethod
    def vocab(self):
        """Dictionary from vocab text token to id token."""
        pass

    @property
    @abstractmethod
    def inv_vocab(self):
        """Dictionary from vocab id token to text token."""
        pass

    @abstractmethod
    def tokenize(self, text):
        pass

    def detokenize(self, token_ids):
        raise NotImplementedError("detokenizer is not implemented for {} " "tokenizer".format(self.name))

    @property
    def cls(self):
        raise NotImplementedError("CLS is not provided for {} " "tokenizer".format(self.name))

    @property
    def sep(self):
        raise NotImplementedError("SEP is not provided for {} " "tokenizer".format(self.name))

    @property
    def pad(self):
        raise NotImplementedError("PAD is not provided for {} " "tokenizer".format(self.name))

    @property
    def eod(self):
        raise NotImplementedError("EOD is not provided for {} " "tokenizer".format(self.name))

    @property
    def mask(self):
        raise NotImplementedError("MASK is not provided for {} " "tokenizer".format(self.name))


class _FalconTokenizer(AbstractTokenizer):
    """Wrapper of huggingface tokenizer."""

    def __init__(self, vocab_extra_ids_list=None, new_tokens=True):
        name = "FalconTokenizer"
        super().__init__(name)
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-40b")

        if vocab_extra_ids_list and new_tokens:
            special_tokens = self.tokenizer.additional_special_tokens + vocab_extra_ids_list.split(",")
            self.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
            self._special_tokens = {tok: self.vocab[tok] for tok in special_tokens}
        else:
            self._special_tokens = {}

        self._inv_vocab = {idx: token for token, idx in self.tokenizer.vocab.items()}

    @property
    def vocab_size(self):
        return len(self.tokenizer.vocab)

    @property
    def vocab(self):
        return self.tokenizer.vocab

    def tokenize(self, text):
        return self.tokenizer.encode(text)

    def detokenize(self, token_ids):
        return self.tokenizer.decode(token_ids)

    @property
    def inv_vocab(self):
        return self._inv_vocab

    @property
    def eod(self):
        return self.eos_token_id

    @property
    def pad(self):
        return self.eos_token_id

    @property
    def eos_token_id(self):
        return self.tokenizer.eos_token_id


class _SentencePieceTokenizer(AbstractTokenizer):
    """SentencePieceTokenizer-Megatron wrapper"""

    def __init__(self, model_file, vocab_extra_ids=0, vocab_extra_ids_list=None, new_tokens=True):
        name = "SentencePieceTokenizer"
        super().__init__(name)

        import sentencepiece

        self._tokenizer = sentencepiece.SentencePieceProcessor(model_file=model_file)

        self._initalize(vocab_extra_ids, vocab_extra_ids_list, new_tokens)

    def _initalize(self, vocab_extra_ids, vocab_extra_ids_list, new_tokens):
        self._vocab = {}
        self._inv_vocab = {}

        self._special_tokens = {}
        self._inv_special_tokens = {}

        self._t5_tokens = []

        for i in range(len(self._tokenizer)):
            t = self._tokenizer.id_to_piece(i)
            self._inv_vocab[i] = t
            self._vocab[t] = i

        def _add_special_token(t):
            if t not in self.vocab and not new_tokens:
                return
            if t not in self._vocab:
                next_id = len(self._vocab)
                self._vocab[t] = next_id
                self._inv_vocab[next_id] = t
            self._special_tokens[t] = self._vocab[t]
            self._inv_special_tokens[self._vocab[t]] = t

        _add_special_token("<CLS>")
        self._cls_id = self._vocab.get("<CLS>")
        _add_special_token("<SEP>")
        self._sep_id = self._vocab.get("<SEP>")
        _add_special_token("<EOD>")
        self._eod_id = self._vocab.get("<EOD>")
        _add_special_token("<MASK>")
        self._mask_id = self._vocab.get("<MASK>")

        pad_id = self._tokenizer.pad_id()
        try:
            pad_token = self._tokenizer.id_to_piece(pad_id)
        except IndexError:
            pad_token = "<PAD>"
        _add_special_token(pad_token)
        self._pad_id = self._vocab.get(pad_token)

        bos_id = self._tokenizer.bos_id()
        try:
            bos_token = self._tokenizer.id_to_piece(bos_id)
        except IndexError:
            bos_token = "<BOS>"
        _add_special_token(bos_token)
        self._bos_id = self._vocab.get(bos_token)

        eos_id = self._tokenizer.eos_id()
        try:
            eos_token = self._tokenizer.id_to_piece(eos_id)
        except IndexError:
            eos_token = "<EOS>"
        _add_special_token(eos_token)
        self._eos_id = self._vocab.get(eos_token)

        for i in range(vocab_extra_ids):
            t = "<extra_id_{}>".format(i)
            _add_special_token(t)
            self._t5_tokens += [t]
        if vocab_extra_ids_list:
            for t in vocab_extra_ids_list.split(","):
                _add_special_token(t)
        print("Special tokens: {}".format(self._special_tokens))

    @property
    def vocab_size(self):
        return len(self._vocab)

    @property
    def vocab(self):
        return self._vocab

    @property
    def inv_vocab(self):
        return self._inv_vocab

    # From:
    # https://github.com/NVIDIA/NeMo/blob/c8fa217e811d60d11d014827c7f3845ff6c99ae7/nemo/collections/common/tokenizers/sentencepiece_tokenizer.py#L89
    def tokenize(self, text):
        ids = []
        idx = 0

        while 1:
            indices = {}
            for token in self._special_tokens:
                try:
                    indices[token] = text[idx:].index(token)
                except ValueError:
                    continue
            if len(indices) == 0:
                break

            next_token = min(indices, key=indices.get)
            next_idx = idx + indices[next_token]

            ids.extend(self._tokenizer.encode_as_ids(text[idx:next_idx]))
            ids.append(self._special_tokens[next_token])
            idx = next_idx + len(next_token)

        ids.extend(self._tokenizer.encode_as_ids(text[idx:]))
        return ids

    # From:
    # https://github.com/NVIDIA/NeMo/blob/c8fa217e811d60d11d014827c7f3845ff6c99ae7/nemo/collections/common/tokenizers/sentencepiece_tokenizer.py#L125
    def detokenize(self, ids):
        text = ""
        last_i = 0

        for i, id in enumerate(ids):
            if id in self._inv_special_tokens:
                text += self._tokenizer.decode_ids(ids[last_i:i]) + " "
                text += self._inv_special_tokens[id] + " "
                last_i = i + 1
        text += self._tokenizer.decode_ids(ids[last_i:])
        return text.strip()

    @property
    def cls(self):
        return self._cls_id

    @property
    def sep(self):
        return self._sep_id

    @property
    def pad(self):
        return self._pad_id

    @property
    def bos_token_id(self):
        return self._bos_id

    @property
    def bos(self):
        return self._bos_id

    @property
    def eod(self):
        if self._eod_id is not None:
            return self._eod_id
        return self._eos_id  # in case no eod we can patch this up with an eos

    @property
    def eos_token_id(self):
        if self._eod_id is not None:
            return self._eod_id
        return self._eos_id

    @property
    def eos(self):
        return self._eos_id

    @property
    def mask(self):
        return self._mask_id

    @property
    def additional_special_tokens_ids(self):
        return [self.vocab[k] for k in self._t5_tokens]
