import torch
class strLabelConverter(object):
    """Convert between str and label.
    NOTE:
        Insert `blank` to the alphabet for CTC.
    Args:
        alphabet (str): set of the possible characters.
        ignore_case (bool, default=True): whether or not to ignore all of the case.
    """

    def __init__(self, dataset, ignore_case=False):
        self._ignore_case = ignore_case
        # if self._ignore_case:
        #     alphabet = alphabet.lower()
        self.dataset = list(dataset.word_to_ix.keys())

        self.dict = dataset.ix_to_word
        # for i, char in enumerate(alphabet):
        #     # NOTE: 0 is reserved for 'blank' required by wrap_ctc
        #     self.dict[char] = i + 1
        # print(self.dict)
        # print(self.dict)

    # print(self.dict)
    def encode(self, text):
        """Support batch or single str.
        Args:
            text (str or list of str): texts to convert.
        Returns:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        """
        # length = []
        # result = []
        # for t in text:
        #     list_1 = []
        #     list_1.append(t[2:-1])
        # for item in list_1:
        #     length.append(len(item))
        #     for char in item:
        #         char = char.casefold()
        #         index = self.dict[char]
        #         result.append(index)
        # text = result
        # print(list_1)
        # print(text[0])
        # print(text)
        # if isinstance(text, str):
        #     # for char in text:
        #     #     char = char.encode()
        #     #     print(char)
        #     #print(text[0])
        #     #print(text)

        #     text = [
        #         self.dict[char.lower() if self._ignore_case else char]
        #         for char in text
        #     ]

        #     length = [len(text)]
        # elif isinstance(text, collections.Iterable):
        #     length = [len(s) for s in text]
        #     text = ''.join(text)
        #     text, _ = self.encode(text)
        length = []
        result = []
        for item in text:
            item = item.decode('utf-8', 'strict')
            length.append(len(item))
            # print(len(item))
            # print('11')
            # print(item)
            for char in item:
                # print(char)
                # char = char.lower()
                # print(char)
                # try:
                #     index = self.dict[char]
                # except Exception as e:
                #     pass
                # else:
                #     result.append(index)
                # finally:
                #     #print(index)
                #     #result.append(index)
                #     pass
                index = self.dict[char]
                result.append(index)

        text = result
        # print(text,length)
        return (torch.IntTensor(text), torch.IntTensor(length))

    def decode(self, t, length, raw=False):
        """Decode encoded texts back into strs.
        Args:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        Raises:
            AssertionError: when the texts and its length does not match.
        Returns:
            text (str or list of str): texts to convert.
            numel 返回数组中元素的个数
        """
        if length.numel() == 1:
            length = length[0]
            assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(),
                                                                                                         length)
            if raw:
                return ' '.join([self.dict[i.item()] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.dict[str(t[i].item())])
                return ' '.join(char_list)
        else:
            # batch mode
            assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(
                t.numel(), length.sum())
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(
                    self.decode(
                        t[index:index + l], torch.IntTensor([l]), raw=raw))
                index += l
            return texts