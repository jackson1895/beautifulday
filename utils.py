
from torch.autograd import Variable
import numpy as np
import torch
import logging
import editdistance
from jiwer import wer

def sort_batch(data, label, length):
    batch_size = data.size(0)
    # 先将数据转化为numpy()，再得到排序的index
    inx = torch.from_numpy(np.argsort(length.numpy())[::-1].copy())
    data = data[inx]
    label = label[inx]
    length = length[inx]
    # length转化为了list格式，不再使用torch.Tensor格式
    length = list(length.numpy())
    return (data, label, length)
def loadData(v, data):
    v.resize_(data.size()).copy_(data)
    #print(v.size())
def reverse_padded_sequence(inputs, lengths, batch_first=True):
    '''这个函数输入是Variable，在Pytorch0.4.0中取消了Variable，输入tensor即可
    '''
    """Reverses sequences according to their lengths.
    Inputs should have size ``T x B x *`` if ``batch_first`` is False, or
    ``B x T x *`` if True. T is the length of the longest sequence (or larger),
    B is the batch size, and * is any number of dimensions (including 0).
    Arguments:
        inputs (Variable): padded batch of variable length sequences.
        lengths (list[int]): list of sequence lengths
        batch_first (bool, optional): if True, inputs should be B x T x *.
    Returns:
        A Variable with the same size as inputs, but with each sequence
        reversed according to its length.
    """
    if batch_first:
        inputs = inputs.transpose(0, 1)
    max_length, batch_size = inputs.size(0), inputs.size(1)
    if len(lengths) != batch_size:
        raise ValueError("inputs is incompatible with lengths.")
    ind = [list(reversed(range(0, length))) + list(range(length, max_length))
           for length in lengths]
    ind = torch.LongTensor(ind).transpose(0, 1)
    for dim in range(2, inputs.dim()):
        ind = ind.unsqueeze(dim)
    ind = Variable(ind.expand_as(inputs))
    if inputs.is_cuda:
        ind = ind.cuda(inputs.get_device())
    reversed_inputs = torch.gather(inputs, 0, ind)
    if batch_first:
        reversed_inputs = reversed_inputs.transpose(0, 1)
    return reversed_inputs

class averager(object):
    """Compute average for `torch.Variable` and `torch.Tensor`. """

    def __init__(self):
        self.reset()

    def add(self, v):
        if isinstance(v, Variable):
            count = v.data.numel()
            v = v.data.sum()
        elif isinstance(v, torch.Tensor):
            count = v.numel()
            v = v.sum()

        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res
def collate_fn(batch):
    # batch.sort(key=lambda x: len(x[0]), reverse=True)
    video_feature, label,mask,gt,video_id= zip(*batch)
    pad_label = []
    lens = []
    max_len = len(label[0])
    for i in range(len(label)):
        temp_label = [0] * max_len
        temp_label[:len(label[i])] = label[i]
        pad_label.append(temp_label)
        lens.append(len(label[i]))
    return img, pad_label, lens



class AttrDict(dict):
    """
    Dictionary whose keys can be accessed as attributes.
    """

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)

    def __getattr__(self, item):
        if item not in self:
            return None
        if type(self[item]) is dict:
            self[item] = AttrDict(self[item])
        return self[item]

    def __setattr__(self, item, value):
        self.__dict__[item] = value


def init_logger(log_file=None):
    log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]

    if log_file and log_file != '':
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)
    return logger


def computer_cer(preds, labels):
    dist = sum(editdistance.eval(label, pred) for label, pred in zip(labels, preds))
    total = sum(len(l) for l in labels)
    return dist, total

def computer_wer(preds,labels):
    assert len(preds)==len(labels)
    total_wer=0
    for label,pred in zip(labels,preds):
        t_label = ' '.join(str(l) for l in label)
        t_pred = ' '.join(str(p) for p in pred)
        t_wer = wer(t_label,t_pred)
        total_wer+=t_wer
    # tt = total_wer/len(preds)
    return total_wer/len(preds)
    # batch_wer = sum(wer(' '.join(str(l) for l in label),' '.join(str(p) for p in pred)) for label,pred in zip(labels,preds))
    # return batch_wer


def get_saved_folder_name(config):
    return '_'.join([config.data.name, config.training.save_model])


def count_parameters(model):
    n_params = sum([p.nelement() for p in model.parameters()])
    enc = 0
    dec = 0
    fir_enc=0
    for name, param in model.named_parameters():
        if 'encoder' in name:
            enc += param.nelement()
        elif 'decoder' in name:
            dec += param.nelement()
        elif 'fir_enc' in name:
            fir_enc +=param.nelement()
    return n_params, enc, dec,fir_enc


def init_parameters(model, type='xnormal'):
    for p in model.parameters():
        if p.dim() > 1:
            if type == 'xnoraml':
                torch.nn.init.xavier_normal_(p)
            elif type == 'uniform':
                torch.nn.init.uniform_(p, -0.1, 0.1)
        else:
            pass


def save_model(model, optimizer, config, save_name):
    multi_gpu = True if config.training.num_gpu > 1 else False
    if config.model.fir_enc_or_not:
        checkpoint = {
            'fir_enc': model.module.fir_enc.state_dict() if multi_gpu else model.fir_enc.state_dict(),
            'encoder': model.module.encoder.state_dict() if multi_gpu else model.encoder.state_dict(),
            'decoder': model.module.decoder.state_dict() if multi_gpu else model.decoder.state_dict(),
            'joint': model.module.joint.state_dict() if multi_gpu else model.joint.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': optimizer.current_epoch,
            'step': optimizer.global_step
        }
    else:
        checkpoint = {
            'encoder': model.module.encoder.state_dict() if multi_gpu else model.encoder.state_dict(),
            'decoder': model.module.decoder.state_dict() if multi_gpu else model.decoder.state_dict(),
            'joint': model.module.joint.state_dict() if multi_gpu else model.joint.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': optimizer.current_epoch,
            'step': optimizer.global_step
        }
    torch.save(checkpoint, save_name)


def save_ctc_model(model, optimizer, config, save_name):
    multi_gpu = True if config.training.num_gpu > 1 else False
    checkpoint = {
        'fir_enc':model.module.fir_enc.state_dict() if multi_gpu else model.fir_enc.state_dict(),
        'encoder': model.module.encoder.state_dict() if multi_gpu else model.encoder.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': optimizer.current_epoch,
        'step': optimizer.global_step
    }

    torch.save(checkpoint, save_name)


def save_language_model(model, optimizer, config, save_name):
    multi_gpu = True if config.training.num_gpu > 1 else False
    checkpoint = {
        'decoder': model.module.decoder.state_dict() if multi_gpu else model.decoder.state_dict(),
        'project_layer': model.module.project_layer.state_dict() if multi_gpu else model.project_layer.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': optimizer.current_epoch,
        'step': optimizer.global_step
    }

    torch.save(checkpoint, save_name)



