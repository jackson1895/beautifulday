import json
import os
import argparse
import numpy as np
import random
import misc.utils as utils
from utils import loadData,averager
import opts_only
import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_value_
from dataloader_only import VideoDataset
from misc.rewards import get_self_critical_reward, init_cider_scorer
from models import DecoderRNN, EncoderRNN, S2VTAttModel, S2VTModel,CTCmodel,Two_Lstm,CTC_Hieratical_LSTM,two_lstm
from torch import nn
from torch.utils.data import DataLoader
# from warpctc_pytorch import  CTCLoss
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from utils import collate_fn
from torch.nn.utils import rnn
import shutil
from torch.nn.functional import log_softmax
from strLabelConverter import strLabelConverter
from jiwer import wer
import time
from utils import AttrDict, init_logger, count_parameters, save_model,computer_cer,save_ctc_model
import yaml
from optim import Optimizer
from dataset import SignVideoDataset
from models.model import Transducer
from models.VideoModel import VideoModel


def train(loader, model, crit, optimizer, lr_scheduler, opt, rl_crit=None,converter=None):

    model.cuda()
    # crit.cuda()
    # optimizer.cuda()
    # lr_scheduler.cuda()
    # video = torch.FloatTensor(params.batchSize, 3, params.imgH, params.imgH)
    #TODO 原本中国手语是30
    text = torch.LongTensor(opt['batch_size'] * opt['max_len'])
    # text = torch.IntTensor(opt['batch_size'] * 30)
    length = torch.LongTensor(opt['batch_size'])
    converter = strLabelConverter(loader.dataset)
    # model = nn.DataParallel(model)
    writer = SummaryWriter("two_lstm_exp_German")
    loss_avg = averager()
    wer_val = 1.0
    for epoch in range(opt["epochs"]):
        n_correct = 0
        model.train()
        if opt['lr_schluder'] == 'StepLR':
            lr_scheduler.step()
        elif opt['lr_schluder'] == 'ReduceLROnPlateau':
            lr_scheduler.step(wer_val)
        iteration = 0
        f_wer=0.0
        for data in loader:
            torch.cuda.synchronize()
            for p in model.parameters():
                p.requires_grad = True
            fc_feats = data['fc_feats'].cuda() # (batch_size, 80, 512)

            # 1. slice 10 * (batch_size, 8, 512)
            # 2. send each slice to LSTM 10 * (batch_size, 1024)
            # 3. set another mask M2(batch_size, 10)
            # 4. if a slice is full of Zero, set the corresponding index of M2 zero
            # 5. LSTM2
            # 6. obtain final result bt *

            labels = data['labels'].cuda()
            # masks = data['masks'].cuda()
            # clip_nums = data['clip_num']
            # sorted_clip_nums,indices = torch.sort(clip_nums,descending=True)
            # _, desorted_indices = torch.sort(indices, descending=False)
            # fc_feats=fc_feats[indices]
            # pack = rnn.pack_padded_sequence(fc_feats,sorted_clip_nums,batch_first=True)
            #TODO
            optimizer.zero_grad()
            output = model(fc_feats)
            # desorted_res = output[desorted_indices]


            output=output.log_softmax(2).requires_grad_()
            _, preds = output.max(2)
            output = output.transpose(0, 1).contiguous()
            labels_ctc = []
            ys=[]
            for i in labels:
                for j in i:
                    if not j==-1:
                        labels_ctc.append(j)
            for i in labels:
                non_zero = (i == -1).nonzero()
                if not non_zero.numel():
                    ys.append(opt['max_len'])
                else:
                    ys.append(non_zero[0][0])
            loadData(text,torch.LongTensor(labels_ctc))
            loadData(length,torch.LongTensor(ys))
            preds_size = Variable(torch.LongTensor([output.size(0)] * output.size(1)))

            loss = crit(output, text.cuda(), preds_size.cuda(), length.cuda())
            # loss= crit(output,text,preds_size,length)/opt['batch_size']
            preds = preds.contiguous().view(-1)
            sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
            list_1 = []

            for pred, target in zip(sim_preds, labels):
                ts = target.squeeze().cpu().numpy().tolist()
                res = []
                for i in ts :
                    if i == -1:
                        continue
                    res.append(loader.dataset.ix_to_word[str(i)])
                target = ' '.join(res)
                tmp_wer = wer(target,pred)
                f_wer += tmp_wer

                if pred == target:
                    n_correct += 1
            loss_avg.add(loss)
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
            iteration += 1
        acc=n_correct/float(len(loader))
        # print(len(loader)*opt['batch_size'])
        f_wer = f_wer/float(len(loader)*opt['batch_size'])
        print("[epoch %d]->train_loss = %.6f , wer = %.6f" % (epoch, loss_avg.val(),f_wer))

        if epoch % opt["eval_every"] == 0:
            for p in model.parameters():
                p.requires_grad = False

            loss_eval,wer_val=val(model,crit,opt,writer,epoch)
            writer.add_scalars('loss_epcho', {'train_loss':loss_avg.val(),'val_loss':loss_eval},epoch)
            writer.add_scalars('wer_epcho',{'train_wer':f_wer,'eval_wer':wer_val},epoch)

        if epoch % opt["save_checkpoint_every"] == 0:
            path = opt['root_model_path']
            # if not os.path.exists(path):
            #     os.mkdir(path)
            # else:
            #     shutil.rmtree(path)
            #     os.mkdir(path)
            model_path = os.path.join(path,
                                      'model_%d.pth' % (epoch))
            model_info_path = os.path.join(path,
                                           'model_score.txt')
            torch.save(model.state_dict(), model_path)
            print("model saved to %s" % (model_path))
            with open(model_info_path, 'a') as f:
                f.write("model_%d, loss: %.6f  train wer: %.6f val wer: %.6f\n" % (epoch, loss_avg.val(),f_wer,wer_val))
        loss_avg.reset()

def val(model, crit, opt,writer=None,epoch=0):
    dataset = VideoDataset(opt,'test')
    dataloader = DataLoader(dataset,batch_size=opt['batch_size'],shuffle=True)
    opt["vocab_size"] = dataset.get_vocab_size()
    model.eval()
    # TODO 原本中国手语是30
    text = torch.LongTensor(opt['batch_size'] * opt['max_len'])
    # text = torch.IntTensor(opt['batch_size'] * 30)
    length = torch.LongTensor(opt['batch_size'])
    loss_avg=averager()
    n_correct=0
    f_wer= 0.0
    # converter = strLabelConverter(dataset)
    converter = strLabelConverter(dataloader.dataset)
    for data in dataloader:
        fc_feats = data['fc_feats'].cuda()
        labels = data['labels'].cuda()
        # masks = data['masks'].cuda()
        # clip_nums = data['clip_num']
        # sorted_clip_nums, indices = torch.sort(clip_nums, descending=True)
        # _, desorted_indices = torch.sort(indices, descending=False)
        # fc_feats = fc_feats[indices]
        # pack = rnn.pack_padded_sequence(fc_feats, sorted_clip_nums, batch_first=True)
        with torch.no_grad():
            output = model(fc_feats)

        # desorted_res = output[desorted_indices]

        output = output.log_softmax(2).requires_grad_()
        _, preds = output.max(2)
        output = output.transpose(0, 1).contiguous()
        labels_ctc = []
        ys = []
        for i in labels:
            for j in i:
                if not j == -1:
                    labels_ctc.append(j)
        for i in labels:
            non_zero = (i == -1).nonzero()
            if not non_zero.numel():
                ys.append(opt['max_len'])
            else:
                ys.append(non_zero[0][0])
        loadData(text, torch.LongTensor(labels_ctc))
        loadData(length, torch.LongTensor(ys))
        preds_size = Variable(torch.LongTensor([output.size(0)] * output.size(1)))
        loss = crit(output.cuda(), text.cuda(), preds_size.cuda(), length.cuda())

        preds = preds.contiguous().view(-1)
        sim_preds =converter.decode(preds.data,preds_size.data,raw=False)
        for pred, target in zip(sim_preds, labels):
            ts = target.squeeze().cpu().numpy().tolist()
            res = []
            for i in ts:
                if i == -1:
                    continue
                res.append(dataloader.dataset.ix_to_word[str(i)])
            target = ' '.join(res)
            tmp_wer = wer(target, pred)
            f_wer += tmp_wer
            if pred == target:
                n_correct += 1
        loss_avg.add(loss)
    acc = n_correct/float(len(dataloader))
    f_wer = f_wer/float(len(dataloader)*opt['batch_size'])
    print("[epoch %d]->val_loss = %.6f , wer = %.6f" % (epoch, loss_avg.val(),f_wer))

    # writer.add_scalar('scalar/val_loss_epcho', loss_avg.val())
    return loss_avg.val(),f_wer



def train_rnnt(epoch, config, model, training_data, optimizer, logger, visualizer=None):

    model.train()
    start_epoch = time.process_time()
    total_loss = 0
    optimizer.epoch()
    batch_steps = len(training_data)

    for step, (inputs, inputs_length, targets, targets_length) in enumerate(training_data):

        if config.training.num_gpu > 0:
            inputs, inputs_length = inputs.cuda(), inputs_length.cuda()
            targets, targets_length = targets.cuda(), targets_length.cuda()

        max_inputs_length = inputs_length.max().item()
        max_targets_length = targets_length.max().item()
        inputs = inputs[:, :max_inputs_length, :]
        targets = targets[:, :max_targets_length]

        optimizer.zero_grad()
        start = time.process_time()
        loss = model(inputs, inputs_length, targets, targets_length)

        if config.training.num_gpu > 1:
            loss = torch.mean(loss)

        loss.backward()

        total_loss += loss.item()

        grad_norm = nn.utils.clip_grad_norm_(
            model.parameters(), config.training.max_grad_norm)

        optimizer.step()

        if visualizer is not None:
            visualizer.add_scalar(
                'train_loss', loss.item(), optimizer.global_step)
            visualizer.add_scalar(
                'learn_rate', optimizer.lr, optimizer.global_step)

        avg_loss = total_loss / (step + 1)
        if optimizer.global_step % config.training.show_interval == 0:
            end = time.process_time()
            process = step / batch_steps * 100
            logger.info('-Training-Epoch:%d(%.5f%%), Global Step:%d, Learning Rate:%.6f, Grad Norm:%.5f, Loss:%.5f, '
                        'AverageLoss: %.5f, Run Time:%.3f' % (epoch, process, optimizer.global_step, optimizer.lr,
                                                              grad_norm, loss.item(), avg_loss, end-start))

        # break
    end_epoch = time.process_time()
    logger.info('-Training-Epoch:%d, Average Loss: %.5f, Epoch Time: %.3f' %
                (epoch, total_loss / (step+1), end_epoch-start_epoch))


def eval_rnnt(epoch, config, model, validating_data, logger, visualizer=None):
    model.eval()
    total_loss = 0
    total_dist = 0
    total_word = 0
    batch_steps = len(validating_data)
    for step, (inputs, inputs_length, targets, targets_length) in enumerate(validating_data):

        if config.training.num_gpu > 0:
            inputs, inputs_length = inputs.cuda(), inputs_length.cuda()
            targets, targets_length = targets.cuda(), targets_length.cuda()

        max_inputs_length = inputs_length.max().item()
        max_targets_length = targets_length.max().item()
        inputs = inputs[:, :max_inputs_length, :]
        targets = targets[:, :max_targets_length]

        preds = model.recognize(inputs, inputs_length)

        transcripts = [targets.cpu().numpy()[i][:targets_length[i].item()]
                       for i in range(targets.size(0))]

        dist, num_words = computer_cer(preds, transcripts)
        total_dist += dist
        total_word += num_words

        cer = total_dist / total_word * 100
        if step % config.training.show_interval == 0:
            process = step / batch_steps * 100
            logger.info('-Validation-Epoch:%d(%.5f%%), CER: %.5f %%' % (epoch, process, cer))

    val_loss = total_loss/(step+1)
    logger.info('-Validation-Epoch:%4d, AverageLoss:%.5f, AverageCER: %.5f %%' %
                (epoch, val_loss, cer))

    if visualizer is not None:
        visualizer.add_scalar('cer', cer, epoch)

    return cer


def train_ctc_model(epcho,config,model,training_data,optimizer,logger,visualizer=None):
    model.train()
    start_epoch = time.process_time()
    total_loss = 0
    optimizer.epoch()
    batch_steps = len(training_data)

    for step, (inputs, inputs_length, targets, targets_length) in enumerate(training_data):

        if config.training.num_gpu > 0:
            inputs, inputs_length = inputs.cuda(), inputs_length.cuda()
            targets, targets_length = targets.cuda(), targets_length.cuda()

        max_inputs_length = inputs_length.max().item()
        max_targets_length = targets_length.max().item()
        inputs = inputs[:, :max_inputs_length, :]
        targets = targets[:, :max_targets_length]

        optimizer.zero_grad()
        start = time.process_time()
        loss = model(inputs, inputs_length, targets, targets_length)

        if config.training.num_gpu > 1:
            loss = torch.mean(loss)

        loss.backward()

        total_loss += loss.item()

        grad_norm = nn.utils.clip_grad_norm_(
            model.parameters(), config.training.max_grad_norm)

        optimizer.step()

        if visualizer is not None:
            visualizer.add_scalar(
                'train_loss', loss.item(), optimizer.global_step)
            visualizer.add_scalar(
                'learn_rate', optimizer.lr, optimizer.global_step)

        avg_loss = total_loss / (step + 1)
        if optimizer.global_step % config.training.show_interval == 0:
            end = time.process_time()
            process = step / batch_steps * 100
            logger.info('-Training-Epoch:%d(%.5f%%), Global Step:%d, Learning Rate:%.6f, Grad Norm:%.5f, Loss:%.5f, '
                        'AverageLoss: %.5f, Run Time:%.3f' % (epoch, process, optimizer.global_step, optimizer.lr,
                                                              grad_norm, loss.item(), avg_loss, end - start))

        # break
    end_epoch = time.process_time()
    logger.info('-Training-Epoch:%d, Average Loss: %.5f, Epoch Time: %.3f' %
                (epoch, total_loss / (step + 1), end_epoch - start_epoch))

def eval_ctc_model(epcho,config,model,validating_data,logger,visualizer=None):
    model.eval()
    total_loss = 0
    total_dist = 0
    total_word = 0
    batch_steps = len(validating_data)
    for step, (inputs, inputs_length, targets, targets_length) in enumerate(validating_data):

        if config.training.num_gpu > 0:
            inputs, inputs_length = inputs.cuda(), inputs_length.cuda()
            targets, targets_length = targets.cuda(), targets_length.cuda()

        max_inputs_length = inputs_length.max().item()
        max_targets_length = targets_length.max().item()
        inputs = inputs[:, :max_inputs_length, :]
        targets = targets[:, :max_targets_length]

        preds = model.recognize(inputs, inputs_length)

        transcripts = [targets.cpu().numpy()[i][:targets_length[i].item()]
                       for i in range(targets.size(0))]

        dist, num_words = computer_cer(preds, transcripts)
        total_dist += dist
        total_word += num_words

        cer = total_dist / total_word * 100
        if step % config.training.show_interval == 0:
            process = step / batch_steps * 100
            logger.info('-Validation-Epoch:%d(%.5f%%), CER: %.5f %%' % (epoch, process, cer))

    val_loss = total_loss / (step + 1)
    logger.info('-Validation-Epoch:%4d, AverageLoss:%.5f, AverageCER: %.5f %%' %
                (epoch, val_loss, cer))

    if visualizer is not None:
        visualizer.add_scalar('cer', cer, epoch)

    return cer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', type=str, default='config/ctc.yaml')
    parser.add_argument('-log', type=str, default='train.log')
    parser.add_argument('-mode', type=str, default='retrain')
    opt = parser.parse_args()
    configfile =open(opt.config)
    config = AttrDict(yaml.load(configfile, Loader=yaml.FullLoader))
    exp_name = os.path.join(config.data.name, config.data.exp_name, config.training.save_model)
    if not os.path.isdir(exp_name):
        os.makedirs(exp_name)
    logger = init_logger(os.path.join(exp_name, opt.log))
    shutil.copyfile(opt.config, os.path.join(exp_name, 'config.yaml'))
    logger.info(config)
    logger.info('Save config info.')
    os.environ['CUDA_VISIBLE_DEVICES'] = config.training.gpu
    if config.training.num_gpu > 0:
        torch.cuda.manual_seed(config.training.seed)
        torch.backends.cudnn.deterministic = True  # 保证实验结果的可重复性
    else:
        torch.manual_seed(config.training.seed)
    logger.info('Set random seed: %d' % config.training.seed)
    ##-              loading train/val dataset                     -##
    num_workers = config.training.num_gpu * 2
    train_dataset = SignVideoDataset(config.data,'train')
    train_data = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.data.batch_size * config.training.num_gpu,
        shuffle=config.data.shuffle, num_workers=num_workers)
    logger.info('Load Train Set!')
    val_dataset =SignVideoDataset(config.data,'val')
    val_data =torch.utils.data.DataLoader(
        val_dataset, batch_size=config.data.batch_size * config.training.num_gpu,
        shuffle=False, num_workers=num_workers)
    logger.info('Load Dev Set!')

    model = VideoModel(config.model)

    if config.training.num_gpu > 0:
        model = model.cuda()
        if config.training.num_gpu > 1:
            device_ids = list(range(config.training.num_gpu))
            model = torch.nn.DataParallel(model, device_ids=device_ids)
        logger.info('Loaded the model to %d GPUs' % config.training.num_gpu)

    n_params, enc, dec ,fir_enc= count_parameters(model)
    logger.info('# the number of parameters in the whole model: %d' % n_params)
    logger.info('# the number of parameters in the Encoder: %d' % enc)
    logger.info('# the number of parameters in the Decoder: %d' % (n_params-enc))
    optimizer = Optimizer(model.parameters(), config.optim)
    logger.info('Created a %s optimizer.' % config.optim.type)

    start_epoch = 0


    # create a visualizer
    if config.training.visualization:
        visualizer = SummaryWriter(os.path.join(exp_name, 'log'))
        logger.info('Created a visualizer.')
    else:
        visualizer = None

    for epoch in range(start_epoch, config.training.epochs):
        # eval_ctc_model(epoch, config, model, val_data, logger, visualizer)
        train_ctc_model(epoch, config, model, train_data,
              optimizer, logger, visualizer)

        if config.training.eval_or_not:
            _ = eval_ctc_model(epoch, config, model, val_data, logger, visualizer)

        save_name = os.path.join(exp_name, '%s.epoch%d.chkpt' % (config.training.save_model, epoch))
        save_ctc_model(model, optimizer, config, save_name)
        logger.info('Epoch %d model has been saved.' % epoch)
        if optimizer.lr < 1e-6:
            logger.info('The learning rate is too low to train.')
            break
        logger.info('Epoch %d update learning rate: %.6f' %(epoch, optimizer.lr))
    logger.info('The training process is OVER!')
    # main(opt)
