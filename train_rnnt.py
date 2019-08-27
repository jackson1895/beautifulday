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
from utils import AttrDict, init_logger, count_parameters, save_model,computer_cer,save_ctc_model,computer_wer
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

def main(opt):
    dataset = VideoDataset(opt, 'train')

    dataloader = DataLoader(dataset, batch_size=opt["batch_size"], shuffle=True)
    opt["vocab_size"] = dataset.get_vocab_size()
    if opt["model"] == 'S2VTModel':
        model = S2VTModel(
            opt["vocab_size"],
            opt["max_len"],
            opt["dim_hidden"],
            opt["dim_word"],
            opt['dim_vid'],
            rnn_cell=opt['rnn_type'],
            n_layers=opt['num_layers'],
            rnn_dropout_p=opt["rnn_dropout_p"])
    elif opt["model"] == "S2VTAttModel":
        encoder = EncoderRNN(
            opt["dim_vid"],
            opt["dim_hidden"],
            bidirectional=opt["bidirectional"],
            input_dropout_p=opt["input_dropout_p"],
            rnn_cell=opt['rnn_type'],
            rnn_dropout_p=opt["rnn_dropout_p"])
        decoder = DecoderRNN(
            opt["vocab_size"],
            opt["max_len"],
            opt["dim_hidden"],
            opt["dim_word"],
            input_dropout_p=opt["input_dropout_p"],
            rnn_cell=opt['rnn_type'],
            rnn_dropout_p=opt["rnn_dropout_p"],
            bidirectional=opt["bidirectional"])
        model = S2VTAttModel(encoder, decoder)
    elif opt["model"] == "CTCmodel":
        # input_dim, hidden_dim, output_dim, num_layers, biFlag, dropout = 0.5
        # model = CTCmodel(opt["dim_vid"],opt["dim_hidden"],opt["vocab_size"]+1)
        model=CTCmodel(opt['vocab_size'],opt['dim_hidden'])
    elif opt["model"] == "CTC_Hieratical_LSTM":
        encoder = EncoderRNN(
            opt["dim_vid"],
            opt["dim_hidden"],
            # bidirectional=opt["bidirectional"],
            input_dropout_p=opt["input_dropout_p"],
            rnn_cell=opt['rnn_type'],
            rnn_dropout_p=opt["rnn_dropout_p"])
        second_lstm = two_lstm(
            opt["dim_hidden"]*2,
            opt['vocab_size'],
            # bidirectional=opt["bidirectional"],
            input_dropout_p=opt["input_dropout_p"],
            rnn_cell=opt['rnn_type'],
            rnn_dropout_p=opt["rnn_dropout_p"]
        )
        model =CTC_Hieratical_LSTM(encoder,second_lstm,opt['vocab_size'],opt['dim_word'],opt['dim_hidden'],opt['duration'],opt['video_duration'])

    # model = model.cuda()
    # crit = utils.LanguageModelCriterion()
    # rl_crit = utils.RewardCriterion()
    ctc_loss = nn.CTCLoss(reduction='mean')
    optimizer = optim.Adam(
        model.parameters(),
        lr=opt["learning_rate"],
        weight_decay=opt["weight_decay"])
    if opt['lr_schluder'] == 'StepLR':
        lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=opt["learning_rate_decay_every"],
            gamma=opt["learning_rate_decay_rate"])
    elif opt['lr_schluder'] == 'ReduceLROnPlateau':
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.1,
            patience=opt['patience'],
            verbose=True,
            threshold_mode='rel',
            threshold=opt['threshold'],
            cooldown=0,
            min_lr=opt['min_lr'],
            eps=1e-8)
    else:
        raise NotImplementedError('Only implement ReduceLROnPlateau | StepLR')
    opt['check_bool']=False
    if opt['check_bool']:
        check_path = os.path.join(opt['check_path'],'model_10.pth')
        model.load_state_dict(torch.load(check_path))
        opt['root_model_path']=opt['check_path']
        print('have loaded model info from:',check_path)
        #TODO断点重新训练
        val(model, ctc_loss,opt)
    else:
        opt_json = os.path.join(opt["checkpoint_path"], time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time())),
                            'opt_info.json')
        root_model_path = os.path.join(opt['checkpoint_path'],
                                   time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time())))
        opt['root_model_path'] = root_model_path
        if not os.path.isdir(opt["checkpoint_path"]):
            os.mkdir(opt["checkpoint_path"])
        if not os.path.isdir(root_model_path):
            os.mkdir(root_model_path)
        with open(opt_json, 'w') as f:
            json.dump(opt, f)
        print('save opt details to %s' % (opt_json))
    train(dataloader, model, ctc_loss, optimizer, lr_scheduler, opt)

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
        # loss = model.recognize(inputs, inputs_length)
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
    return loss.item()


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
        # tmp_wer = computer_wer(preds,transcripts)*100
        # total_wer +=tmp_wer
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
        loss = model(inputs, inputs_length,targets,targets_length)
        # loss = model.recognize(inputs,inputs_length)
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
    parser.add_argument('-config', type=str, default='config/rnnt.yaml')
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

    # opt = opts_only.parse_opt()
    # opt = vars(opt)
    # print(opt['input_json'])
    # print(opt['info_json'])
    # print(opt['caption_json'])
    # os.environ['CUDA_VISIBLE_DEVICES'] = opt["gpu"]

    # random.seed(1234)
    # np.random.seed(1234)
    # torch.manual_seed(1234)

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

    model = Transducer(config.model)
    # model = VideoModel(config.model)
    if config.training.load_model:
        checkpoint = torch.load(config.training.model_path)
        if config.model.fir_enc_or_not:
            model.fir_enc.load_state_dict(checkpoint['fir_enc'])
            model.encoder.load_state_dict(checkpoint['encoder'])
            model.decoder.load_state_dict(checkpoint['decoder'])
            model.joint.load_state_dict(checkpoint['joint'])
            logger.info('Loaded model from %s' % config.training.model_path)
        else:
            model.encoder.load_state_dict(checkpoint['encoder'])
            model.decoder.load_state_dict(checkpoint['decoder'])
            model.joint.load_state_dict(checkpoint['joint'])
            logger.info('Loaded model from %s' % config.training.load_model)
    elif config.training.load_encoder or config.training.load_decoder:
        if config.training.load_encoder:
            checkpoint = torch.load(config.training.encoder_path)
            model.encoder.load_state_dict(checkpoint['encoder'])
            logger.info('Loaded encoder from %s' %
                        config.training.load_encoder)
        if config.training.load_decoder:
            checkpoint = torch.load(config.training.decoder_path)
            # model.decoder.embedding.load_state_dict(checkpoint['embed'])
            model.decoder.lstm.load_state_dict(checkpoint['lstm'])
            # model.decoder.load_state_dict(checkpoint['decoder'])
            # model.decoder.load_state_dict(checkpoint['decoder'])
            logger.info('Loaded decoder from %s' %
                        config.training.decoder_path)

    if config.training.num_gpu > 0:
        model = model.cuda()
        if config.training.num_gpu > 1:
            device_ids = list(range(config.training.num_gpu))
            model = torch.nn.DataParallel(model, device_ids=device_ids)
        logger.info('Loaded the model to %d GPUs' % config.training.num_gpu)

    n_params, enc, dec,fir_enc = count_parameters(model)
    logger.info('# the number of parameters in the whole model: %d' % n_params)
    logger.info('# the number of parameters in the Encoder: %d' % enc)
    logger.info('# the number of parameters in the Decoder: %d' % dec)
    logger.info('# the number of parameters in the fir_enc: %d' % fir_enc)
    logger.info('# the number of parameters in the JointNet: %d' %
                (n_params - dec - enc-fir_enc))
    optimizer = Optimizer(model.parameters(), config.optim)
    # optimizer = torch.optim.adam(model.parameters(),lr=config.optim.lr,betas=(0.9, 0.98),eps=1e-08,weight_decay=config.optim.weight_decay)
    logger.info('Created a %s optimizer.' % config.optim.type)
    if opt.mode == 'continue':
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        logger.info('From epcho:%d training! '%start_epoch)
        logger.info('Load Optimizer State!')
    else:
        start_epoch = 0


    # create a visualizer
    if config.training.visualization:
        visualizer = SummaryWriter(os.path.join(exp_name, 'log'))
        logger.info('Created a visualizer.')
    else:
        visualizer = None

    # for epoch in range(start_epoch, config.training.epochs):
    #
    #     train_ctc_model(epoch, config, model, train_data,
    #           optimizer, logger, visualizer)
    #
    #     if config.training.eval_or_not:
    #         _ = eval_ctc_model(epoch, config, model, val_data, logger, visualizer)
    #
    #     save_name = os.path.join(exp_name, '%s.epoch%d.chkpt' % (config.training.save_model, epoch))
    #     save_ctc_model(model, optimizer, config, save_name)
    #     logger.info('Epoch %d model has been saved.' % epoch)
    #     if optimizer.lr < 1e-6:
    #         logger.info('The learning rate is too low to train.')
    #         break
    #     logger.info('Epoch %d update learning rate: %.6f' %(epoch, optimizer.lr))
    # logger.info('The training process is OVER!')
    loss_cer = {}
    for epoch in range(start_epoch, config.training.epochs):

        loss = train_rnnt(epoch, config, model, train_data,
              optimizer, logger, visualizer)

        if config.training.eval_or_not and epoch%config.training.eval_fre==0:
            cer = eval_rnnt(epoch, config, model, val_data, logger, visualizer)
            loss_cer[cer] = {}
            loss_cer[cer]['loss']=loss
            loss_cer[cer]['epoch']=epoch
            index = min(list(loss_cer.keys()))
            logger.info('Util epoch %d ,minmize cer : %.6f ,corresponding loss: %.6f  epoch: %d'  % (epoch, index,loss_cer[index]['loss'],loss_cer[index]['epoch']))





        save_name = os.path.join(exp_name, '%s.epoch%d.chkpt' % (config.training.save_model, epoch))
        save_model(model, optimizer, config, save_name)
        logger.info('Epoch %d model has been saved.' % epoch)
        if optimizer.lr < 1e-6:
            logger.info('The learning rate is too low to train.')
            break
        logger.info('Epoch %d update learning rate: %.6f' %(epoch, optimizer.lr))
        logger.info(exp_name)
    logger.info('The training process is OVER!')
    # main(opt)
