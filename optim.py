import torch.optim as optim


class Optimizer(object):
    def __init__(self, parameters, config):
        self.config = config

        self.optimizer = build_optimizer(parameters, config)
        self.lr_schluer = build_schluer(self.optimizer,self.config.step_size,self.config.gamma)
        self.global_step = 1
        self.current_epoch = 0
        self.lr = config.lr
        self.decay_ratio = config.decay_ratio
        self.epoch_decay_flag = False

    def step(self):
        self.global_step += 1
        self.optimizer.step()

    def epoch(self):
        self.current_epoch += 1
        self.lr_schluer.step()
        self.lr=self.lr_schluer.optimizer.param_groups[0]['lr']


    def zero_grad(self):
        self.optimizer.zero_grad()

    def state_dict(self):
        return (self.optimizer.state_dict(),self.lr_schluer.state_dict())

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict[0])
        self.lr_schluer.load_state_dict(state_dict[1])

    # def decay_lr(self):
    #     self.lr *= self.decay_ratio
    #     for param_group in self.optimizer.param_groups:
    #         param_group['lr'] = self.lr

def build_schluer(optimizer,step_size,gamma):
    return optim.lr_scheduler.StepLR(optimizer,step_size,gamma)
def build_optimizer(parameters, config):
    if config.type == 'adam':
        return optim.Adam(
            parameters,
            lr=config.lr,
            weight_decay=config.weight_decay
        )
    elif config.type == 'sgd':
        return optim.SGD(
            params=parameters,
            lr=config.lr,
            momentum=config.momentum,
            nesterov=config.nesterov,
            weight_decay=config.weight_decay
        )
    elif config.type == 'adadelta':
        return optim.Adadelta(
            params=parameters,
            lr=config.lr,
            rho=config.rho,
            eps=config.eps,
            weight_decay=config.weight_decay
        )
    else:
        raise NotImplementedError
