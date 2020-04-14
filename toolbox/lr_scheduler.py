

class LR_Scheduler:
    def get_lr(self, epoch):
        pass
        
class Step(LR_Scheduler):
    def __init__(self, cfg):
        self.initial = cfg.initial
        self.interval = cfg.interval
        self.factor = cfg.factor
        print('[Step LR Scheduler] init: {}; interval: {}; factor: {}'.format(self.initial, self.interval, self.factor))
        
    def __call__(self, epoch):
        return self.initial * (self.factor ** (epoch // self.interval))

class Manual(LR_Scheduler):
    def __init__(self, cfg):
        self.steps = sorted(zip(cfg.timesteps, cfg.values)) # Ascent order
        print('[Manual LR Scheduler] {}'.format(self.steps))

    def __call__(self, epoch):
        lr = None
        for t, v in self.steps:
            if t <= epoch:
                lr = v
        return lr
