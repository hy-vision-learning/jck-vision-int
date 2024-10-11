from enum import Enum


class ModelEnum(Enum):
    custom = 'custom'
    
    resnet18 = 'resnet18'
    resnet34 = 'resnet34'
    resnet50 = 'resnet50'
    resnet101 = 'resnet101'
    resnet152 = 'resnet152'
    
    resnext50 = 'resnext50'
    wide_resnet_16_4 = 'wide_resnet_16_4'
    wide_resnet_28_10_03 = 'wide_resnet_28_10_03'

    def __str__(self):
        return self.value
    
    
class OptimizerEnum(Enum):
    sgd = 'sgd'
    adam = 'adam'

    def __str__(self):
        return self.value
    
    
class LRSchedulerEnum(Enum):
    none = 'none'
    lambda_lr = 'lambda_lr'
    step_lr = 'step_lr'
    cos_annealing = 'cos_annealing'
    custom_annealing = 'custom_annealing'
    one_cycle = 'one_cycle'
    cycle = 'cycle'

    def __str__(self):
        return self.value
    
    