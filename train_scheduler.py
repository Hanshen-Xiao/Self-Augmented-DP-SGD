                    
import enum
import torch
import time
from tqdm import tqdm
import random
import utility
import torchvision
import torchvision.transforms as T
from functorch import combine_state_for_ensemble, make_functional, make_functional_with_buffers
from functorch import vmap, grad
from copy import deepcopy
import os
import numpy as np
import math
''' '''
import logger

mean_list = []
quantile_75_list = []
std_list =[]
quantile_25_list = []
use_cpu = True




''' helpers '''
TRAIN_SETUP_LIST = ('epoch', 'device', 'optimizer', 'loss_metric', 'enable_per_grad')

class Phase(enum.Enum):
    TRAIN = enum.auto()
    VAL = enum.auto()
    TEST = enum.auto()
    PHASE_to_PHASE_str = { TRAIN: "Training", VAL: "Validation", TEST: "Testing"}

class train_master:
    def __init__(self, *,
                model,
                loaders = (None, None, None),
                train_setups = dict(),
                # expected_batchsize = None,
                # batch_para_computer_batch_size = None,
                arg_setup = None,
                ):
        self.data_logger = utility.log_master(root = arg_setup.log_dir)
        logger.init_log(dir = arg_setup.log_dir)
        self.arg_setup = arg_setup

        self.data_recorder = logger.data_recorder(f'clip_c{self.arg_setup.C}.json')

        self.model = model  
        self.num_of_classes = self.model.num_of_classes
        self.num_of_models = arg_setup.num_groups

        self.num_of_groups = arg_setup.num_groups // self.arg_setup.group_size

        models = [ deepcopy(self.model) for _ in range( self.num_of_groups ) ]
        self.worker_model_func, self.worker_param_func, self.worker_buffers_func = combine_state_for_ensemble(models)
        
        # print(111, len(self.worker_param_func))
        for p in self.worker_param_func:
            p.requires_grad = False
            # print('p shape:', p.shape)
        
         
        self.aug_num = arg_setup.aug_num
        self.num_groups = arg_setup.num_groups
        self.times_larger = self.arg_setup.samples_per_group 

        # print(f'==> duplicating model {self.num_groups} times...', end = '')
        # self.worker_models, self.worker_params, self.worker_buffers = combine_state_for_ensemble( [deepcopy(self.model) for _ in range(self.num_groups)] ) 
        # print('done')

        self.loaders = {'train': loaders[0], 'val': loaders[1], 'test': loaders[2], 'batch_computer_loader': loaders[3]}
        # assert len(self.loaders['train']) == len(self.loaders['batch_computer_loader']), (len(self.loaders['train']), len(self.loaders['batch_computer_loader']))

        self.train_setups = train_setups
        
        self.loss_metric = self.train_setups['loss_metric']
        
        ''' sanity check '''
        if self.loaders['train'] is None and self.loaders['val'] is None and self.loaders['test'] is None:
            raise ValueError('at least one loader must be provided')
        for setup in TRAIN_SETUP_LIST:
            if setup not in self.train_setups:
                raise ValueError(f'{setup} must be provided in train_setups')
        for setup in self.train_setups:
            if setup is None:
                raise ValueError(f'invalid setups (no NONE setup allowed): {self.train_setups}')
        
        ''' processing the model '''
        self.sigma = self.train_setups['sigma']
        logger.write_log(f'==>  sigma: {self.sigma}')
        
        ''' set the optimizer after extension '''
        self.optimizer = self.train_setups['optimizer']
        
        ''''''
        self.count_parameters() 
        
        
        print(f'==> have {torch.cuda.device_count()} cuda devices')
        # print(f'current device: {self.model.device}')
        
        # print('==> initializing the momemtum history container...')
        # self.computing_device = self.model.device
        # self.container_device = torch.device("cuda:1")
        # self.per_momentum_history = torch.zeros(50000, self.total_params, device = self.container_device)

        self.shape_interval = []
        self.shape_list = []
        last = 0
        for p in self.model.parameters():
            if p.requires_grad:
                self.shape_list.append(p.shape)
                total_param_sub = p.numel()
                self.shape_interval.append([last, last + total_param_sub])
                last += total_param_sub
            else:
                self.shape_interval.append(None)
        self.all_indexes = list(range(self.arg_setup.usable_train_data_samples))
        
        
        

        self.reindexing = self.get_reindex(self.num_of_models)
        
        ''' transformation list '''
        self.transforms = [
                            T.RandomHorizontalFlip(p = 1),
                            ]


        self.grad_momentum = [ torch.zeros_like(p.data) if p.requires_grad else None for p in self.model.parameters()  ]
        self.iterator_check = [0 for _ in self.model.parameters()]
        self.per_grad_momemtum = [ 0 for _ in self.model.parameters()  ]
    
        # self.un_flattened_grad = []
        # ratio = (self.arg_setup.C_insig**0.5 * self.arg_setup.C_ + sig_C**0.5 * all_big_norm) / self.arg_setup.C

        self.norm_choices = [1+0.25*i for i in range(16)]
        self.avg_pnorms_holder = {norm_choice: [] for norm_choice in self.norm_choices}
        self.avg_inverse_pnorms_holder = {norm_choice: [] for norm_choice in self.norm_choices}

        loader = self.loaders['test']
        '''get whole test data '''
        print('==> stacking test data...')
        self.whole_data_container_test = None
        self.whole_label_container_test = None
        self.whole_index_container_test = None
        for index, train_batch in enumerate(loader):
                        # print(index, end='/')
            # if isinstance(train_batch[1], list) and len(train_batch[1]) ==2:
            #     data_index = train_batch[1][1]
            #     train_batch = (train_batch[0], train_batch[1][0])
            #     # batch_para_batch = None
                
            ''' get training data '''
            inputs, targets = map(lambda x: x.to(self.train_setups['device']), train_batch)
            if self.whole_data_container_test is None:
                self.whole_data_container_test = inputs
                self.whole_label_container_test = targets
            else:   
                self.whole_data_container_test = torch.cat([self.whole_data_container_test, inputs], dim=0)
                self.whole_label_container_test = torch.cat([self.whole_label_container_test, targets], dim=0)
        print(f'==> test data size: {self.whole_data_container_test.size()}')
        print(f'==> all labels:', set(self.whole_label_container_test.tolist()))
        self.transformation = T.Compose([
                                    my_RandomHorizontalFlip(p = 1),
                                    # my_randcrop(32, padding = 4),
                                    ])


        ''' using pub data '''
        self.pub_num = self.arg_setup.pub_num
        self.dummy_index_pub = torch.tensor( [ i for i in range(self.pub_num) ] )
        
        if self.pub_num == 0:
            self.dummy_index_pub = torch.tensor( [] )
        else:
            self.dummy_index_pub_tmp = []
            for _ in range(self.num_groups):
                tmp_index = torch.randint(self.pub_num, (self.times_larger,))
                self.dummy_index_pub_tmp.append( tmp_index)
            self.dummy_index_pub = torch.cat(self.dummy_index_pub_tmp)
        
        print('==> pub data generation done, pub data shape:', self.dummy_index_pub.shape)
        
        '''logging'''
        self.data_logger.write_log(f'weighted_recall.csv', self.arg_setup)
        logger.write_log(f'arg_setup: {self.arg_setup}')
        for i in range(torch.cuda.device_count()):
            logger.write_log(f'\ncuda memory summary for device {i}:\n{torch.cuda.memory_summary(device=f"cuda:{i}", abbreviated=True)}', verbose=True)
    
    def get_reindex(self, real_batch_num):
        ''' reindexing '''
        ''' [pri, pri, pub, pub, pub, pub] -> [pri, pub, pub, pri, pub, pub] '''
        if self.times_larger == 0:
            return torch.tensor( np.arange(real_batch_num) )

        reindexing = []
        for i in range(real_batch_num):
            reindexing.append(i)
            # self.reindexing += list( range(self.num_of_models + i * times_larger, self.num_of_models + (i + 1) * times_larger) )
            ''' fetch the data sample from each data batch, in the same position i'''
            reindexing += [i + self.num_of_models * j for j in range(1, self.times_larger+1)]
        reindexing = torch.tensor(self.reindexing, device = self.model.device)
        return reindexing

    def count_parameters(self):
        total = 0
        cnn_total = 0
        linear_total = 0

        tensor_dic = {}
        for submodule in self.model.modules():
            for s in submodule.parameters():
                if s.requires_grad:
                    if id(s) not in tensor_dic:
                        tensor_dic[id(s)] = 0
                    if isinstance(submodule, torch.nn.Linear):
                            tensor_dic[id(s)] = 1

        for p in self.model.parameters():
            if p.requires_grad:
                total += int(p.numel())
                if tensor_dic[id(p)] == 0:
                    cnn_total += int(p.numel())
                if tensor_dic[id(p)] == 1:
                    linear_total += int(p.numel())

        self.cnn_total = cnn_total
        logger.write_log(f'==>  model parameter summary:')
        logger.write_log(f'     non_linear layer parameter: {self.cnn_total}' )
        self.linear_total = linear_total
        logger.write_log(f'     Linear layer parameter: {self.linear_total}' )
        self.total_params = self.arg_setup.total_para = total
        logger.write_log(f'     Total parameter: {self.total_params}\n' )
    def train(self):
        
        s = time.time()

        for epoch in range(self.train_setups['epoch']):
            logger.write_log(f'\n\nEpoch: [{epoch}] '.ljust(11) + '#' * 35)
            ''' lr rate scheduler '''
            self.epoch = epoch
            train_metrics, val_metrics, test_metrics = None, None, None
            self.record_data_type = 'weighted_recall'

            ''' training '''
            if self.loaders['train'] is not None:
                train_metrics = self.one_epoch(train_or_val=Phase.TRAIN, loader=self.loaders['train'])
                for i in range(torch.cuda.device_count()):
                    logger.write_log(f'\ncuda memory summary for device {i}:\n{torch.cuda.memory_summary(device=f"cuda:{i}", abbreviated=True)}', verbose=False)
            
            ''' validation '''
            if self.loaders['val'] is not None:
                val_metrics = self.one_epoch(train_or_val = Phase.VAL, loader = self.loaders['val'])

            ''' testing '''
            if self.loaders['test'] is not None:
                test_metrics = self.one_epoch(train_or_val = Phase.TEST, loader = self.loaders['test'])

            '''logging data '''
            data_str = (' '*3).join([
                                f'{epoch}',
                                f'{ float( train_metrics.__getattr__(self.record_data_type) ) * 100:.2f}%'.rjust(7)
                                if train_metrics else 'NAN',

                                f'{ float( val_metrics.__getattr__(self.record_data_type) ) * 100:.2f}%'.rjust(7)
                                if val_metrics else 'NAN',

                                f'{ float( test_metrics.__getattr__(self.record_data_type) ) * 100:.2f}%'.rjust(7)
                                if test_metrics else 'NAN',
                                ])
            
            self.data_logger.write_log(f'{self.record_data_type}.csv', data_str)

        ''' ending '''

        self.data_recorder.save()
        logger.write_log(f'\n\n=> TIME for ALL : {time.time()-s:.2f}  secs')
    
    def _per_sample_augmentation(self):
        ''' per sample augmentation '''
        # if self.times_larger == 0:
        #     return torch.tensor([], device=self.model.device), torch.tensor([], dtype=torch.int64, device=self.model.device)
        if self.pub_num == 0:
            return torch.tensor([], device=self.model.device), torch.tensor([], dtype=torch.int64, device=self.model.device)
        tmp_index = np.random.permutation(self.dummy_index_pub)

        pub_input = self.whole_data_container_test[tmp_index]
        pub_target = self.whole_label_container_test[tmp_index]
        
        # print('pub_input shape:', pub_input.shape,  'pub_target shape:', pub_target.shape)
        return pub_input, pub_target
    

    def sampling_noise_summary(self, index, per_grad):
        grad_flatten = self.flatten_to_rows(per_grad[0].shape[0], per_grad)

        grad_flatten_mean = torch.mean(grad_flatten, dim=0, keepdim=True)
        center_around_mean = grad_flatten - grad_flatten_mean
        grad_norm = torch.norm(grad_flatten, dim=1)
        
        mean_of_grad_norm = grad_norm.mean()
        print(mean_of_grad_norm)
        
        ## ADAPTIVE CLIPPING 
        self.arg_setup.C = float(mean_of_grad_norm)
        
        # if float(mean_of_grad_norm) >5:
        #     self.arg_setup.C = 5
        # else:
        #     self.arg_setup.C = float(mean_of_grad_norm)
        
        grad_norm_0 = grad_norm - self.arg_setup.C
        
        #grad_norm = grad_norm - 2
        grad_norm_0[grad_norm_0 < 0] = 0
        #sorted_grad_norm = torch.sort(grad_norm)[0]
        exceed_norm = grad_norm_0.mean()
        print(exceed_norm/self.arg_setup.C)
        
        # grad_flatten = self.flatten_to_rows(per_grad[0].shape[0], per_grad)

        # grad_flatten_mean = torch.mean(grad_flatten, dim=0, keepdim=True)
        # center_around_mean = grad_flatten - grad_flatten_mean
       
        # grad_norm = torch.norm(grad_flatten, dim=1)
        # mean_of_grad_norm = grad_norm.mean()
        # grad_norm = grad_norm - float(mean_of_grad_norm)
        # print(float(mean_of_grad_norm))
        # self.arg_setup.C = float(mean_of_grad_norm)
        
        
        # grad_norm[grad_norm < 0] = 0
        
        # sorted_grad_norm = torch.sort(grad_norm)[0]
        
        
        # quantile_0_25_50_75_100 = [sorted_grad_norm[0]] + [sorted_grad_norm[int(len(sorted_grad_norm) * i / 4) - 1] for i in range(1,5)]
        # quantile_0_25_50_75_100 = [round(float(q),3) for q in quantile_0_25_50_75_100]
        # last_time_norm_of_grad_used_to_update_model = torch.cat([p.reshape(1, -1) for p in self.grad_momentum], dim=1).norm()
        # sampling_noise = torch.norm(center_around_mean, dim=1).mean()
        # per_grad_mean_norm = grad_flatten_mean.norm()

        # self.data_recorder.add_record('sampling_noise', float(sampling_noise))
        # self.data_recorder.add_record('quantile_0', quantile_0_25_50_75_100[0])
        # self.data_recorder.add_record('quantile_25', quantile_0_25_50_75_100[1])
        # self.data_recorder.add_record('quantile_50', quantile_0_25_50_75_100[2])
        # self.data_recorder.add_record('quantile_75', quantile_0_25_50_75_100[3])
        # self.data_recorder.add_record('quantile_100', quantile_0_25_50_75_100[4])
        # self.data_recorder.add_record('per_grad_mean_norm', float(per_grad_mean_norm) )
        # self.data_recorder.add_record('last_time_norm_of_grad', float(last_time_norm_of_grad_used_to_update_model))

        # mean_of_grad_norm = grad_norm.mean()
        # logger.write_log(f'    ----> last time norm of grad used to update model: {last_time_norm_of_grad_used_to_update_model:.2f}')
        # logger.write_log(f'    ----> sampling noise: {sampling_noise:.2f}')
        # logger.write_log(f'    ----> quantile 0, 25, 50, 75, 100: {quantile_0_25_50_75_100}')
        # logger.write_log(f'    ----> grad norm mean: {mean_of_grad_norm:.2f}, std: {grad_norm.std():.2f}')
        # logger.write_log(f'    ----> norm of avg of per grad: {per_grad_mean_norm:.2f}')
        # logger.write_log('\n\n')
        
        # mean_list.append(float(mean_of_grad_norm))
        # quantile_25_list.append(quantile_0_25_50_75_100[1])
        # item_std = float(grad_norm.std())
        # std_list.append(item_std)
        # quantile_75_list.append(quantile_0_25_50_75_100[3])
        
        # print('mean', mean_list)
        # print('std', std_list)
        # print('quantile_25', quantile_25_list)
        # print('quantile_75', quantile_75_list)


    def get_per_grad(self, inputs, targets):

        def compute_loss(model_para, buffers,  inputs, targets):
            # print(f'inputs shape: {inputs.shape}')
            predictions = self.worker_model_func(model_para, buffers, inputs)
            # print(f'predictions shape: {predictions.shape}, targets shape: {targets.shape}')
            ''' only compute the loss of the first(private) sample '''
            loss = self.loss_metric(predictions, targets.flatten()) #* inputs.shape[0]
            return loss
        
        def self_aug_per_grad(model_para, buffers, inputs, targets):
            
            init_model = [p.clone() for p in model_para]
            # running_model = [torch.clone(p) for p in model_para]
            
            momemtum = [0 for _ in range(len(model_para))]
            chain_len = self.arg_setup.chain_len
            beta = self.arg_setup.forward_beta
            a = self.epoch
            lr_0 = 0.025
            # if a < 50:
            #     lr_0 = 0.04
            # else:
            #     lr_0 = 0.01

            for _ in range(chain_len):
                per_grad = grad(compute_loss)(model_para, buffers, inputs, targets)
                # for p in per_grad:
                #     print(p.shape)
                momemtum = [beta*m + g for m, g in zip(momemtum, per_grad)]
                for p_worker, p_momemtum in zip(model_para, momemtum):
                    # print(p_worker.data.shape, p_momemtum.shape)
                    p_worker.add_(- lr_0 * p_momemtum)
                    
            per_grad = [i - p for p, i in zip(model_para, init_model)]

            return list(per_grad)
        

        per_grad = vmap( self_aug_per_grad, in_dims=(0, 0, 0, 0) )(self.worker_param_func, self.worker_buffers_func, inputs, targets)

        return per_grad
    

    def one_epoch(self, *, train_or_val, loader):
        metrics = utility.ClassificationMetrics(num_classes=self.num_of_classes)
        metrics.num_images = metrics.loss = 0
        is_training = train_or_val is Phase.TRAIN
        
        T_normalize = T.Normalize(
                        mean = [0.485, 0.456, 0.406],
                        std = [0.229, 0.224, 0.225],
                        )
        transformation = T.Compose([
                                    T.ToPILImage(),
                                    # T.AutoAugment(T.AutoAugmentPolicy.CIFAR10),
                                    # T.RandomCrop(size=(32, 32), padding=4),
                                    T.RandomHorizontalFlip(),  
                                    # T.RandomRotation(degrees=(-10, 10),),
                                    
                                    T.ToTensor(),
                                    T_normalize,
                                            
                                    # T.AutoAugment(T.AutoAugmentPolicy.CIFAR10),
                                    # T.RandomPerspective(distortion_scale=0.5, p=1.0),
                                    # T.RandomHorizontalFlip(),
                                    ])

        with torch.set_grad_enabled(is_training):
            self.model.train(is_training)
            s = time.time()
            if is_training:
                print(f'==> have {len(loader)} iterations in this epoch')
                count = 0 
                per_grad_list = []

                for index, train_batch in enumerate(loader):
                    count = count + 1
                    ''' get training data '''
                    the_inputs, the_targets = map(lambda x: x.to(self.train_setups['device']), train_batch)
                    #print(the_inputs.shape, the_targets.shape)
                    new_inputs, new_targets = [], []
                    for img, label in zip(the_inputs, the_targets):
                        new_inputs += [transformation(img) for _ in range(self.aug_num)]
                        new_targets += [label for _ in range(self.aug_num)]
                    new_inputs, new_targets = torch.stack(new_inputs).to(self.train_setups['device']), torch.stack(new_targets).to(self.train_setups['device'])
                    #print(new_inputs.shape, new_targets.shape)
                    
                    #pub_inputs, pub_targets = self._per_sample_augmentation()
                    #new_inputs = torch.concat([the_inputs, pub_inputs], dim=0)
                    #new_targets = torch.concat([the_targets, pub_targets], dim=0)

                    #reindexing = self.get_reindex(the_inputs.shape[0])
                    #assert new_inputs.shape[0] == len(reindexing)
                    #new_inputs = new_inputs[reindexing]
                    #new_targets = new_targets[reindexing]

                    #assert new_inputs.shape[0] == the_inputs.shape[0] * (
                    #        self.times_larger + 1), f'new input shape: {new_inputs.shape}'

                    before_group_size = new_inputs.shape[0]
                    group_size = self.arg_setup.group_size * self.aug_num
                    new_inputs = torch.stack(torch.split(new_inputs, group_size, dim=0))
                    new_targets = torch.stack(torch.split(new_targets, group_size, dim=0))
                    #print(new_inputs.shape, new_targets.shape)

                    assert new_inputs.shape[0] == before_group_size // group_size, f'new input shape: {new_inputs.shape}'

                    per_grad = self.get_per_grad(new_inputs, new_targets)
                    if len(per_grad_list) == 0:
                        per_grad_list = per_grad
                    else:
                        for i, p in enumerate(per_grad):
                            if use_cpu:
                                per_grad_list[i] = torch.cat((per_grad_list[i].cpu(), p.cpu()), dim = 0).to(self.train_setups['device'])
                            else:
                                per_grad_list[i] = torch.cat((per_grad_list[i], p), dim = 0)
                        # i=0
                        # for p, p_list in zip(per_grad, per_grad_list):
                        #     #print(p.shape, p_list.shape)
                        #     p_list = torch.cat((p_list, p), dim = 0)
                        #     per_grad_list[i] = p_list
                        #     i +=1
                           
                    
                    
                    for p_worker, p_model in zip(self.worker_param_func, self.model.parameters()):
                        # assert p_worker.shape == p_model.data.shape
                        shape_len = len(p_model.data.shape)
                        p_worker.copy_( p_model.data.unsqueeze(0).repeat(self.num_of_groups, *[1 for _ in range(shape_len)]) )
                    #print("VALUE IS", torch.sum(self.worker_param_func[3]))
                    
                    if count ==1:
                        self.sampling_noise_summary(index, per_grad_list)       
                    
                    if count % self.arg_setup.seqnum == 0:
                        # if index % 1 == 0:
                        #     self.sampling_noise_summary(index, per_grad_list)
                        
                        # for i in range(len(per_grad)):
                        #     for j in range(self.arg_setup.seqnum-1):
                        #         temp = per_grad_list[j]
                        #         per_grad[i] = temp[i] + per_grad[i]
                            
                        #     per_grad[i] = per_grad[i] /self.arg_setup.seqnum
                       
                        self.other_routine(per_grad_list)
                        '''update batch metrics'''
                        with torch.no_grad():
                            predictions = self.model(the_inputs)
                            loss = self.train_setups['loss_metric'](predictions, the_targets.flatten())
                        metrics.batch_update(loss, predictions, the_targets)
                        count = 0
                        per_grad_list = []

                self.data_recorder.add_record('train_acc', float(metrics.__getattr__(self.record_data_type)))
            else:
                for batch in loader:
                    inputs, targets = map(lambda x: x.to(self.train_setups['device']), batch)

                    predicts = self.model(inputs)
                    loss = self.train_setups['loss_metric'](predicts, targets.flatten())

                    '''update batch metrics'''
                    metrics.batch_update(loss, predicts, targets)

                self.data_recorder.add_record('test_acc', float(metrics.__getattr__(self.record_data_type)))

        metrics.loss /= metrics.num_images
        logger.write_log(f'==> TIME for {train_or_val}: {int(time.time() - s)} secs')
        logger.write_log(f' {train_or_val}: {self.record_data_type} = {float(metrics.__getattr__(self.record_data_type)) * 100:.2f}%')

    def clip_per_grad(self, per_grad_list):
        
        per_grad_norm = ( self._compute_per_grad_norm(per_grad_list, which_norm = self.arg_setup.which_norm, use_cpu = use_cpu) + 1e-6)
        
        # self.arg_setup.C = self.arg_setup.C * math.exp( math.log(2/10) / self.arg_setup.iter_num)

        ''' clipping/normalizing '''
        multiplier = torch.clamp(self.arg_setup.C / per_grad_norm, max = 1)
        for index, p in enumerate(per_grad_list):
            
            ''' normalizing '''
            # per_grad[index] = p / self._make_broadcastable(per_grad_norm / self.arg_setup.C, p) 
            ''' clipping '''
            # print(f'p shape: {p.shape}, mutiplier shape: {multiplier.shape}')
            per_grad_list[index] = p * self._make_broadcastable( multiplier, p ) 
        return per_grad_list

    def other_routine(self, per_grad_list):

        ''' vanilla dp-sgd '''
        per_grad_list = self.clip_per_grad(per_grad_list)
        # per_grad_list = torch.tensor(per_grad_list)
        # print("after clip", per_grad_list.shape)
        assert len(self.iterator_check) == len(per_grad_list)
        #assert len(self.iterator_check) * self.arg_setup.seqnum == len(per_grad_list)
        for p_stack, p in zip(per_grad_list, self.model.parameters()):
            if p.requires_grad:
                p.grad = torch.sum(p_stack, dim = 0) 
                p.grad += self.arg_setup.C * self.sigma * torch.randn_like(p.grad) 
                #p.grad /= p_stack.shape[0] 
                p.grad /= (self.num_of_groups*self.arg_setup.seqnum)
                #print(p_stack.shape[0])
        #print(self.num_of_groups*self.arg_setup.seqnum)
        #print('C',self.arg_setup.C)
        self.model_update()
        
    def _compute_per_grad_norm(self, iterator, which_norm = 2, use_cpu = False):
        assert(len(iterator) > 0)
        if use_cpu:
            it_cpu = [p.cpu() for p in iterator]
            all_grad = torch.cat([p.reshape(p.shape[0], -1) for p in it_cpu], dim = 1)
        else:
            all_grad = torch.cat([p.reshape(p.shape[0], -1) for p in iterator], dim = 1)
        per_grad_norm = torch.norm(all_grad, dim = 1, p = which_norm)
        # assert int(per_grad_norm.numel()) == self.num_groups, (int(per_grad_norm.numel()), self.num_groups)
        return per_grad_norm.cuda(iterator[0].device)
    
    def _make_broadcastable(self, tensor_to_be_reshape, target_tensor):
        broadcasting_shape = (-1, *[1 for _ in target_tensor.shape[1:]])
        return tensor_to_be_reshape.reshape(broadcasting_shape)
    
    def model_update(self):
        # ''' lr scheduling '''
        # if self.epoch > 20:
        #     self.optimizer.param_groups[0]['lr'] = 0.05
        # if self.epoch > 30:
        #     self.optimizer.param_groups[0]['lr'] = 0.02
        # if self.epoch > 40:
        #     self.optimizer.param_groups[0]['lr'] = 0.01
        
        ''' update the model '''
        self.optimizer.step()
        
        # ''' copy global model to worker model'''
        # for p_worker, p_model in zip(self.worker_param_func, self.model.parameters()):
        #     assert p_worker.shape == p_model.data.shape
        #     p_worker.copy_(p_model.data)
            
        ''' copy global model to worker model'''
        for p_worker, p_model in zip(self.worker_param_func, self.model.parameters()):
            # assert p_worker.shape == p_model.data.shape
            shape_len = len(p_model.data.shape)
            #print(len(p_model.data.shape))
            p_worker.copy_( p_model.data.unsqueeze(0).repeat(self.num_of_groups, *[1 for _ in range(shape_len)]) )
            

    def flatten_to_rows(self, leading_dim, iterator):
        return torch.cat([p.reshape(leading_dim, -1) for p in iterator], dim = 1)
    
    
    def each_layer_grad_summary(self, per_grad):
        
        for index, p in enumerate(self.model.parameters()):
            print(f'model layer-{index}, epoch-{self.epoch}:\n \
                \tshape: {p.shape}') 

        for index, p in enumerate(per_grad):
            print(f'layer-{index}, epoch-{self.epoch}:\n \
                \tshape: {p.shape}\n \
                \tgrad mean: {torch.mean(p)}\n \
                \tgrad std: {torch.std(p)}\n' 
                )
        exit()
    
                    
                    



import random
class my_randcrop(T.RandomCrop):
    @staticmethod
    def get_params(img, output_size) :
        _, h, w = None, 32,32
        th, tw = output_size

        if h < th or w < tw:
            raise ValueError(f"Required crop size {(th, tw)} is larger than input image size {(h, w)}")

        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th )
        j = random.randint(0, w - tw )
        return i, j, th, tw

class my_RandomHorizontalFlip(T.RandomHorizontalFlip):
    def forward(self, img):
        if random.random() < self.p:
            return img.flip(-1)
        return img
