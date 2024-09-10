import torch
#
import train_scheduler
from privacy_analysis import accounting_analysis as aa
from privacy_analysis import handler as ph

import datasets.cifar10 as dms
# import datasets.svhn as dms
# import datasets.imdb as dms
# import datasets.gtsrb as dms
import utility


if __name__ == '__main__':
    arg_setup = utility.parse_args()
    expected_batchsize   = int(arg_setup.expected_batchsize)
    EPOCH                = int(arg_setup.EPOCH)
    epsilon              = float(arg_setup.epsilon)
    lr                   = float(arg_setup.lr)
    aug_num              = float(arg_setup.aug_num)
    
    ''' model and loaders '''
    all_datasets, all_loader = dms.get_all(seed = 1, batchsize_train = expected_batchsize)# the batchsize here is of on meaning
    model = dms.model(num_of_classes = dms.num_of_classes).to(dms.device)
    model.device = dms.device

    ''' old train loader'''
    train_loader = ph.privatized_loader(all_datasets[0], expected_batchsize)
    
    ''' total image number for batch parameter computation, pub data, from train data'''
    ''' batch size for pub data used to compute batch para.'''
    arg_setup.usable_train_data_samples = len(all_datasets[0])
    # batch_para_computer_len, batch_para_computer_batch_size = 50000 - arg_setup.usable_train_data_samples, 0 
    # # using totally 500 public data to compute the batch parameters
    
    ''' sampling rate for training private data '''
    arg_setup.seqnum = 100
    seqnum = arg_setup.seqnum
    
    sampling_rate = expected_batchsize* seqnum/ len(train_loader.dataset) 
    
    ''' compute dp noise '''
    arg_setup.pub_num = 0
    arg_setup.samples_per_group = 0
    arg_setup.self_aug_times = 1

    sigma = arg_setup.sigma =  0.98*aa.get_std(#q = expected_batchsize*seqnum / (arg_setup.usable_train_data_samples),
                                           q = sampling_rate,
                                                 EPOCH = EPOCH, epsilon = epsilon, delta = 1e-5, verbose = True)
    
    
    arg_setup.num_groups = expected_batchsize
    arg_setup.beta = 0
    
    num_of_groups = 80
    
    
    if expected_batchsize%num_of_groups != 0:
        raise ValueError(f'expected_batchsize should be divisible by num_of_groups, {expected_batchsize}, { num_of_groups}')
        # else expected_batchsize//num_of_groups + 1
    arg_setup.group_size = expected_batchsize//num_of_groups 
    arg_setup.chain_len = 10
    arg_setup.forward_beta = 0.8
    # if EPOCH < 20: 
    #     arg_setup.forward_lr = 1
    # else:
    #     arg_setup.forward_lr = 0.1
    
    arg_setup.which_norm = 2
    arg_setup.C = 2.5
    #25*arg_setup.chain_len*0.25
    
    #sigma = arg_setup.sigma = 3*3/1000

    arg_setup.iter_num = int(EPOCH / (arg_setup.num_groups) * 50000)

    opti = torch.optim.SGD(model.parameters(), lr = lr, momentum = 0.725)
    
    ''' function signature '''
    TRAIN_SETUP_LIST = ('epoch', 'device', 'optimizer', 'loss_metric', 'enable_per_grad')
    train_setups = {
                    'epoch': EPOCH, 
                    'device': dms.device, 
                    'optimizer': opti,
                    'loss_metric': dms.loss_metric, 
                    'enable_per_grad': (True, 'opacus'),
                    'sigma': sigma,
                    }

    trainer = train_scheduler.train_master(
                                            model = model,
                                            loaders = [
                                                        train_loader,
                                                        None, 
                                                        all_loader[2], 
                                                        None,
                                                       ],
                                            train_setups = train_setups,
                                            # expected_batchsize = expected_batchsize,
                                            # batch_para_computer_batch_size = batch_para_computer_batch_size,
                                            arg_setup = arg_setup,
                                            
                                            )   
    trainer.train()


