import json
import matplotlib.pyplot as plt
'''json'''
def get_data_from_record(filename):
    path =  '/'.join(__file__.split('/')[:-1]) + f'/data_records/{filename}'
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def down_sample_data(data_dict):
    for item in data_dict:
        if not item.endswith('acc'):
            data_dict[item] = data_dict[item][::10]

    return data_dict


def obs_6_figures():
    data_sgd = get_data_from_record('clip_c1000.json')
    data_clip_c5 = get_data_from_record('clip_c10.json')
    data_clip_c10 = get_data_from_record('clip_c1.json')

    data_dict_sgd = {
        'sampling_noise': data_sgd['sampling_noise'],
        # 'last_time_norm_of_grad': data_sgd['last_time_norm_of_grad'],
        'train_acc': data_sgd['train_acc'],
        'test_acc': data_sgd['test_acc'],
        'quantile_0': data_sgd['quantile_0'],
        'quantile_25': data_sgd['quantile_25'],
        'quantile_50': data_sgd['quantile_50'],
        'quantile_75': data_sgd['quantile_75'],
        'quantile_100': data_sgd['quantile_100'],
        'per_grad_mean_norm': data_sgd['per_grad_mean_norm'],
    }

    data_dict_clip_c5 = {
        'sampling_noise': data_clip_c5['sampling_noise'],
        # 'last_time_norm_of_grad': data_clip_c5['last_time_norm_of_grad'],
        'train_acc': data_clip_c5['train_acc'],
        'test_acc': data_clip_c5['test_acc'],
        'quantile_0': data_clip_c5['quantile_0'],
        'quantile_25': data_clip_c5['quantile_25'],
        'quantile_50': data_clip_c5['quantile_50'],
        'quantile_75': data_clip_c5['quantile_75'],
        'quantile_100': data_clip_c5['quantile_100'],
        'per_grad_mean_norm': data_clip_c5['per_grad_mean_norm'],
    }

    data_dict_clip_c10 = {
        'sampling_noise': data_clip_c10['sampling_noise'],
        # 'last_time_norm_of_grad': data_clip_c10['last_time_norm_of_grad'],
        'train_acc': data_clip_c10['train_acc'],
        'test_acc': data_clip_c10['test_acc'],
        'quantile_0': data_clip_c10['quantile_0'],
        'quantile_25': data_clip_c10['quantile_25'],
        'quantile_50': data_clip_c10['quantile_50'],
        'quantile_75': data_clip_c10['quantile_75'],
        'quantile_100': data_clip_c10['quantile_100'],
        'per_grad_mean_norm': data_clip_c10['per_grad_mean_norm'],
    }

    plot_style = 'seaborn-v0_8'
    line_wd = 1.5
    mk_size = 5
    lg_size = 16

    def sub_plot_proc_norm(the_data_dict, name):
        plt.figure(figsize=(6, 4))
        plt.style.use(plot_style)
        
        plt.plot(the_data_dict['sampling_noise'], '--', label = 'sampling_noise', markersize = mk_size, linewidth = line_wd)
        plt.plot(the_data_dict['per_grad_mean_norm'], '--',label = 'per_grad_mean_norm', markersize = mk_size, linewidth = line_wd)
        plt.plot(the_data_dict['quantile_0'], '^--', label = 'quantile_0', markersize = mk_size, linewidth = line_wd)
        plt.plot(the_data_dict['quantile_25'], '*--', label = 'quantile_25', markersize = mk_size, linewidth = line_wd)
        plt.plot(the_data_dict['quantile_50'], 'o--', label = 'quantile_50', markersize = mk_size, linewidth = line_wd)
        # plt.plot(the_data_dict['quantile_75'], label = 'quantile_100', linewidth = 0.8)

        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)

        global norm_max
        plt.ylim(0, norm_max)

        plt.legend(fontsize=lg_size)
        
        plt.tight_layout()
        plt.savefig(f'obs_6_{name}' + '.pdf', bbox_inches='tight')

    def sub_plot_proc_accuracy(the_data_dict, name):
        plt.figure(figsize=(6, 4))
        plt.style.use(plot_style)
        

        plt.plot(the_data_dict['test_acc'], '*--', label = 'test_acc', markersize = mk_size, linewidth = line_wd)
        plt.plot(the_data_dict['train_acc'], '.--', label = 'train_acc', markersize = mk_size, linewidth = line_wd)

        global acc_max
        plt.ylim(0, acc_max)

        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)

        plt.legend(fontsize=lg_size)

        plt.tight_layout()
        plt.savefig(f'obs_6_{name}' + '.pdf', bbox_inches='tight')

    global norm_max
    norm_max = max( max(data_dict_sgd['quantile_50']), max(data_dict_clip_c5['quantile_50']), max(data_dict_clip_c10['quantile_50']),
                    max(data_dict_sgd['sampling_noise']), max(data_dict_clip_c5['sampling_noise']), max(data_dict_clip_c10['sampling_noise'])
                )
    ''' a '''
    the_data_dict = down_sample_data(data_dict_sgd)
    sub_plot_proc_norm(the_data_dict, 'a')

    ''' b '''
    the_data_dict = down_sample_data(data_dict_clip_c5)
    sub_plot_proc_norm(the_data_dict, 'b')

    ''' c '''
    the_data_dict = down_sample_data(data_dict_clip_c10)
    sub_plot_proc_norm(the_data_dict, 'c')

    global acc_max
    acc_max = max(  max(data_dict_sgd['test_acc']), max(data_dict_clip_c5['test_acc']), max(data_dict_clip_c10['test_acc']),
                    max(data_dict_sgd['train_acc']), max(data_dict_clip_c5['train_acc']), max(data_dict_clip_c10['train_acc'])
                )
    ''' e '''
    the_data_dict = data_dict_sgd       
    sub_plot_proc_accuracy(the_data_dict, 'd')

    ''' d '''
    the_data_dict = data_dict_clip_c5
    sub_plot_proc_accuracy(the_data_dict, 'e')

    ''' f '''
    the_data_dict = data_dict_clip_c10
    sub_plot_proc_accuracy(the_data_dict, 'f')


if __name__ == '__main__':
    '''obs_6_figures'''
    obs_6_figures()