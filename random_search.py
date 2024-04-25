import os
import time
import shutil
import argparse
from types import SimpleNamespace

import joblib
import numpy as np
import torch
from sklearn.model_selection import train_test_split

from src.utils.tools import (get_learning_rate_scheduler, get_scaler,
                             get_optimizer, get_criterion, get_model, 
                             set_device, set_seed, train, validation,
                             parse_model_name, get_data_folder, ffkgcn_norm_data)
from src.utils.checkpoint import load_checkpoint, save_checkpoint
from src.utils.files_manager import get_directory_for_results, save_results
from src.utils.generate_search_space import generate_search_space
from src.utils.load_dataset import Dataset
from src.utils.logger import set_logger
from src.utils.printing import color_text
from src.configs.configs_handler import toml_to_configs


def main(config_file:str='/media/oalab/jhon/last_repo_tnnls/gegen/configs_mrgnn/configs.random_search_intel_m01.toml'):

    logger = set_logger('random_search')

    configs = toml_to_configs(config_file)

    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')

    set_seed(configs.dev_seed)

    # check if checkpoint exists
    if os.path.isfile(os.path.join('checkpoints', f'checkpoint_{configs.run_name}.pth')):
        logger.debug(color_text('Checkpoint found. Loading...', color='magenta', style='bold'))
        state = load_checkpoint(os.path.join('checkpoints', f'checkpoint_{configs.run_name}.pth'))
        folder, split, repetition = state['folder'], state['split'], state['repetition']+1
        hyperparameters, hyperparameters_list = state['hyperparameters'], state['hyperparameters_list']
        gcn_errors= state['mse_error_gcn']
        logger.debug(color_text(f'Loading checkpoint for split {split+1} and repetition {repetition+1}...',
                                color='green', style='bold'))
    else:
        folder = get_directory_for_results(configs.output_folder, configs.run_name)
        shutil.copyfile(config_file, os.path.expanduser(os.path.join(folder, 'settings.toml')))
        # shutil.copyfile(models.__file__, os.path.expanduser(os.path.join(folder, 'model.py')))
        hyperparameters, hyperparameters_list = generate_search_space(configs, n_runs=configs.n_runs, round_decimals=5)
        gcn_errors = np.zeros((len(hyperparameters)))
        save_results(folder, 'random_search_list.mat', 
                {'hyperparameters': hyperparameters, 
                'hyperparameters_list' : hyperparameters_list})
        split = repetition = 0

    # setting cuda device
    cuda = not configs.no_cuda and torch.cuda.is_available()
    device = set_device(cuda, configs.gpu)

    # loading data
    dataset = Dataset(
            root=get_data_folder(configs.data_folder, configs.model), 
            dataset=configs.dataset,
            sampling_density=configs.sampling_density,
            transform=get_scaler(configs.normalization))
    data = dataset()

    model_name = parse_model_name(configs.model)

    data.x = torch.mm(data.x, data.Dh)

    train_idx = data.train_idx.numpy().astype(int)

    for iseed in range(split, len(configs.val_seeds)):

        if not os.path.exists(os.path.join(folder, f'split_{iseed+1}')):
            os.makedirs(os.path.join(folder, f'split_{iseed+1}'))

        train_index, val_index = train_test_split(
                                    train_idx,
                                    test_size=np.max((configs.validation_size, 0.1)),
                                    random_state=configs.val_seeds[iseed]
                                    )
        data.train_idx = torch.tensor(train_index, dtype=torch.long)
        data.val_idx = torch.tensor(val_index, dtype=torch.long)

        if dataset.scaler is not None:
            joblib.dump(dataset.scaler, os.path.join(folder, f'split_{iseed+1}', 'norm_scaler.pkl'))

        save_results(os.path.join(folder, f'split_{iseed+1}'), 'dataset_attrs.mat',
            {'sampling_patterns' : data.sampling_pattern.cpu().numpy(),
            'train_index' : train_index,
            'val_index' : val_index,
            'test_index' : data.test_idx.cpu().numpy().astype(int),
            'split': iseed+1,
            'seed_val': configs.val_seeds[iseed]}
            )

        mse_error_gcn = gcn_errors
        lowest_mse_all = np.Inf

        for i, (lr, epsilon, order, alpha, heads, n_epochs, dropout, k, gamma_norm, output, lambda_par, n_conv_layers, n_conv_units, 
                n_linear_layers, n_linear_units, weight_decay) in enumerate(hyperparameters[repetition:,:], start=repetition):
            logger.info(color_text(f'model {configs.model} | split {iseed+1}/{len(configs.val_seeds)}, search rep {i+1}/{len(hyperparameters)} | '+
                f'lr : {lr:2.3f}, epsilon : {epsilon:2.2f}, order : {order:d}, alpha : {alpha:2.1f}, heads : {heads}, ' +
                f'n_epochs : {n_epochs:d}, dropout : {dropout:2.2f}, lambda_par : {lambda_par:.3e}, n_conv_layers : {n_conv_layers:d}, ' + 
                f'n_conv_units : {n_conv_units:d}, n_linear_layers : {n_linear_layers:d}, ' +
                f'n_linear_units : {n_linear_units:d}, weight_decay : {weight_decay:.3e}', color='blue', style='bold'))

            # adding epsilon to the diagonal
            data.Le = data.L.to(device) + (epsilon*torch.eye(data.L.shape[0])).to(device)

            # parsing model name
            model_params = SimpleNamespace(layer_normalization=configs.layer_normalization, lr=lr, epsilon=epsilon, filter_order=order, 
                                           alpha=alpha, heads=heads, n_epochs=n_epochs, dropout=dropout, k=k, gamma_norm=gamma_norm, 
                                           output=output, lambda_par=lambda_par, n_conv_layers=n_conv_layers, n_conv_hidden=n_conv_units,
                                           n_linear_layers=n_linear_layers, n_linear_hidden=n_linear_units, weight_decay=weight_decay,
                                           device=device, model_kwargs=configs.model_kwargs)

            if model_name == 'ffk':
                data = ffkgcn_norm_data(data, model_params)

            model_path, model = get_model(model_name, data, model_params)
            model = model.to(device)
            shutil.copyfile(model_path, os.path.expanduser(os.path.join(folder, 'model.py')))

            optimizer = get_optimizer(configs.optim, model.parameters(), lr=lr, weight_decay=weight_decay, **configs.optim_kwargs)
            scheduler = get_learning_rate_scheduler(configs.lr_scheduler, optimizer, **configs.lr_scheduler_kwargs)
            criterion = get_criterion(configs.criterion, **configs.criterion_kwargs)

            lowest_mse = np.Inf
            train_mse_error = []
            val_mse_error = []
            t_total = time.time()
            opts = SimpleNamespace(verbose=configs.verbose, lambda_param=lambda_par, epochs=n_epochs)

            for epoch in range(n_epochs):
                train_error = train(model, data.to(device), optimizer, criterion, epoch, opts)
                train_mse_error.append(train_error)

                val_error = validation(model, data.to(device), criterion, opts)
                val_mse_error.append(val_error)

                if val_error < lowest_mse:
                    logger.info(color_text('Saving best model...', color='yellow', style='bold'))
                    torch.save(model.state_dict(), os.path.join(folder, f'split_{iseed+1}', 'best_model.pth'))
                    lowest_mse = val_error

                # Update learning rate
                if configs.lr_scheduler == 'plateau':
                    scheduler.step(val_error)
                else:
                    scheduler.step()

            logger.info(color_text(f'Total time elapsed: {time.time() - t_total:.4f}s', color='green', style='bold'))

            mse_error_gcn[i] = lowest_mse

            save_results(os.path.join(folder, f'split_{iseed+1}'), f'errors_repetition_{i+1}.mat',
                {'val_mse_error': mse_error_gcn[i],
                'val_rmse_error': np.sqrt(mse_error_gcn[i]),
                'train_error': train_mse_error,
                'val_error': val_mse_error}
                )

            save_results(os.path.join(folder, f'split_{iseed+1}'), 'random_search_errors.mat',
                {'all_val_mse_error': mse_error_gcn,
                'all_val_rmse_error': np.sqrt(mse_error_gcn)})

            if lowest_mse < lowest_mse_all:
                lowest_mse_all = lowest_mse
                logger.info(color_text(f'Lowest gcn error {lowest_mse_all:2.4f}\n', color='blue', style='bold'))
                save_results(os.path.join(folder, f'split_{iseed+1}'), 'model_lowest_error.mat',
                        {'mse': lowest_mse_all,
                        'repetition' : i+1})

            # saving checkpoint
            if configs.save_checkpoint:
                save_checkpoint(
                    os.path.join('checkpoints', f'checkpoint_{configs.run_name}.pth'),
                    {'folder' : folder,
                    'hyperparameters' : hyperparameters,
                    'hyperparameters_list' : hyperparameters_list,
                    'split' : iseed,
                    'repetition' : i,
                    'mse_error_gcn' : mse_error_gcn}
                    )

            repetition = 0

        # Remove checkpoint file
        logger.info(color_text('Removing checkpoint...', color='white', style='bold'))
        os.remove(os.path.join('checkpoints', f'checkpoint_{configs.run_name}.pth'))

        logger.info(color_text('Optimization Finished!', color='green', style='bold'))


def _cli():
    '''

    command line interface

    '''

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        argument_default=argparse.SUPPRESS
    )

    # add in the file name
    parser.add_argument('-f', '--config_file',
                        help='Input configuration file. Defaults to settings.sample.toml')

    args = parser.parse_args()

    return vars(args)

if __name__ == '__main__':
    main(**_cli())