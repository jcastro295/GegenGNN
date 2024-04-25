import os
import time
import shutil
import argparse

import joblib
import torch
import numpy as np
from sklearn.model_selection import train_test_split

from src.utils.tools import (get_learning_rate_scheduler, get_scaler,
                             get_optimizer, get_criterion, get_model, 
                             set_device, set_seed, train, test, validation,
                             parse_model_name, get_data_folder, ffkgcn_norm_data)
from src.utils.checkpoint import load_checkpoint, save_checkpoint
from src.utils.files_manager import get_directory_for_results, save_results
from src.utils.early_stopping import EarlyStopping
from src.utils.load_dataset import Dataset
from src.utils.logger import set_logger
from src.utils.printing import color_text
from src.configs.configs_handler import toml_to_configs


def main(config_file:str='configs.single_train.toml'):

    logger = set_logger('train')

    configs = toml_to_configs(config_file)

    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')

    if os.path.isfile(os.path.join('checkpoints', f'checkpoint_{configs.run_name}.pth')):
            logger.debug(color_text('Checkpoint found. Loading...', color='magenta', style='bold'))
            state = load_checkpoint(os.path.join('checkpoints', f'checkpoint_{configs.run_name}.pth'))
            folder, partition = state['folder'], state['partition']
    else:
        folder = get_directory_for_results(configs.output_folder, configs.run_name)
        shutil.copyfile(config_file, os.path.expanduser(os.path.join(folder, 'settings.toml')))
        # shutil.copyfile(models.__file__, os.path.expanduser(os.path.join(folder, 'model.py')))
        partition = 0

    # setting cuda device
    cuda = not configs.no_cuda and torch.cuda.is_available()
    device = set_device(cuda, configs.gpu)

    # setting random seeds
    set_seed(configs.dev_seed)

    # loading data
    dataset = Dataset(
            root=get_data_folder(configs.data_folder, configs.model), 
            dataset=configs.dataset,
            sampling_density=configs.sampling_density,
            transform=get_scaler(configs.normalization))
    data = dataset()

    model_name = parse_model_name(configs.model)

    if model_name == 'ffk':
        data = ffkgcn_norm_data(data, configs)

    data.x = torch.mm(data.x, data.Dh)
    # adding epsilon to the diagonal
    data.Le = data.L + configs.epsilon*torch.eye(data.L.shape[0])

    train_idx = data.train_idx.numpy().astype(int)

    for iseed in range(partition, len(configs.test_seeds)):

        if not os.path.exists(os.path.join(folder, f'model_{iseed+1}')):
            os.makedirs(os.path.join(folder, f'model_{iseed+1}'))
        
        train_index, val_index = train_test_split(
                                    train_idx,
                                    test_size=np.max((configs.validation_size, 0.1)),
                                    random_state=configs.test_seeds[iseed]
                                    )
        data.train_idx = torch.tensor(train_index, dtype=torch.long)
        data.val_idx = torch.tensor(val_index, dtype=torch.long)

        # saving normalization scaler
        if dataset.scaler is not None:
            joblib.dump(dataset.scaler, os.path.join(folder, f'model_{iseed+1}', 'norm_scaler.pkl'))

        logger.info(color_text(f'order : {configs.filter_order:d}, dropout : {configs.dropout:2.2f}, ' +
            f'lr : {configs.lr:2.3f}, n_conv_layers: {configs.n_conv_layers:d}, ' +
            f'n_conv_units : {configs.n_conv_hidden:d}, n_linear_layers: {configs.n_linear_layers:d}, '+
            f'n_linear_units : {configs.n_linear_hidden:d}, weight_decay : {configs.weight_decay:.3e}, ' +
            f'lambda : {configs.lambda_param:.3e}, n_epochs : {configs.epochs:d}' , color='blue', style='bold'))

        model_path, model = get_model(model_name, data, configs)
        model = model.to(device)
        shutil.copyfile(model_path, os.path.expanduser(os.path.join(folder, 'model.py')))

        optimizer = get_optimizer(
             configs.optim, model.parameters(), lr=configs.lr, weight_decay=configs.weight_decay, **configs.optim_kwargs)
        scheduler = get_learning_rate_scheduler(configs.lr_scheduler, optimizer, **configs.lr_scheduler_kwargs)
        criterion = get_criterion(configs.criterion, **configs.criterion_kwargs)

        # initialize the early_stopping object
        early_stopping = EarlyStopping(patience=configs.patience,
                                    path=os.path.join(folder, f'model_{iseed+1}', 'model.pth'),
                                    verbose=True, trace_func=logger.debug)
        t_total = time.time()

        train_loss = []
        val_loss = []
        lowest_error = np.Inf
        for epoch in range(configs.epochs):
            # Go through training stage
            train_error = train(model, data.to(device), optimizer, criterion, epoch, configs)
            train_loss.append(train_error)

            # Check for validation (if requested)
            val_error = validation(model, data.to(device), criterion, configs)
            val_loss.append(val_error)

            if configs.early_stopping:
                early_stopping(val_error, model)

            if val_error < lowest_error:
                logger.info(color_text('Saving best model...', color='yellow', style='bold'))
                torch.save(model.state_dict(), os.path.join(folder, f'model_{iseed+1}', 'best_model.pth'))
                lowest_error = val_error
                test_loss, test_mae_error, \
                test_mape_error, output = test(model, data.to(device), criterion, configs)

            # Update learning rate
            if configs.lr_scheduler == 'plateau':
                    scheduler.step(val_error)
            else:
                scheduler.step()

            if early_stopping.early_stop:
                logger.debug(color_text('Early stopping...', color='gray', style='bold'))
                break

        # Save checkpoint
        if configs.save_checkpoint:
            save_checkpoint(
                os.path.join('checkpoints', f'checkpoint_{configs.run_name}.pth'),
                {'folder' : folder,
                'partition' : iseed}
                )

        logger.info(color_text('Optimization Finished!', color='green', style='bold'))
        logger.info(color_text(f'Total time elapsed: {time.time() - t_total:.4f}s', color='green', style='bold'))

        # denormalizing data
        output = output if dataset.scaler is None else dataset.scaler.inverse_transform(output.T).T

        save_results(os.path.join(folder, f'model_{iseed+1}'), 'errors.mat',
                    {'test_mse_error': test_loss,
                    'test_rmse_error' : np.sqrt(test_loss),
                    'test_mae_error' : test_mae_error,
                    'test_mape_error' : test_mape_error,
                    'train_error': train_loss,
                    'val_error': val_loss}
                    )

        save_results(os.path.join(folder, f'model_{iseed+1}'), 'dataset_attrs.mat', 
                {'sampling_patterns' : data.cpu().sampling_pattern.numpy(),
                'train_index' : data.cpu().train_idx.numpy(),
                'val_index' : data.cpu().val_idx.numpy(),
                'test_index' : data.cpu().test_idx.numpy(),
                'output': output}
                )

    # Remove checkpoint file
    logger.info(color_text('Removing checkpoint...', color='white', style='bold'))
    os.remove(os.path.join('checkpoints', f'checkpoint_{configs.run_name}.pth'))

    logger.info(color_text('Training complete!', color='green', style='bold'))


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