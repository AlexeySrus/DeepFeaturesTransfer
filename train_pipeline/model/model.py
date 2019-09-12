import torch
import tqdm
import os
import re
from functools import reduce
import torch.nn.functional as F
from train_pipeline.utils.losses import l2
from train_pipeline.utils.losses import acc as acc_function
from train_pipeline.utils.tensor_utils import flatten
from train_pipeline.utils.tensor_utils import crop_batch_by_center


def add_prefix(path, pref):
    """
    Add prefix to file in path
    Args:
        path: path to file
        pref: prefixvectors2line
    Returns:
        path to file with named with prefix
    """
    splitted_path = list(os.path.split(path))
    splitted_path[-1] = pref + splitted_path[-1]
    return reduce(lambda x, y: x + '/' + y, splitted_path)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


class Model:
    def __init__(self, rgb_net, edge_net, _device='cpu', callbacks_list=None):
        self.device = torch.device('cpu' if _device == 'cpu' else 'cuda')
        self.rgb_model = rgb_net.to(self.device)
        self.edge_model = edge_net.to(self.device)
        self.callbacks = [] if callbacks_list is None else callbacks_list
        self.last_n = 0
        self.last_optimiser1_state = None
        self.last_optimiser2_state = None

    def fit(self,
            train_loader,
            optimizer1,
            optimizer2,
            epochs=1,
            loss_function=l2,
            validation_loader=None,
            verbose=False,
            init_start_epoch=1,
            acc_f=acc_function,
            is_epoch_scheduler=True):
        """
        Model train method
        Args:
            train_loader: DataLoader
            optimizer: optimizer from torch.optim with initialized parameters
            or tuple of (optimizer, scheduler)
            epochs: epochs count
            loss_function: Loss function
            validation_loader: DataLoader
            verbose: print evaluate validation prediction
            init_start_epoch: start epochs number
        Returns:
        """
        scheduler = None
        if type(optimizer1) is tuple:
            scheduler = optimizer1[1]
            optimizer1 = optimizer1[0]
            optimizer2 = optimizer1[0]

        for epoch in range(init_start_epoch, epochs + 1):
            self.rgb_model.train()
            self.edge_model.train()

            batches_count = len(train_loader)
            avg_epoch_loss = 0
            avg_epoch_acc = 0

            if scheduler is not None and is_epoch_scheduler:
                scheduler.step(epoch)

            self.last_n = epoch

            test_loss_ = 0

            with tqdm.tqdm(total=batches_count) as pbar:
                for i, batch in enumerate(train_loader):
                    self.last_optimiser1_state = optimizer1.state_dict()
                    self.last_optimiser2_state = optimizer2.state_dict()

                    rgb_x, edge_x = batch
                    rgb_x = rgb_x.to(self.device)
                    edge_x = edge_x.to(self.device)

                    optimizer1.zero_grad()
                    optimizer2.zero_grad()
                    rgb_y_pred, rgb_f = self.rgb_model(rgb_x)
                    edge_y_pred, edge_f = self.edge_model(edge_x)

                    features_loss = F.l1_loss(rgb_f, edge_f)

                    rgb_loss = loss_function(
                        rgb_y_pred, rgb_x
                    ) + features_loss
                    edge_loss = loss_function(
                        edge_y_pred, edge_x
                    ) + features_loss

                    rgb_loss.backward(retain_graph=True)
                    edge_loss.backward()
                    optimizer1.step()
                    optimizer2.step()

                    acc = 0
                    # acc = acc_f(
                    #     flatten(y_pred),
                    #     flatten(crop_batch_by_center(y_true, y_pred.shape))
                    # )

                    pbar.postfix = \
                        'Epoch: {}/{}, loss: {:.8f}, acc: {:.8f}, lr: {:.8f}'.format(
                            epoch,
                            epochs,
                            (rgb_loss.item() + edge_loss.item()) / train_loader.batch_size,
                            acc,
                            get_lr(optimizer1)
                        )
                    avg_epoch_loss += \
                        (rgb_loss.item() + edge_loss.item()) / train_loader.batch_size / batches_count

                    avg_epoch_acc += acc

                    for cb in self.callbacks:
                        cb.per_batch({
                            'model': self,
                            'rgb_loss': rgb_loss.item() / train_loader.batch_size,
                            'edge_loss': edge_loss.item() / train_loader.batch_size,
                            'features_loss': features_loss.item() / train_loader.batch_size,
                            'n': (epoch - 1)*batches_count + i + 1,
                            'rgb_x': rgb_x.detach(),
                            'edge_x': edge_x.detach(),
                            'rgb_y_pred': rgb_y_pred.detach(),
                            'edge_y_pred': edge_y_pred.detach(),
                            'edge_after_rgb': self.edge_model.decode(
                                rgb_f.detach()
                            ).detach(),
                            'rgb_after_edge': self.rgb_model.decode(
                                rgb_f.detach()
                            ).detach(),
                            'acc': acc
                        })

                    pbar.update(1)

            test_loss = None
            test_acc = None

            for cb in self.callbacks:
                cb.per_epoch({
                    'model': self,
                    'loss': avg_epoch_loss,
                    'val loss': test_loss_,
                    'n': epoch,
                    'optimize_state1': optimizer1.state_dict(),
                    'optimize_state2': optimizer2.state_dict(),
                    'acc': avg_epoch_acc,
                    'val acc': test_acc
                })

            if scheduler is not None and not is_epoch_scheduler:
                scheduler.step(avg_epoch_loss)

    def evaluate(self,
                 test_loader,
                 loss_function=l2,
                 verbose=False,
                 acc_f=acc_function):
        """
        Test model
        Args:
            test_loader: DataLoader
            loss_function: loss function
            verbose: print progress

        Returns:

        """
        # self.model.eval()
        #
        # test_loss = 0
        # test_acc = 0
        #
        # with torch.no_grad():
        #     set_range = tqdm.tqdm(test_loader) if verbose else test_loader
        #     for _x, _y_true in set_range:
        #         x = _x.to(self.device)
        #         y_true = _y_true.to(self.device)
        #         y_pred = self.model(x)
        #         test_loss += loss_function(
        #             y_pred, y_true
        #         ).item() / test_loader.batch_size / len(test_loader)
        #         test_acc += \
        #             acc_f(y_pred, y_true).detach().numpy() / len(test_loader)
        #
        # return test_loss, test_acc

    def predict(self,
                image,
                window_size=224,
                verbose=False):
        """
        Predict by cv2 frame (numpy uint8 array)
        Args:
            image: image in numpy uint8 RGB format
            window_size: window size
            verbose: print prediction progress

        Returns:

        """

        pass

    def set_callbacks(self, callbacks_list):
        self.callbacks = callbacks_list

    def save(self, path):
        torch.save(
            self.rgb_model.cpu().state_dict(),
            add_prefix(path, 'rgb_')
        )
        self.rgb_model = self.rgb_model.to(self.device)

        torch.save(
            self.edge_model.cpu().state_dict(),
            add_prefix(path, 'edge_')
        )
        self.edge_model = self.edge_model.to(self.device)

    def load(self, path):
        self.rgb_model.load_state_dict(
            torch.load(
                add_prefix(path, 'rgb_'),
                map_location='cpu'
            )['model_state']
        )
        self.edge_model.load_state_dict(
            torch.load(
                add_prefix(path, 'edge_'),
                map_location='cpu'
            )['model_state']
        )

        self.rgb_model.eval()
        self.rgb_model = self.rgb_model.to(self.device)

        self.edge_model.eval()
        self.edge_model = self.edge_model.to(self.device)

    def __del__(self):
        for cb in self.callbacks:
            cb.early_stopping(
                {
                    'model': self,
                    'n': self.last_n,
                    'optimize_state1': self.last_optimiser1_state,
                    'optimize_state2': self.last_optimiser2_state
                }
            )


def get_last_epoch_weights_path(checkpoints_dir, log=None):
    """
    Get last epochs weights from target folder
    Args:
        checkpoints_dir: target folder
        log: logging, default standard print
    Returns:
        (
            path to current weights file,
            path to current optimiser file,
            current epoch number
        )
    """
    if not os.path.isdir(checkpoints_dir):
        os.makedirs(checkpoints_dir)
        return None, None, 0

    weights_files_list = [
        matching_f.group()
        for matching_f in map(
            lambda x: re.match('model-\d+.trh', x),
            os.listdir(checkpoints_dir)
        ) if matching_f if not None
    ]

    if len(weights_files_list) == 0:
        return None, None, 0

    weights_files_list.sort(key=lambda x: -int(x.split('-')[1].split('.')[0]))

    if log is not None:
        log('LOAD MODEL PATH: {}'.format(
            os.path.join(checkpoints_dir, weights_files_list[0])
        ))

    n = int(
        weights_files_list[0].split('-')[1].split('.')[0]
    )

    return os.path.join(checkpoints_dir,
                        weights_files_list[0]
                        ), \
           os.path.join(checkpoints_dir, 'optimize_state-{}.trh'.format(n)), n