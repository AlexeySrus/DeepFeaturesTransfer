import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Module


def l2(y_pred, y_true):
    return ((y_pred - y_true) ** 2).mean()


def acc(y_pred, y_true):
    return (y_pred.argmax(dim=1) == y_true.argmax(dim=1)).sum().type(
        torch.FloatTensor
    ) / y_true.size(0)


class FourierLoss(Module):
    """L2 loss between normed fft2 transforms by L1 normalisation"""

    R = 0.299
    G = 0.587
    B = 0.114

    def __init__(self, base_loss=F.mse_loss,
                 loss_sum_coeffs=(1, 1), four_normalized=True,
                 k=1/125):
        super(FourierLoss, self).__init__()

        self.four_normalized = four_normalized
        self.base_loss = base_loss
        self.coeffs = loss_sum_coeffs
        self.k = k

    def rgb2gray(self, x):
        return x[:, 0] * self.R + x[:, 1] * self.G + x[:, 2] * self.B

    @staticmethod
    def drop_fft_center(x, k=1 / 225):
        x[:, :int(x.size(1) * k), :int(x.size(2) * k)] = 0
        x[:, -int(x.size(1) * k):, :int(x.size(2) * k)] = 0
        # x[:, int(x.size(1) * k):-int(x.size(1) * k),
        #     int(x.size(2) * k):-int(x.size(2) * k)] = 0
        return x

    @staticmethod
    def complex_abs(x):
        return (x[:, :, :, 0] ** 2 + x[:, :, :, 1] ** 2) ** (1/2)

    def forward(self, y_pred, y_true):
        """
        Loss forward
        Args:
            y_pred: batch which contains RGB channels images
            y_true: batch which contains RGB channels images

        Returns:
            L2 loss between normed fft2 transforms by L1 normalisation
        """
        y_pred_gray = self.rgb2gray(y_pred)
        y_true_gray = self.rgb2gray(y_true)

        fourier_transform_pred = torch.rfft(
            y_pred_gray, 2, normalized=self.four_normalized
        )

        fourier_transform_true = torch.rfft(
            y_true_gray, 2, normalized=self.four_normalized
        )

        abs_fourier_transform_pred = self.complex_abs(fourier_transform_pred)
        abs_fourier_transform_true = self.complex_abs(fourier_transform_true)

        n_fourier_transform_pred = self.drop_fft_center(
            abs_fourier_transform_pred
        )

        n_fourier_transform_true = self.drop_fft_center(
            abs_fourier_transform_true
        )

        return self.base_loss(
            y_pred, y_true
        ) * self.coeffs[0] + l2(
            n_fourier_transform_pred, n_fourier_transform_true
        ) * self.coeffs[1]
