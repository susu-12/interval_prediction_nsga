
from scipy.stats import norm
import torch
# 定义 PIVEN 损失函数
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def z_score(alpha):
    score = norm.ppf(1 - alpha / 2)
    return round(score, 3)

def pi_to_gauss(y_pred_all, method, alpha=0.01):
    """
    input is individual NN estimates of upper and lower bounds
    1. combine into ensemble estimates of upper and lower bounds
    2. convert to mean and std dev of gaussian

    @param y_pred_all: predictions. shape [no. ensemble, no. predictions, 2]
                       or [no. ensemble, no. predictions, 3] in case method is piven or only-rmse
    @param method: method name
    """
    in_ddof = 1 if y_pred_all.shape[0] > 1 else 0
    z_score_ = z_score(alpha)

    y_upper_mean, y_upper_std = np.mean(y_pred_all[:, :, 1], axis=0), np.std(y_pred_all[:, :, 1], axis=0, ddof=in_ddof)
    y_lower_mean, y_lower_std = np.mean(y_pred_all[:, :, 0], axis=0), np.std(y_pred_all[:, :, 0], axis=0, ddof=in_ddof)

    y_pred_U = y_upper_mean + z_score_ * y_upper_std / np.sqrt(y_pred_all.shape[0])
    y_pred_L = y_lower_mean - z_score_ * y_lower_std / np.sqrt(y_pred_all.shape[0])

    if method == 'qd' or method == 'mid' or method == 'deep-ens':
        v = None
    elif method == 'piven' or method == 'only-rmse':
        v = np.mean(y_pred_all[:, :, 2], axis=0)
    else:
        raise ValueError(f"Unknown method {method}")

    # need to do this before calc mid and std dev
    y_pred_U_temp = np.maximum(y_pred_U, y_pred_L)
    y_pred_L = np.minimum(y_pred_U, y_pred_L)
    y_pred_U = y_pred_U_temp

    y_pred_gauss_mid = np.mean((y_pred_U, y_pred_L), axis=0)
    y_pred_gauss_dev = (y_pred_U - y_pred_gauss_mid) / z_score_

    return y_pred_gauss_mid, y_pred_gauss_dev, y_pred_U, y_pred_L, v

class LSTMQDPLUS(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=3, dropout=0.1):
        super(LSTMQDPLUS, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)
        # self.fc = nn.Linear(hidden_size * 2, output_size)  # Multiply by 2 for bidirectional LSTM
                # Output layers
        self.pi = nn.Linear(hidden_size * 2, output_size)  # The last dimension is for v

        # Initialization using RandomNormal and Constant
        nn.init.normal_(self.pi.weight, mean=0.0, std=0.2)
        self.pi.bias.data = torch.tensor([1., -1., 0.])

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        out, _ = self.lstm(x)
        out = F.dropout(out, p=self.dropout, training=self.training)  # Apply dropout
        pi = self.pi(out)
        pi = pi[:,0,:]
        return pi

lambda_ = 0.01 # lambda in loss fn
alpha_ = 0.05  # capturing (1-alpha)% of samples
soften_ = 160.
n_ = 32 # batch size

# define loss fn

def penalty_func(y_l: torch.Tensor, y_p: torch.Tensor, y_u: torch.Tensor) -> torch.Tensor:
    m_l = torch.relu(y_l - y_p)
    m_u = torch.relu(y_p - y_u)
    return torch.mean(m_l + m_u)


def k_func(y_l: torch.Tensor, y: torch.Tensor, y_u: torch.Tensor) -> torch.Tensor:
    """
    Returns a boolean mask showing whether item is inside the interval. Using `sign()`.
    """
    k_l = torch.relu(torch.sign(y - y_l))
    k_u = torch.relu(torch.sign(y_u - y))
    k = torch.mul(k_l, k_u)
    return k


def k_func_soft(y_l: torch.Tensor, y: torch.Tensor, y_u: torch.Tensor,
                soften: float) -> torch.Tensor:
    """
    Returns a boolean mask showing whether item is inside the interval. Using `sigmoid()`.
    """
    k_l = torch.sigmoid((y - y_l) * soften)
    k_u = torch.sigmoid((y_u - y) * soften)
    k = torch.mul(k_l, k_u)
    return k
def picp_func(y_l: torch.Tensor, y: torch.Tensor, y_u: torch.Tensor, **kwargs) -> torch.Tensor:
    return torch.mean(k_func(y_l, y, y_u))


# alias
picp = picp_func


def picp_func_soft(y_l: torch.Tensor, y: torch.Tensor, y_u: torch.Tensor,
                   soften: float) -> torch.Tensor:
    return torch.mean(k_func_soft(y_l, y, y_u, soften))


def mpiw_func(y_l: torch.Tensor, y_u: torch.Tensor, **kwargs):
    return torch.mean(y_u - y_l)

def abs_loss_mpiw(y_l: torch.Tensor, y_u: torch.Tensor, k: torch.Tensor,
                  epsilon: float) -> torch.Tensor:
    return torch.div(torch.sum(torch.abs(y_u - y_l) * k), torch.sum(k) + epsilon)

def loss_picp(alpha: float, picp: torch.Tensor) -> torch.Tensor:
    return torch.pow(torch.relu((1. - alpha) - picp), 2)
def qd_plus_objective(y_pred, y_true, alpha, lambda_1, lambda_2, ksi, soften, epsilon=1e-3, **kwargs):
    y_l = y_pred[:, 0]  # lower bound
    y_u = y_pred[:, 1]  # upper bound
    y_p = y_pred[:, 2]  # point prediction
    y_t = y_true[:, 0]  # ground truth

    if soften is not None and soften > 0:
        # Soft: uses `sigmoid()`
        k_ = k_func_soft(y_l, y_t, y_u, soften)
        picp_ = picp_func_soft(y_l, y_t, y_u, soften)
    else:
        # Hard: uses `sign()`
        k_ = k_func(y_l, y_t, y_u)
        picp_ = picp_func(y_l, y_t, y_u)
    loss_mpiw_ = abs_loss_mpiw(y_l, y_u, k_, epsilon)
    loss_picp_ = loss_picp(alpha, picp_)
    mse_ = torch.nn.functional.mse_loss(y_p, y_t)

    loss = (1 - lambda_1) * (1 - lambda_2) * loss_mpiw_ + \
           lambda_1 * (1 - lambda_2) * loss_picp_ + \
           lambda_2 * mse_ + ksi * penalty_func(y_l, y_p, y_u)

    return loss
def eval_plot_qdplus(y_l_pred,y_u_pred,y_v_pred,y_test_tensor,scaler_y_test):
    y = y_test_tensor[:,0]
    n_y_test_tensor = scaler_y_test.inverse_transform(y_test_tensor)
    n_y_u_pred = scaler_y_test.inverse_transform(y_u_pred.reshape(-1, 1))
    n_y_l_pred = scaler_y_test.inverse_transform(y_l_pred.reshape(-1, 1))
    n_y_v_pred = scaler_y_test.inverse_transform(y_v_pred.reshape(-1,1))


    mape_qdplus = np.abs((y- y_v_pred) / y).mean() * 100
    rmse = np.sqrt(mean_squared_error( y_v_pred, y))

    # mae =np.abs(n_y_test_tensor - y_piven).mean()                     
    print('mape的值',mape_qdplus)
    print('rmse的值',rmse)
    # print('mae的值',mae)
    
    K_u = y_u_pred> y_test_tensor[:,0]
    K_l = y_l_pred<y_test_tensor[:,0]
    # K_u = n_y_u_pred > n_y_test_tensor[:,0]
    # K_l = n_y_l_pred <n_y_test_tensor[:,0]
    PICP = np.mean(K_u * K_l)
    MPIW = round(np.mean(y_u_pred - y_l_pred), 3)
    print('PICP:', PICP)
    print('MPIW:', MPIW)
    # Convert y_test_sensor tensor to a numpy array
#         x_dates = X.iloc[split_index_2:,0].values
    # Plot the curves
    plt.figure(figsize=(12, 6))
    plt.plot( n_y_u_pred, label='y_u_pred')
    plt.plot( n_y_l_pred, label='y_l_pred')
    plt.plot(n_y_v_pred, label='y_qdplus')
    # plt.plot(x_dates, n_y_v_pred,label='y_v')
    plt.plot(n_y_test_tensor, label='y_true', linestyle='--', color='black')
    plt.xlabel('Dates')
    plt.ylabel('Prediction')
    plt.legend()
    plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
    plt.tight_layout()
    plt.show()

