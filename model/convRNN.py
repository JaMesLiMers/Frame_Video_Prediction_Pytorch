import torch.nn as nn
from torch.autograd import Variable
import torch
import torch.nn.functional as F


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

class ConvLSTMCell(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.
        
        Parameters
        ----------
        input_size: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        # init parameters
        self.height, self.width = input_size                            # init hight and width
        self.input_dim  = input_dim                                     # init input dim
        self.hidden_dim = hidden_dim                                    # init output dim

        self.kernel_size = kernel_size                                  # init kernel size
        self.padding     = kernel_size[0] // 2, kernel_size[1] // 2     # caculate padding size automatically
        self.bias        = bias                                         # init bias

        # all gate conv operation can be include in a big conv layer
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

        # init parameters
        nn.init.orthogonal(self.conv.weight)
        if self.bias is True:
            nn.init.ones_(self.conv.bias)

    def forward(self, input_tensor, cur_state):
        """
        Forward of Conv LSTM Cell
        Inputs:
        ---------------------------------------
        input_tensor: (b, c, h, w)
        cur_state: [(b, c_hidden, h, w), (b, c_hidden, h, w)]
        ---------------------------------------
        Returns:
        ---------------------------------------
        h_next, c_next :  ((b, c_hidden, h, w), (b, c_hidden, h, w))
        next hidden state
        """
        
        h_cur, c_cur = cur_state
        
        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis
        
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1) 
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = F.relu(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next

    def init_hidden(self, batch_size):
        return (torch.zeros(batch_size, self.hidden_dim, self.height, self.width).to(device),
                torch.zeros(batch_size, self.hidden_dim, self.height, self.width).to(device))
        

class ConvGRUCell(nn.Module):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize the ConvLSTM cell
        :param input_size: (int, int)
            Height and width of input tensor as (height, width).
        :param input_dim: int
            Number of channels of input tensor.
        :param hidden_dim: int
            Number of channels of hidden state.
        :param kernel_size: (int, int)
            Size of the convolutional kernel.
        :param bias: bool
            Whether or not to add the bias.
        """
        super(ConvGRUCell, self).__init__()

        # init all parameter
        self.height, self.width = input_size                            # init hight and width
        self.input_dim  = input_dim                                     # init input dim
        self.hidden_dim = hidden_dim                                    # init output dim

        self.kernel_size = kernel_size                                  # init kernel size
        self.padding     = kernel_size[0] // 2, kernel_size[1] // 2     # caculate padding size automatically
        self.bias        = bias                                         # init bias

        # all gate conv operation can be include in a big conv layer
        self.conv_gates = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                                    out_channels=2*self.hidden_dim,  # for update_gate,reset_gate respectively
                                    kernel_size=self.kernel_size,
                                    padding=self.padding,
                                    bias=self.bias)

        self.conv_can = nn.Conv2d(in_channels=self.input_dim+self.hidden_dim,
                              out_channels=self.hidden_dim, # for candidate neural memory
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

        # init conv parameter and bias parameter
        nn.init.orthogonal(self.conv_can.weight)
        nn.init.orthogonal(self.conv_gates.weight)
        if self.bias is True:
            nn.init.ones_(self.conv_can.bias)
            nn.init.ones_(self.conv_gates.bias)

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_dim, self.height, self.width).to(device)

    def forward(self, input_tensor, cur_state):
        """
        Forward of Conv GRU Cell
        Inputs:
        ---------------------------------------
        input_tensor: (b, c, h, w)
        cur_state: (b, c_hidden, h, w)
        ---------------------------------------
        Returns:
        ---------------------------------------
        h_next : ((b, c_hidden, h, w))
        next hidden state
        """
        combined = torch.cat([input_tensor, cur_state], dim=1)
        combined_conv = self.conv_gates(combined)

        gamma, beta = torch.split(combined_conv, self.hidden_dim, dim=1)
        reset_gate = torch.sigmoid(gamma)
        update_gate = torch.sigmoid(beta)

        combined = torch.cat([input_tensor, reset_gate*cur_state], dim=1)
        cc_cnm = self.conv_can(combined)
        cnm = F.tanh(cc_cnm)

        h_next = (1 - update_gate) * cur_state + update_gate * cnm
        return h_next


def test():
    pass
    


if __name__ == "__main__":
    test()

