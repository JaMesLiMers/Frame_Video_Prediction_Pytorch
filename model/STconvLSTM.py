import torch.nn as nn
import torch
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

class STConvLSTMCell(nn.Module):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias, forget_bias=1.0, layer_norm=True):
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
        m_dim: int
            Number of channels of M state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(STConvLSTMCell, self).__init__()

        # init parameters
        self.height, self.width = input_size                            # 初始化高和宽
        self.input_dim  = input_dim                                     # 初始化输入的维度
        self.hidden_dim = hidden_dim                                    # 初始化输出的维度

        self.kernel_size = kernel_size                                  # 初始化核的大小
        self.padding     = kernel_size[0] // 2, kernel_size[1] // 2     # 自动算padding的大小
        self.bias        = bias                
        self.forget_bias = forget_bias
        self.layer_norm  = layer_norm



        # split the conv gate layers
        # for W * X_t
        self.conv_wx = nn.Conv2d(in_channels=self.input_dim,
                                out_channels=7 * self.hidden_dim,
                                kernel_size=self.kernel_size,
                                padding=self.padding,
                                bias=self.bias)

        # for W * H^t_1
        self.conv_wht_1 = nn.Conv2d(in_channels=self.hidden_dim,
                                out_channels=4 * self.hidden_dim,
                                kernel_size=self.kernel_size,
                                padding=self.padding,
                                bias=self.bias)
                                
        # for W * M^l_1
        self.conv_wml_1 = nn.Conv2d(in_channels=self.hidden_dim,
                                out_channels=3 * self.hidden_dim,
                                kernel_size=self.kernel_size,
                                padding=self.padding,
                                bias=self.bias)

        # for W * M^l
        self.conv_wml = nn.Conv2d(in_channels=self.hidden_dim,
                                out_channels= self.hidden_dim,
                                kernel_size=self.kernel_size,
                                padding=self.padding,
                                bias=self.bias)

        # for W * C^l
        self.conv_wcl = nn.Conv2d(in_channels=self.hidden_dim,
                                out_channels=self.hidden_dim,
                                kernel_size=self.kernel_size,
                                padding=self.padding,
                                bias=self.bias)
                                
        
        # for generate H^l
        self.conv_h = nn.Conv2d(in_channels=self.hidden_dim + self.hidden_dim,
                                out_channels=self.hidden_dim,
                                kernel_size=[1,1],
                                padding=0,
                                bias=self.bias)

        # init parameters
        nn.init.orthogonal(self.conv_wx.weight)
        nn.init.orthogonal(self.conv_wht_1.weight)
        nn.init.orthogonal(self.conv_wml_1.weight)
        nn.init.orthogonal(self.conv_wml.weight)
        nn.init.orthogonal(self.conv_wcl.weight)
        nn.init.orthogonal(self.conv_h.weight)

        # for layerNorm
        if self.layer_norm:
            self.conv_wx_norm = nn.BatchNorm2d(7 * self.hidden_dim)
            self.conv_wht_1_norm = nn.BatchNorm2d(4 * self.hidden_dim)
            self.conv_wml_1_norm = nn.BatchNorm2d(3 * self.hidden_dim)
            self.conv_wml_norm = nn.BatchNorm2d(self.hidden_dim)
            self.conv_wcl_norm = nn.BatchNorm2d(self.hidden_dim)
            self.conv_h_norm = nn.BatchNorm2d(self.hidden_dim)


        # for bias
        self.forget_bias_h = torch.nn.Parameter(torch.tensor(self.forget_bias))
        self.forget_bias_m = torch.nn.Parameter(torch.tensor(self.forget_bias))
    
    def forward(self, input_tensor, cur_state):
        """
        Forward of Conv LSTM Cell
        Inputs:
        ---------------------------------------
        input_tensor: (b, c, h, w)
        cur_state: [           H,                  C,                 M          ]
        cur_state: [(b, c_hidden, h, w), (b, c_hidden, h, w), (b, c_hidden, h, w)]
        ---------------------------------------
        Returns:
        ---------------------------------------
        h_next, c_next, m_next :  ((b, c_hidden, h, w), (b, c_hidden, h, w), (b, c_hidden, h, w))
        next hidden state
        """
        # state input
        h_cur, c_cur, m_cur = cur_state
        # conv gate result
        conved_wx = self.conv_wx(input_tensor)
        conved_wht_1 = self.conv_wht_1(h_cur)
        conved_wml_1 = self.conv_wml_1(m_cur)
        # for bn
        if self.layer_norm:
            conved_wx = self.conv_wx_norm(conved_wx)
            conved_wht_1 = self.conv_wht_1_norm(conved_wht_1)
            conved_wml_1 = self.conv_wml_1_norm(conved_wml_1)
        # split gate result
        wxg, wxi, wxf, wxg_, wxi_, wxf_, wxo = torch.split(conved_wx, self.hidden_dim, dim=1)
        whg, whi, whf, who = torch.split(conved_wht_1, self.hidden_dim, dim=1)
        wmg, wmi, wmf = torch.split(conved_wml_1, self.hidden_dim, dim=1)
        # for c_next
        g_t = torch.tanh(wxg + whg)
        i_t = torch.sigmoid(wxi + whi)
        f_t = torch.sigmoid(wxf + whf + self.forget_bias_h)
        c_next = f_t * c_cur + i_t * g_t
        # for m_next
        g_t_ = torch.tanh(wxg_ + wmg)
        i_t_ = torch.sigmoid(wxi_ + wmi)
        f_t_ = torch.sigmoid(wxf_ + wmf + self.forget_bias_m)
        m_next = f_t_ * m_cur + i_t_ * g_t_
        # for wco, wmo
        wco = self.conv_wcl(c_next)
        wmo = self.conv_wml(m_next)
        # for bn
        if self.layer_norm:
            wco = self.conv_wcl_norm(wco)
            wmo = self.conv_wml_norm(wmo)
        # for output gate
        o_t = torch.sigmoid(wxo + who + wco + wmo)
        # for h_next
        combined_cmn = torch.cat([c_next, m_next], dim=1)
        h_next = o_t * torch.tanh(self.conv_h(combined_cmn))

        return h_next, c_next, m_next


    def init_hidden(self, batch_size):
        return (torch.zeros(batch_size, self.hidden_dim, self.height, self.width).to(device),
                torch.zeros(batch_size, self.hidden_dim, self.height, self.width).to(device),
                torch.zeros(batch_size, self.hidden_dim, self.height, self.width).to(device))



def test():
    pass



if __name__ == "__main__":
    test()

