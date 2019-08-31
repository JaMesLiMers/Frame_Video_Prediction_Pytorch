import torch
import torch.nn as nn
from model.convRNN import ConvGRUCell, ConvLSTMCell

class Custom(nn.Module):
    def __init__(self, cfg):
        """
        需要定义的东西都在cfg里面, cfg为传入的一个字典
        """
        super(Custom, self).__init__()
        self.input_size=cfg['input_size']
        self.hidden_dim=cfg['hidden_dim']
        self.input_dim=cfg['input_dim']

        self.convlstm_1 = ConvLSTMCell(input_size=self.input_size,
                                       input_dim=self.input_dim,
                                       hidden_dim=self.hidden_dim,
                                       kernel_size=(3, 3),
                                       bias=True)

        self.convlstm_2 = ConvLSTMCell(input_size=self.input_size,
                                       input_dim=self.hidden_dim,
                                       hidden_dim=self.hidden_dim,
                                       kernel_size=(3, 3),
                                       bias=True)

        self.conv2d = nn.Conv2d(in_channels=self.hidden_dim,
                                out_channels=self.input_dim,
                                bias=False,
                                kernel_size=(3, 3),
                                padding=1)

        self.conv3d = nn.Conv3d(in_channels=self.hidden_dim,
                                out_channels=self.input_dim,
                                bias=False,
                                kernel_size=(3, 3, 3),
                                padding=1)
                

    def forward(self, input, hidden=None, future=0):
        """
        input: (b,t,c,h,w)
        hidden: hidden of last time (b, c_hidden, h, w)
        future: number of future frame to predict
        """
        # Init hidden
        if hidden is None:
            h_t, c_t = self.convlstm_1.init_hidden(input.size(0))
            h_t2, c_t2 = self.convlstm_1.init_hidden(input.size(0))
        else:
            # TODO: 写hidden的处理机制
            raise NotImplementedError

        outputs = []

        seq_len = input.size(1)

        for t in range(seq_len):

            h_t, c_t = self.convlstm_1(input_tensor=input[:,t,:,:,:],
                                       cur_state=[h_t, c_t])
            
            h_t2, c_t2 = self.convlstm_2(input_tensor=h_t,
                                         cur_state=[h_t2, c_t2])
        
            output = self.conv2d(h_t2)
            output = nn.Sigmoid()(output)
            outputs += [output]

        for i in range(future):

            h_t, c_t = self.convlstm_1(input_tensor=output,
                                       cur_state=[h_t, c_t])
            
            h_t2, c_t2 = self.convlstm_2(input_tensor=h_t,
                                         cur_state=[h_t2, c_t2])
                        
            output = self.conv2d(h_t2)
            output = nn.Sigmoid()(output)
            outputs += [output]
        
        outputs = torch.stack(outputs, 1)

        return outputs

