import torch
import torch.nn as nn
from model.STconvLSTM import STConvLSTMCell

class Custom(nn.Module):
    def __init__(self, cfg):
        """
        All we need to use is in cfg
        cfg.somepart["par_name"]
        """
        super(Custom, self).__init__()
        self.input_size=cfg['input_size']
        self.hidden_dim=cfg['hidden_dim']
        self.input_dim=cfg['input_dim']
        self.kernel_size=tuple(cfg['kernel_size'])

        self.stlstm_1 = STConvLSTMCell(input_size=self.input_size,
                                       input_dim=self.input_dim,
                                       hidden_dim=self.hidden_dim,
                                       kernel_size=self.kernel_size,
                                       bias=True)

        self.stlstm_2 = STConvLSTMCell(input_size=self.input_size,
                                       input_dim=self.hidden_dim,
                                       hidden_dim=self.hidden_dim,
                                       kernel_size=self.kernel_size,
                                       bias=True)

        self.stlstm_3 = STConvLSTMCell(input_size=self.input_size,
                                       input_dim=self.hidden_dim,
                                       hidden_dim=self.hidden_dim,
                                       kernel_size=self.kernel_size,
                                       bias=True)

        self.stlstm_4 = STConvLSTMCell(input_size=self.input_size,
                                       input_dim=self.hidden_dim,
                                       hidden_dim=self.hidden_dim,
                                       kernel_size=self.kernel_size,
                                       bias=True)

        self.head = nn.Conv2d(in_channels=self.hidden_dim,
                              out_channels=self.input_dim,
                              kernel_size=(1,1),
                              bias=True)

    def forward(self, input, hidden=None, future=10):
        """
        input: (b,t,c,h,w)
        hidden: hidden of last time (b, c_hidden, h, w)
        future: number of future frame to predict
        """
        # Init hidden
        if hidden is None:
            h_t1, c_t1, m_t1 = self.stlstm_1.init_hidden(input.size(0))
            h_t2, c_t2, _ = self.stlstm_2.init_hidden(input.size(0))
            h_t3, c_t3, _ = self.stlstm_3.init_hidden(input.size(0))
            h_t4, c_t4, _ = self.stlstm_4.init_hidden(input.size(0))
        else:
            # TODO: build a stateful model
            raise NotImplementedError

        outputs = []

        seq_len = input.size(1)

        for t in range(seq_len):
            if t is not 0:
                m_t1 = m_t4

            h_t1, c_t1, m_t1 = self.stlstm_1(input_tensor=input[:,t,:,:,:],
                                       cur_state=[h_t1, c_t1, m_t1])
            
            h_t2, c_t2, m_t2 = self.stlstm_2(input_tensor=h_t1,
                                         cur_state=[h_t2, c_t2, m_t1])
            
            h_t3, c_t3, m_t3 = self.stlstm_3(input_tensor=h_t2,
                                         cur_state=[h_t3, c_t3, m_t2])
            
            h_t4, c_t4, m_t4 = self.stlstm_4(input_tensor=h_t3,
                                         cur_state=[h_t4, c_t4, m_t3])

            output = self.head(h_t4)
            outputs += [output]

        
        for t in range(future):
            m_t1 = m_t4

            h_t1, c_t1, m_t1 = self.stlstm_1(input_tensor=outputs[-1],
                                       cur_state=[h_t1, c_t1, m_t1])
            
            h_t2, c_t2, m_t2 = self.stlstm_2(input_tensor=h_t1,
                                         cur_state=[h_t2, c_t2, m_t1])
            
            h_t3, c_t3, m_t3 = self.stlstm_3(input_tensor=h_t2,
                                         cur_state=[h_t3, c_t3, m_t2])
            
            h_t4, c_t4, m_t4 = self.stlstm_4(input_tensor=h_t3,
                                         cur_state=[h_t4, c_t4, m_t3])
        
            output = self.head(h_t4)
            outputs += [output]

        
        outputs = torch.stack(outputs, 1)

        return outputs


        
