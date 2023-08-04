from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch import nn, Tensor
from typing import Optional
import torch


class Biaffine(nn.Module):

    """
    Биаффинный модуль
    На вход получает данные размерности (batch_size, seq_max_len, input_dim) - результаты применения BiLSTM 
    (forward hidden state & backward hidden state)
    Модуль вычисляет матрицу зависимостей для каждого токена с каждым, возвращает усредненные вектора 
    биаффинного внимания размерности (batch_size, seq_max_len, dep_vec_dim)

    Attributes
    ----------
    input_dim : int, required.
        Размерность последовательности.
    dep_vec_dim : int, required.
        Размерность вектора зависимости между словами в матрице.
    """

    def __init__(self, input_dim, dep_vec_dim):
        super().__init__()

        self.input_dim = input_dim
        self.dep_vec_dim = dep_vec_dim

        self.U_1 = Parameter(torch.Tensor(input_dim, dep_vec_dim, input_dim))
        self.U_2 = Parameter(torch.Tensor(2*input_dim, dep_vec_dim))
        self.bias = Parameter(torch.zeros(dep_vec_dim))

        nn.init.xavier_uniform_(self.U_1)
        nn.init.xavier_uniform_(self.U_2)
        nn.init.constant_(self.bias, 0.)

    def forward(self, h_forward, h_backward):

        seq_len = h_forward.shape[1]
        batch_size = h_forward.shape[0]

        #Hf.T*U1*Hb # U1 - h*r*h, h - Hf/Hb dim, r - dep_vec_dim
        # batch x seq_len x seq_len x dep_vec_dim
        left_part= torch.einsum('bxi,irj,byj->bxyr', h_forward, self.U_1, h_backward)
        
        # (Hf⊕Hb).T*U2 # U2 - 2h*r
        hf = torch.unsqueeze(h_forward, dim=2)
        hf = torch.tile(hf, (1, 1, h_backward.shape[-2], 1))
        hb = torch.unsqueeze(h_backward, dim=1)
        hb = torch.tile(hb, (1, h_forward.shape[-2], 1, 1))

        concat_h = torch.concat((hf, hb), dim=-1)
        right_part = torch.einsum("bxyd,do->bxyo", concat_h, self.U_2)

        # batch x seq_len x seq_len x dep_vec_dim
        biaff_matrix = left_part + right_part + self.bias

        # batch x seq_len x 1 x dep_vec_dim
        dep_vectors = nn.AvgPool3d((1, seq_len, 1))(biaff_matrix)
        # batch x seq_len x dep_vec_dim
        dep_vectors = dep_vectors.view(batch_size, seq_len, self.dep_vec_dim)        

        return dep_vectors
    

class MultiHeadAttention(nn.Module):
    """
    Модуль многоголового внимания
    """
    def __init__(self, n_dim: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.query_weight = nn.Linear(n_dim, n_dim)
        self.key_weight = nn.Linear(n_dim, n_dim, bias=False)
        self.value_weight = nn.Linear(n_dim, n_dim)
        self.linear = nn.Linear(n_dim, n_dim)

    def forward(
        self,
        sequence: Tensor,
        mask: Optional[Tensor]=None
    ):
        query = self.query_weight(sequence)
        key = self.key_weight(sequence)
        value = self.value_weight(sequence)

        wv, qk = self.compute_attention(query, key, value, mask)
        return self.linear(wv), qk

    def compute_attention(
        self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None
    ):
        n_batch, n_ctx, n_state = q.shape
        scale = (n_state // self.n_heads) ** -0.25
        q = q.view(*q.shape[:2], self.n_heads, -1).permute(0, 2, 1, 3) * scale
        k = k.view(*k.shape[:2], self.n_heads, -1).permute(0, 2, 3, 1) * scale
        v = v.view(*v.shape[:2], self.n_heads, -1).permute(0, 2, 1, 3)

        qk = q @ k
        if mask is not None:
            qk = qk + mask[:n_ctx, :n_ctx]
        qk = qk.float()

        w = F.softmax(qk, dim=-1).to(q.dtype)
        return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2), qk.detach()     
