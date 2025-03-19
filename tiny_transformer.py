from torch.nn import functional as F
import numpy as np
import math
import torch
import torch.nn as nn
import inspect 


'''注意力计算函数
TODO: dropout_module 和 dropout 以及 mask 的作用，方便下面multiheadattention class 中正确调用
'''

def attention(q, k, v, dropout_module = None, is_causal = False):
    # q k v: (batch_size, nh, L, d_k)
    d_k = q.size(-1)
    att = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(d_k)
    if is_causal:
        causal_mask = torch.triu(torch.ones(att.size(-2), att.size(-1)), 1).to(att.device)  # L * L 的上三角矩阵作为mask(上三角不含对角线为1)
        att = att.masked_fill(causal_mask == 1, float('-inf'))   # mask中等于1的位置填充为负无穷
    att = F.softmax(att, dim = -1)  # 在行方向上（针对每个token）进行softmax
    # attention dropout
    att = dropout_module(att) if dropout_module is not None else att
    # att * v: (batch_size, nh, L, L) * (batch_size, nh, L, d_k) = (batch_size, nh, L, d_k)
    output = torch.matmul(att, v)
    return output

'''多头注意力计算模块'''
class MultiHeadAttention(nn.Module):

    def __init__(self, config, is_causal = False):
        super().__init__()
        # 隐藏层维度须是注意力头数的整数倍
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.d_k = config.n_embd // config.n_head
        self.is_causal = is_causal
        self.dropout = nn.Dropout(config.dropout)
        # 线性变换层
        # 定义三个权重矩阵Wq Wk Wv
        # 先一次性计算所有head的QKV 再进行分割
        self.q_linear = nn.Linear(config.n_embd, config.n_embd)    # (batch_size, L, n_embd) * Wq -> (batch_size, L, n_embd)
        self.k_linear = nn.Linear(config.n_embd, config.n_embd)
        self.v_linear = nn.Linear(config.n_embd, config.n_embd)
        self.out_linear = nn.Linear(config.n_embd, config.n_embd)

    def forward(self, q, k, v, mask = None):
        # q k v: (batch_size, L, n_embd)
        batch_size = q.size(0)
        # 线性变换
        q = self.q_linear(q).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        k = self.k_linear(k).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        # attention
        att = attention(q, k, v, self.dropout, self.is_causal)
        # 拼接多头注意力
        att = att.transpose(1, 2).contiguous().view(batch_size, -1, self.n_head * self.d_k)
        # 线性变换
        output = self.out_linear(att)
        return output
    
'''前馈神经网络（全连接层）模块'''
class FeedForward(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.n_embd, config.n_embd * 4, bias = config.bias)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(config.n_embd * 4, config.n_embd, bias = config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # x: (batch_size, L, n_embd)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x
    
'''LayerNorm层'''
class LayerNorm(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(config.n_embd))
        self.beta = nn.Parameter(torch.zeros(config.n_embd)) if config.bias else None

    def forward(self, x):
        # x: (batch_size, L, n_embd)
        return F.layer_norm(x, self.gamma.shape, self.gamma, self.beta, 1e-5)

'''Encoder Layer'''
class EncoderLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.norm1 = LayerNorm(config)
        self.attention = MultiHeadAttention(config, is_causal = False)
        self.norm2 = LayerNorm(config)
        self.ffn = FeedForward(config)

    def forward(self, x, mask = None):
        # x: (batch_size, L, n_embd)
        x = self.norm1(x)
        # 残差连接：让输入 x 直接加到后续层的输出上，避免网络层数过深时造成梯度消失或梯度爆炸问题
        x = x + self.attention(x, x, x, mask)
        x = x + self.ffn(self.norm2(x))
        return x

'''Encoder'''
class Encoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.n_layer)])
        self.norm = LayerNorm(config)

    def forward(self, x):
        # x: (batch_size, L, n_embd)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)
    

'''Decoder Layer'''
class DecoderLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        # 一个 Layer 中有三个 LayerNorm，分别在 Mask Attention 之前、Self Attention 之前和 FFN 之前
        self.norm1 = LayerNorm(config)
        self.m_attn = MultiHeadAttention(config, is_causal = True)
        self.norm2 = LayerNorm(config)
        self.attn = MultiHeadAttention(config, is_causal = False)
        self.norm3 = LayerNorm(config)
        self.ffn = FeedForward(config)

    def forward(self, x, enc_out):
        x = self.norm1(x)
        # 第一部分是一个 Mask Self Attention，Q、K、V 都是 x
        x = x + self.m_attn(x, x, x)
        x = self.norm2(x)
        # 第二部分是一个 Encoder-Decoder Attention，Q 是 x，K、V 是 Encoder 的输出
        x = x + self.attn(x, enc_out, enc_out)
        x = self.norm3(x)
        x = x + self.ffn(x)
        return x
    
'''Decoder'''
class Decoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.n_layer)])
        self.norm = LayerNorm(config)

    def forward(self, x, enc_out):
        for layer in self.layers:
            x = layer(x, enc_out)
        return self.norm(x)
    
'''Positional Encoding'''
class PositionalEncoding(nn.Module):

    def __init__(self, config):
        super().__init__()
        # Dropout 层
        self.dropout = nn.Dropout(p = config.dropout)

         # block size 是序列的最大长度, PE shape: L * n_embd
        pe = torch.zeros(config.block_size, config.n_embd)
        position = torch.arange(0, config.block_size).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, config.n_embd, 2) * -(math.log(10000.0) / config.n_embd)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (batch_size, L, n_embd)
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)
    
'''Transformer'''
class Transformer(nn.Module):

    def __init__(self, config):
        super().__init__()
        # 必须输入词表大小和 block size
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = PositionalEncoding(config),
            drop = nn.Dropout(config.dropout),
            encoder = Encoder(config),
            decoder = Decoder(config),
        ))
        # 最后的线性层，输入是 n_embd，输出是词表大小
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # 初始化所有的权重
        self.apply(self._init_weights)

        # 查看所有参数的数量
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    '''统计所有参数的数量'''
    def get_num_params(self, non_embedding=False):
        # non_embedding: 是否统计 embedding 的参数
        n_params = sum(p.numel() for p in self.parameters())
        # 如果不统计 embedding 的参数，就减去
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params
    
    '''初始化权重'''
    def _init_weights(self, module):
        # 线性层和 Embedding 层初始化为正态分布
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    '''前向计算函数'''
    def forward(self, idx, targets=None):
        # 输入为 idx，维度为 (batch size, sequence length)；targets 为目标序列，用于计算 loss
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"不能计算该序列，该序列长度为 {t}, 最大序列长度只有 {self.config.block_size}"

        # 通过 self.transformer
        # 首先将输入 idx 通过 Embedding 层，得到维度为 (batch size, sequence length, n_embd)
        print("idx",idx.size())
        # 通过 Embedding 层得到的维度是 (batch size, sequence length, vocab_size, n_embd)，因此我们去掉倒数第二个维度
        tok_emb = self.transformer.wte(idx)
        print("tok_emb",tok_emb.size())
        # 然后通过位置编码
        pos_emb = self.transformer.wpe(tok_emb) 
        # 再进行 Dropout
        x = self.transformer.drop(pos_emb)
        # 然后通过 Encoder
        print("x after wpe:",x.size())
        enc_out = self.transformer.encoder(x)
        print("enc_out:",enc_out.size())
        # 再通过 Decoder
        x = self.transformer.decoder(x, enc_out)
        print("x after decoder:",x.size())

        if targets is not None:
            # 训练阶段，如果我们给了 targets，就计算 loss
            # 先通过最后的 Linear 层，得到维度为 (batch size, sequence length, vocab size)
            logits = self.lm_head(x)
            # 再跟 targets 计算交叉熵
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # 推理阶段，我们只需要 logits，loss 为 None
            # 取 -1 是只取序列中的最后一个作为输出
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss
    
    '''配置优化器'''
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # weight_decay: 权重衰减系数，learning_rate: 学习率，betas: AdamW 的 betas，device_type: 设备类型
        # 首先获取所有命名参数
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # 过滤掉不需要更新的参数
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # 参数根据维度分为两组。
        # 维度大于等于2的参数（通常是权重）会应用权重衰减，而维度小于2的参数（通常是偏置和层归一化参数）不会应用权重衰减。
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        # 打印一下参数数量
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"应用权重衰减的层数: {len(decay_params)}； 总参数量为：{num_decay_params:,}")
        print(f"不应用权重衰减的层数: {len(nodecay_params)}, 总参数量为：{num_nodecay_params:,}")
        # 检查 torch.optim.AdamW 是否支持融合版本（fused version），这是针对 CUDA 设备优化的版本。如果可用且 device_type 为 'cuda'，则使用融合版本。
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        # 创建优化器
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"是否使用 fused AdamW: {use_fused}")

        return optimizer
    
    '''进行推理'''
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        # 推理阶段，输入为 idx，维度为 (batch size, sequence length)，max_new_tokens 为最大生成的 token 数量即按序推理 max_new_tokens 次
        for _ in range(max_new_tokens):
            # 如果输入序列太长，我们需要将它截断到 block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # 前向计算，得到 logits，维度为 (batch size, sequence length, vocab size)
            logits, _ = self(idx_cond)
            # 使用最后一个 token 的 logits 作为当前输出，除以温度系数控制其多样性
            logits = logits[:, -1, :] / temperature
            # 如果使用 Top K 采样，将 logits 中除了 top_k 个元素的概率置为 0
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # 对输出结果进行 Softmax
            probs = F.softmax(logits, dim=-1)
            # 对结果概率进行采样
            idx_next = torch.multinomial(probs, num_samples=1)
            # 将输出结果拼接到输入序列后面，作为下一次的输入
            idx = torch.cat((idx, idx_next), dim=1)
            # print("idx:", idx)

        return idx
