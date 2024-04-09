import math
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from typing import List, Dict, Tuple

class ScaleDotProductAttention(nn.Module):
	"""
 	缩放点积注意力
 	"""
    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()

	def forward(self,
				Q: torch.Tensor,
				K: torch.Tensor,
				V: torch.Tensor,
				attn_mask: torch.Tensor,
				Dropout: nn.Dropout) -> Tuple[torch.Tensor, torch.Tensor]:
		"""
		:param Q: [batch_size, n_heads, len_q, d_k]
  		:param K: [batch_size, n_heads, len_k, d_k]
		:param V: [batch_size, n_heads, len_v(=len_k), d_v]
  		:param attn_mask: [bacth_size, n_heads, len_q, len_v]
		:param dropout: nn.Dropout
  		"""
		# scores: [batch_size, n_heads, len_q, len_k]
		scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(Q.size(-1)) # Q.size(-1)==d_k
		if attn_mask:
			scores = scores.masked_fill(attn_mask==1, -1e9)

		# 对scores的最后一维进行softmax操作
		attn = nn.Softmax(dim=-1)(scores)
		if Dropout:
			attn = Dropout(attn)
		# attn: [batch_size, n_heads, len_q, len_k] * V: [batch_size, n_heads, len_v(=len_k), d_v]
		# context: [batch_size, n_heads, len_q, d_v]
		context = torch.matmul(attn, V)

		return context, attn


class MultiHeadAttention(nn.Module):
	"""
	多头注意力网络
 	本class可以实现：
  	Encoder的Self-Attention
   	Decoder的Masked Self-Attention
	Encoder-Decoder的Attention
  	"""
	def __init__(self,
				 n_heads: int,
				 d_k: int,
				 d_v: int,
				 d_model: int,
				 dropout=None: float):
		"""
  		:param n_heads:多头注意力的头数
		:param d_k: 注意力中Q和K的嵌入维度，Q和K的嵌入维度必须相等，不然Q和K.T无法矩阵相乘
  		:param d_v: 注意力中V的嵌入维度，可以跟d_k不同，为了方便一般取相同
		:param d_model: token Embedding和position Embedding的维度
  		:param dropout: 置零比率，nn.Dropout(p=dropout)
  		"""
		super(MultiHeadAttention, self).__init__()

		# 生成Q，K，V的线性变化矩阵
		# Q和K在分头后的最后一维，即d_k必须是相同的，不然Q和K.T无法相乘
		self.W_Q = nn.Linear(d_model, n_heads * d_k, bias=False)
		self.W_K = nn.Linear(d_model, n_heads * d_k, bias=False)
		# V分头的最后一维理论上可以单独取，但为了方便一般取d_v = d_k
		# K和V在分头后的倒数第二维必须是相同的，即len_k == len_v
		self.W_V = nn.Linear(d_model, n_heads * d_v, bias=False)
		# 这个全连接层将多头Attention的输出变为[batch_size, seq_len, d_model]
		self.fc = nn.Linear(n_heads * d_v, d_model)
		self.attention = ScaleDotProductAttention()
		if dropout:
			self.Dropout = nn.Dropout(p=dropout)
		else:
			self.Dropout = None
		self.n_heads = n_heads
		self.d_k = d_k
		self.d_v = d_v
		self.d_model = d_model

	def forward(self,
				input_Q: torch.Tensor,
				input_K: torch.Tensor,
				input_V: torch.Tensor,
				attn_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		"""
		:param input_Q: [batch_size, len_q, d_model]
  		:param input_K: [batch_size, len_k, d_model]
		:param input_V: [batch_size, len_v(=len_k), d_model]
  		:param attn_mask: [batch_size, len_q, len_k]
		"""
		# Add&Norm需要保留一个原始的输入
		residual, batch_size = input_Q, input_Q.size(0)
		# 下面多头的参数矩阵是放在一起做线性变换的，然后再拆成多个头，这是工程上的实现技巧
		# B：batch_size, S: seq_len, D: d_model
		# (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, Head, W) -trans-> (B, Head, S, W)
		#          线性变换                拆成多头

		# Q: [batch_size, n_heads, len_q, d_k]
		Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
		# K: [batch_size, n_heads, len_k, d_k]
		K = self.W_K(input_K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
		# V: [batch_size, n_heads, len_v(==len_k), d_v]
		V = self.W_V(input_V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)

		# 因为是多头，所以mask矩阵需要拓展一个维度
		# attn_mask: [batch_size, len_q, len_k] -> [batch_size, n_heads, len_q, len_k]
		attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)
		# context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
		context, attn = self.attention(Q, K, V, attn_mask, self.Dropout)

		# 将不同头的输出拼在一起
		# context: [batch_size, n_heads, len_q, d_v] -> [batch_size, len_q, n_heads, d_v] -> [batch_size, len_q, n_heads * d_v]
		context = context.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v)

		# 这个全连接层，保证输出的最后一维仍然是d_model
		output = self.fc(context) # [batch_size, len_q, d_model]

		# Add&Norm
		return nn.LayerNorm(d_model)(output + residual), attn


class PoswiseFeedForwardNet(nn.Module):
	"""
	编码器和解码器的每个层都包含一个全连接的前馈网络， 该网络在每个层的位置相同（都在每个encoder-layer或者decoder-layer的最后）
 	pytorch中的Linear只会对最后一维操作，所以正好是我们希望的每个位置用同一个全连接网络
 	"""
	def __init__(self,
				 d_model: int,
				 d_ff: int,
				 dropout: float):
		"""
  		:param d_model: token Embedding和position Embedding的嵌入维度
		:param d_ff: 前馈层中隐藏层的维度
		:param dropout: 置零比率，nn.Dropout(p=dropout)
  		"""
		super(PoswiseFeedForwardNet, self).__init__()

		self.fc = nn.Sequential(
			nn.Linear(d_model, d_ff, bias=False),
			nn.ReLU(),
			nn.Linear(d_ff, d_model, bias=False)
		)
		if dropout:
			self.Dropout =  nn.Dropout(p=dropout)
		else:
			self.Dropout = None

	def forward(self, inputs):
		"""
  		:param inputs: [batch_size, seq_len, d_model]
		"""
		residual = inputs
		output = self.fc(inputs)
		if self.Dropout:
			output = self.Dropout(output)

		return nn.LayerNorm(d_model)(output + residual) # [batch_size, seq_len, d_model]











	
		



















