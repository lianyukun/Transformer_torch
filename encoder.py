import math
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from typing import List, Dict, Tuple

from module import MultiHeadAttention, PoswiseFeedForwardNet
from util import PositionalEncoding, get_attn_pad_mask

class EncoderLayer(nn.Module):
	"""
 	编码器Encoder中单个编码层
 	"""
	def __init__(self,
				 attn_params: Dict[str, int],
				 common_net_params: Dict[str, int]):
		super(EnocderLayer, self).__init__()

		# 编码层的自注意力网络
		self.enc_self_attn = MultiHeadAttention(attn_params['n_heads'], attn_params['d_k'], attn_params['d_v'], common_net_params['d_model'], common_net_params['dropout'])
		# 编码层的前馈网络（包含Add&Norm）
		self.pos_ffn = PoswiseFeedForwardNet(common_net_params['d_model'], common_net_params['d_ff'], common_net_params['dropout'])

	def forward(self, enc_layer_inputs: torch.Tensor, enc_self_attn_mask: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
		"""
		:param enc_layers_inputs: [batch_size, src_len, d_model]
  		:param enc_self_attn_mask: [batch_size, src_len, src_len]
		Encoder中的self_attn只需要padding mask
  		Decoder中的self_attn需要padding mask和sequence mask， dec_enc_attn只需要padding mask
		"""
		# enc_outputs: [batch_size, src_len, d_model]
		# attn: [batch_size, n_heads, src_len, src_len]
		# 第一个输入enc_layer_inputs * W_Q = Q
		# 第二个输入enc_layer_inputs * W_K = K
		# 第三个输入enc_layer_inputs * W_V = V
		enc_outputs, attn = self.MutltiHeadAttention(enc_layer_inputs, enc_layer_inputs, enc_layer_inputs, enc_self_attn_mask)
		enc_ff_outputs = self.pos_ffn(enc_outputs)

		return enc_ff_outputs, attn

class Encoder(nn.Module):
	"""
	定义Transformer中的编码器Encoder
  	"""
	def __init__(self,
				 src_vocab_size: int,
				 n_layers: int,
				 attn_params: Dict[str, int],
				 common_net_params: Dict[str, int]):
		"""
		:param src_vocab_size: 输入词表大小
  		:param n_layers: 编码器中编码层block的数量
		:param attn_params: 注意力网络参数字典
  		:param common_net_params: 网络结构通用参数
		"""
		super(Encoder, self).__init__()

		# 原始输入，做token Embedding
		self.src_emb = nn.Embedding(src_vocab_size, common_net_params['d_model']
		# add位置embedding
		self.pos_emb = PositionalEncoding(common_net_params['d_model']
		# 多个EncoderLayer组成编码器Encoder
		self.layers = nn.ModuleList([EncoderLayer(attn_params, common_net_params) for _ in range(n_layers)])

	def forward(self, enc_inputs: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
		"""
  		:param enc_inputs: [batch_size, src_len]
  		"""
		# enc_emb_outputs: [batch_size, src_len, d_model]
		enc_emb_outputs = self.src_emb(enc_inputs)
		# enc_outputs: [batch_size, src_len,d_model]
		enc_outputs = self.pos_emb(enc_emb_outputs)
		# Encoder输入中的padding mask矩阵，Encoder中只有padding mask
		enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)
		# 在计算中不会用到，保存attention矩阵，方便画图
		enc_self_attns = []
		# for循环遍历访问ModuleList
		for layer in self.layers:
			# 上一个block的输出是当前block的输入
			# enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
			enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
			enc_self_attns.append(enc_self_attn)

		return enc_outputs, enc_self_attns
	
		
		
