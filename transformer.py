import math
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from typing import List, Dict, Tuple

from encoder import Encoder
from decoder import Decoder

class Transformer(nn.Module):
	"""
	定义Transformer的模型结构
	"""
	def __init__(self, 
		vocab_params: Dict[str, int], 
		enc_dec_params: Dict[str, int], 
		attn_params: Dict[str, int],
		common_net_params: Dict[str, int]):
		"""
		:param vocab_params: 词表参数 
		{
		src_vocab_size: int,   # 输入词表大小
		tgt_vocab_size: int    # 输出词表大小
		}
		:param enc_dec_params: Encoder和Decoder的参数
		{
		n_layers: int   # Encoder和Decoder的层数
		}
		:param attn_params: 注意力网络的参数
		{
		n_heads: int,   # 多头注意力的头数,
		d_k: int,       # 注意力中Q和K的嵌入维度，Q和K的嵌入维度必须相等，不然Q和K.T无法矩阵相乘
		d_v: int        # 注意力中V的嵌入维度，可以跟d_k不同，为了方便一般取相同
		}
		:param common_net_params: 网络结构通用参数
		{
		d_model: int,     # token Embedding和position Encoding的嵌入维度
		d_ff: int,        # 前馈层中隐藏层的维度
		dropout: float    # 置零比率，nn.Dropout(p=dropout)
		}
		"""
		super(Transformer, self).__init__()
		
		self.encoder = Encoder(vocab_params['src_vocab_size'], enc_dec_params['n_layers'], attn_params, common_net_params).to(device)
		self.decoder = Decoder(vocab_params['tgt_vocab_size'], enc_dec_params['n_layers'], attn_params, common_net_params).to(device)
		self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False).to(device)
		
	def forward(enc_inputs: torch.Tensor, dec_inputs: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
		"""
		Transformer的输入：两个序列
		Transformer的训练方式和其他模型不太一样，他在训练过程是采用了Teacher Forcing的训练过程，就是会将原始输入和正确答案都会喂给模型，然后模型进行训练，而在推理过程中，是不会给正确答案的
		Transformer在训练时会把原始输入(enc_inputs)喂给Encoder形成编码向量，然后将正确答案(dec_inputs)喂给Decoder的第一层
		:param enc_inputs:[batch_size, src_len]
		:param dec_inputs:[batch_size, tgt_len]
		:return:
		"""
		# enc_outputs: [batch_size, src_len, d_model]
		# enc_self_attns是个列表，列表长度为n_layers， 每个元素都是个四维张量[batch_size, n_heads, src_len, src_len]
		enc_outputs, enc_self_attns = self.enocder(enc_inputs)
		# dec_outputs: [batch_size, tgt_len, d_model]
		# dec_self_attns, dec_enc_attns也都是个列表，列表长度均为n_layers, 元素尺寸分别为[batch_size, n_heads, tgt_len, tgt_len]和[batch_size, n_heads, tgt_len, src_len]
		dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_outputs)
		# dec_outputs: [batch_size, tgt_len, d_model] -> dec_logits: [batch_size, tgt_len, tgt_vocab_size]
		dec_logits = self.projection(dec_outputs)
		
		# 这里dec_logits从[batch_size, tgt_len, tgt_vocab_size]->[batch_size * tgt_len, tgt_vocab_size]
		# 因为对于训练来说， 其实不太关注是否是一个独立的句子，重要的是每个字都预测对， 所以把前两个维度拉成一个维度， 方便后面计算loss
		return dec_logits.view(-1, dec_logits.size(-1)), enc_self_attns, dec_self_attns, dec_enc_attns
