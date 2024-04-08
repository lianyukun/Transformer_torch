import math
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from typing import List, Dict, Tuple

from module import MultiHeadAttention, PoswiseFeedForwardNet
from util import PositionalEncoding, get_attn_pad_mask, get_attn_subsequence_mask

class DecoderLayer(nn.Module):
	"""
	解码器Decoder中的单个解码层DecoderLayer
	"""
	def __init__(self,
			   attn_params: Dict[str, int],
			   common_net_params: Dict[str, int]):
		super(DecoderLayer, self).__init__()

		# 解码层的自注意力网络
		self.dec_self_attn = MultiHeadAttention(attn_params['n_heads'], attn_params['d_k'], attn_params['d_v'], common_net_params['d_model'], common_net_params['dropout'])
		# 解码层的dec_inputs的Q和enc_outputs的K、V的注意力网络
		self.dec_enc_attn = MultiHeadAttention(attn_params['n_heads'], attn_params['d_k'], attn_params['d_v'], common_net_params['d_model'], common_net_params['dropout'])
		# 解码器的前馈神经网络，包含Add&Norm
		self.pos_ffn = PoswiseFeedForwardNet(common_net_params['d_model'], common_net_params['d_ff'], common_net_params['dropout'])

	def forward(self,
				dec_inputs: torch.Tensor,
				enc_outputs: torch.Tensor,
				dec_self_attn_mask: torch.Tensor,
				dec_enc_attn_mask: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
		"""
  		:param dec_inputs: [batch_size, tgt_len, d_model]
		:param enc_outputs: [batch_size, src_len, d_model]
  		:param dec_self_attn_mask: [batch_size, tgt_len, tgt_len]
		:param dec_enc_attn_mask: [batch_size, tgt_len, src_len]
		"""
		# dec_self_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len]
		# 这里的Q，K，V全部是来自于dec_inputs，属于自注意力，为了得到dec_enc_attn中的Q
		dec_self_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
		# dec_enc_outputs: [batch_size, tgt_len, d_model], dec_enc_attn: [batch_size, n_heads, tgt_len, src_len]
		dec_enc_outputs, dec_enc_attn = self.dec_enc_attn(dec_self_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
		# dec_ff_outputs: [batch_size, tgt_len. d_model
		dec_ff_outputs = self.pos_ffn(dec_enc_outputs)

		return dec_ff_outputs, dec_self_attn, dec_enc_attn


class Decoder(nn.Module):
	"""
	定义Transformer中的解码器Decoder
 	"""
	def __init__(self,
				 tgt_vocab_size: int,
				 n_layers: int,
				 attn_params: Dict[str, int],
				 common_net_params: Dict[str, int]):
		"""
		:param tgt_vocab_size: 输出词表大小
  		:param n_layers: 解码层中解码层block的数量
		:param attn_params: 注意力网络参数字典
  		:param common_net_params: 网络结构通用参数
		"""
		super(Decoder, self).__init__()
		# Decoder输入的embedding维表
		self.tgt_emb = nn.Embedding(tgt_vocab_size, common_net_param['d_model']
		self.pos_emb = PositionalEncoding(common_net_params['d_model']
		# 多个DecoderLayer组成编码器Decoder
		self.layers = nn.ModuleList([DecoderLayer(attn_params, common_net_params) for _ in range(n_layers)])

	def forward(self,
				dec_inputs: torch.Tensor,
				enc_outputs: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
		"""
		:param dec_inputs: [batch_size, tgt_len]
  		:param enc_outputs: [batch_size, src_len, d_model] # 用在Encoder-Decoder的Attention层
  		"""
		# dec_emb_outputs: [batch_size, tgt_len, d_model]
		dec_emb_outputs = self.tgt_emb(dec_inputs)
		# dec_outputs: [batch_size, tgt_len, d_model]
		dec_outputs = self.pos_emb(dec_emb_outputs)
		#  Decoder中self_attn的pad和subsequence mask矩阵的后两个维度都是方阵，因为自注意力中，Q、K、V都来自于同一输入
		# dec_self_attn_pad_mask: [batch_size, tgt_len, tgt_len]
		dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs)
		# sequence mask for Self_Attention: 当前时刻看不到未来信息，Decoder中只有self_attn部分需要sequence mask
		# dec_self_attn_subsequence: [batch_size, tgt_len, tgt_len]
		dec_self_attn_subsequence_mask = get_attn_subsequence_mask(dec_inputs)
		# Decoder中self_attn部分最终的mask矩阵等于pad+subsequence
		dec_self_attn_mask = (dec_self_attn_pad_mask | dec_self_attn_subsequence_mask).type(torch.float)

		# Encoder-Decoder的Attention层
		# 这里的mask只是pad_mask(因为enc是处理K，V的，求Attention时是用v1,v2,..vm其余去加权，要把pad对应的vi的相关系数设置为0， 这样注意力就不会关注pad向量)
		# Q的长度来自dec_inputs, K、V的长度来自于enc_outputs
		# denc_enc_attn_pad_mask: [batch_size, tgt_len, src_len]
		dec_enc_attn_pad_mask = get_attn_pad_mask(dec_inputs, enc_outputs)

		# 在计算中并不会用到，画图用
		dec_self_attns, dec_enc_attns = [], []
		for layer in self.layers:
			# dec_outputs; [batch_size, tgt_len, d_model]
			# dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len]
			# dec_enc_attn: [batch_size, n_heads, tgt_len, src_len]
			# Decoder中当前Block的输入上一个Block的输出dec_outputs(变化）和Encoder网络的输出enc_outputs(固定）
			dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_pad_mask)

			dec_self_attns.append(dec_self_attn)
			dec_enc_attns.append(dec_enc_attn)

		# dec_outputs: [batch_size, tgt_len, d_model]
		return dec_outputs, dec_self_attns, dec_enc_attns
		
		
				

		
		
		
