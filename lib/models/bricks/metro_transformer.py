import numpy as np
import scipy
import torch
from torch import nn
from transformers.models.bert.modeling_bert import (BertConfig, BertEmbeddings, BertEncoder, BertPooler,
                                                    BertPreTrainedModel)


class BertLayerNorm(nn.Module):

    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class METRO_Encoder(BertPreTrainedModel):
    """
        Modified the original METRO encoder to provide an interface for key value and query to be different.
    """

    def __init__(self, config):
        super(METRO_Encoder, self).__init__(config)
        self.config = config
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size) # Independent of K, V or Q
        self.img_dim = config.img_feature_dim
        self.key_dim = config.bps_feature_dim

        try:
            self.use_img_layernorm = config.use_img_layernorm
        except:
            self.use_img_layernorm = None

        self.embedding = nn.Linear(self.img_dim, self.config.hidden_size, bias=True)
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        if self.use_img_layernorm:
            self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.img_layer_norm_eps)

        # self.apply(self.init_weights)
        self.init_weights()

    def _prune_heads(self, heads_to_prune):
        """Prunes heads of the model.
        heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(self,
                query_feats,
                key_feats,
                query_input_ids=None,
                token_type_ids=None,
                attention_mask=None,
                head_mask=None):

        batch_size = len(query_feats)
        
        query_seq_length = len(query_feats[0])
        query_input_ids = torch.zeros([batch_size, query_seq_length], dtype=torch.long).to(query_feats.device)
        
        key_seq_length = len(key_feats[0])
        key_input_ids = torch.zeros([batch_size, key_seq_length], dtype=torch.long).to(key_feats.device)


        query_position_ids = torch.arange(query_seq_length, dtype=torch.long, device=query_input_ids.device)
        query_position_ids = query_position_ids.unsqueeze(0).expand_as(query_input_ids)
        key_position_ids = torch.arange(key_seq_length, dtype=torch.long, device=key_input_ids.device)
        key_position_ids = key_position_ids.unsqueeze(0).expand_as(key_input_ids)

        query_position_embeddings = self.position_embeddings(query_position_ids)
        key_position_embeddings = self.position_embeddings(key_position_ids)

        if attention_mask is None:
            attention_mask = torch.ones_like(query_input_ids)

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(query_input_ids)

        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        elif attention_mask.dim() == 3:
            extended_attention_mask = attention_mask.unsqueeze(1)
        else:
            raise NotImplementedError

        # extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = (head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                            )  # We can specify head_mask for each layer
            # head_mask = head_mask.to(dtype=next(self.parameters()).dtype)  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        # Project input token features to have specified hidden size
        query_embedding_output = self.embedding(query_feats)
        key_embedding_output = self.embedding(key_feats)

        # We empirically observe that adding an additional learnable position embedding leads to more stable training
        
        # print("query_position_embeddings.shape ", query_position_embeddings.shape)
        # print("query_embedding_output.shape ", query_embedding_output.shape)
        # print("key_position_embeddings ", key_position_embeddings.shape)
        # print("key_embedding_output ", key_embedding_output.shape)
        query_embeddings = query_position_embeddings + query_embedding_output
        key_embeddings = key_position_embeddings + key_embedding_output

        if self.use_img_layernorm:
            query_embeddings = self.LayerNorm(query_embeddings)
            
        query_embeddings = self.dropout(query_embeddings)
        key_embeddings = self.dropout(key_embeddings)

        encoder_outputs = self.encoder(
            query_embeddings, 
            extended_attention_mask, 
            head_mask=head_mask,
            encoder_hidden_states=key_embeddings)
        
        sequence_output = encoder_outputs[0]

        outputs = (sequence_output,)
        if self.config.output_hidden_states:
            all_hidden_states = encoder_outputs[1]
            outputs = outputs + (all_hidden_states,)
        if self.config.output_attentions:
            all_attentions = encoder_outputs[-1]
            outputs = outputs + (all_attentions,)

        return outputs



class METROBlock(BertPreTrainedModel):
    """
    The architecture of a transformer encoder block we used in METRO
    """

    def __init__(self, config):
        super(METROBlock, self).__init__(config)
        self.config = config
        self.bert = METRO_Encoder(config)
        self.query_cls_head = nn.Linear(config.hidden_size, self.config.output_feature_dim)
        self.query_residual = nn.Linear(config.img_feature_dim, self.config.output_feature_dim)
        self.key_residual = nn.Linear(config.img_feature_dim, self.config.output_feature_dim)
        # self.apply(self.init_weights)
        self.init_weights()

    def forward(
        self,
        query_feats,
        key_feats,
    ):
        """
        # self.bert has three outputs
        # predictions[0]: output tokens
        # predictions[1]: all_hidden_states, if enable "self.config.output_hidden_states"
        # predictions[2]: attentions, if enable "self.config.output_attentions"
        """
        
        predictions = self.bert(
            query_feats=query_feats,
            key_feats=key_feats
        )

        # We use "self.cls_head" to perform dimensionality reduction. We don't use it for classification.
        pred_score = self.query_cls_head(predictions[0])
        res_img_feats = self.query_residual(query_feats)
        pred_score = pred_score + res_img_feats
        
        key_feats = self.key_residual(key_feats)

        # if self.config.output_attentions and self.config.output_hidden_states:
        #     return pred_score, predictions[1], predictions[-1]
        # else:
        return pred_score, key_feats