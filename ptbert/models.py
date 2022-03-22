import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel, BertSelfAttention
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPoolingAndCrossAttentions

class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)

        return x

class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp

class Pooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'cls_before_pooler': [CLS] representation without the original MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_first_last': average of the first and the last layers.
    """
    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in ["cls", "cls_before_pooler", "avg", "avg_top2", "avg_first_last"], "unrecognized pooling type %s" % self.pooler_type

    def forward(self, attention_mask, outputs):
        last_hidden = outputs.last_hidden_state
        pooler_output = outputs.pooler_output
        hidden_states = outputs.hidden_states

        if self.pooler_type in ['cls_before_pooler', 'cls']:
            return last_hidden[:, 0]
        elif self.pooler_type == "avg":
            return ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
        elif self.pooler_type == "avg_first_last":
            first_hidden = hidden_states[0]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif self.pooler_type == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = ((last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        else:
            raise NotImplementedError

def cl_init(cls, config):
    """
    Contrastive learning class init function.
    """
    cls.pooler_type = cls.model_args.pooler_type
    cls.pooler = Pooler(cls.model_args.pooler_type)
    if cls.model_args.pooler_type == "cls":
        cls.mlp = MLPLayer(config)
    cls.sim = Similarity(temp=cls.model_args.temp)
    cls.init_weights()

def sentemb_forward(
    cls,
    encoder,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
):

    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict

    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=True if cls.pooler_type in ['avg_top2', 'avg_first_last'] else False,
        return_dict=True,
    )

    pooler_output = cls.pooler(attention_mask, outputs)
    if cls.pooler_type == "cls" and not cls.model_args.mlp_only_train:
        pooler_output = cls.mlp(pooler_output)

    if not return_dict:
        return (outputs[0], pooler_output) + outputs[2:]

    return BaseModelOutputWithPoolingAndCrossAttentions(
        pooler_output=pooler_output,
        last_hidden_state=outputs.last_hidden_state,
        hidden_states=outputs.hidden_states,
    )

def polysentemb_forward(
    cls,
    encoder,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
):

    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict

    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=True if cls.pooler_type in ['avg_top2', 'avg_first_last'] else False,
        return_dict=True,
    )

    pooler_output = cls.pooler(attention_mask, outputs)
    if cls.pooler_type == "cls" and not cls.model_args.mlp_only_train:
        pooler_output = cls.mlp(pooler_output)

    poly_codes = torch.arange(cls.model_args.poly_m, dtype = torch.long).to(input_ids.device)
    poly_emb = cls.polyEmbedding(poly_codes.unsqueeze(0).expand(input_ids.shape[0], cls.model_args.poly_m))

    poly_emb = cls.polyAttn(poly_emb,
                            encoder_hidden_states = pooler_output.unsqueeze(1))[0]

    poly_emb = cls.polyAttn(pooler_output.unsqueeze(1),
                            attention_mask = attention_mask,
                            encoder_hidden_states = poly_emb)[0]
    poly_emb = poly_emb.squeeze()

    if not return_dict:
        return (outputs[0], poly_emb) + outputs[2:]

    return BaseModelOutputWithPoolingAndCrossAttentions(
        pooler_output=poly_emb,
        last_hidden_state=outputs.last_hidden_state,
        hidden_states=outputs.hidden_states,
    )

def poly_cl_forward(cls,
    encoder,
    momentum_encoder,
    momentum_queue,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    mlm_input_ids=None,
    mlm_labels=None,
):
    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict
    ori_input_ids = input_ids
    # Number of sentences in one instance
    # 2: pair instance; 3: pair instance with a hard negative
    batch_size, num_sent, seq_length = input_ids.shape #(bs, num_sent, seq_length)
    raw_seq = input_ids[:, 0, :]
    momentum_seq = input_ids[:, 1, :]
    raw_attention_mask = attention_mask[:, 0, :]
    momentum_attention_mask = attention_mask[:, 1, :]

    raw_token_type_ids = None
    momentum_token_type_ids = None
    if token_type_ids is not None:
        raw_token_type_ids = token_type_ids[:, 0, :]
        momentum_token_type_ids = token_type_ids[:, 1, :]

    mlm_outputs = None
    # Flatten input for encoding
    # Get raw embeddings
    raw_outputs = encoder(
        raw_seq,
        attention_mask=raw_attention_mask,
        token_type_ids=raw_token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=True if cls.model_args.pooler_type in ['avg_top2', 'avg_first_last'] else False,
        return_dict=True,
    )
    with torch.no_grad():
        momentum_outputs = momentum_encoder(
            momentum_seq,
            attention_mask=momentum_attention_mask,
            token_type_ids=momentum_token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True if cls.model_args.pooler_type in ['avg_top2', 'avg_first_last'] else False,
            return_dict=True,
        )
        momentum_pooler_output = cls.momentum_encoder_pooler(momentum_attention_mask, momentum_outputs)
        momentum_pooler_output = momentum_pooler_output.view(batch_size, -1) # (bs, hidden)

    pooler_output = cls.pooler(attention_mask, raw_outputs)
    pooler_output = pooler_output.view(batch_size, -1) # (bs, hidden)

    poly_codes = torch.arange(cls.model_args.poly_m, dtype = torch.long).to(input_ids.device)
    poly_emb = cls.polyEmbedding(poly_codes.unsqueeze(0).expand(input_ids.shape[0], cls.model_args.poly_m))

    poly_emb = cls.polyAttn(poly_emb,
                            encoder_hidden_states = pooler_output.unsqueeze(1))[0]

    poly_emb = cls.polyAttn(pooler_output.unsqueeze(1),
                            attention_mask = raw_attention_mask,
                            encoder_hidden_states = poly_emb)[0]
    poly_emb = poly_emb.squeeze(1)

    # If using "cls", we add an extra MLP layer
    # (same as BERT's original implementation) over the representation.
    with torch.no_grad():
        momentum_poly_emb = cls.momentumPolyEmbedding(poly_codes.unsqueeze(0).expand(input_ids.shape[0], cls.model_args.poly_m))
        momentum_poly_emb = cls.momentumPolyAttn(momentum_poly_emb,
                                encoder_hidden_states = momentum_pooler_output.unsqueeze(1))[0]

        momentum_poly_emb = cls.polyAttn(momentum_pooler_output.unsqueeze(1),
                            attention_mask = momentum_attention_mask,
                            encoder_hidden_states = momentum_poly_emb)[0]
        momentum_poly_emb = momentum_poly_emb.squeeze(1)
    if cls.pooler_type == "cls":
        pooler_output = cls.mlp(poly_emb)
        with torch.no_grad():
            momentum_pooler_output = cls.momentum_mlp(momentum_poly_emb)
            momentum_queue.insert(0, momentum_pooler_output)
            momentum_pooler_output = torch.cat(momentum_queue, dim = 0)

    cos_sim = cls.sim(pooler_output.unsqueeze(1), momentum_pooler_output.unsqueeze(0))

    labels = torch.arange(cos_sim.size(0)).long().to(cls.device)
    loss_fct = nn.CrossEntropyLoss()

    loss = loss_fct(cos_sim, labels)

    if not return_dict:
        output = (cos_sim,) + raw_outputs[2:]
        return ((loss,) + output) if loss is not None else output

    return SequenceClassifierOutput(
        loss=loss,
        logits=cos_sim,
        hidden_states=raw_outputs.hidden_states,
        attentions=raw_outputs.attentions,
    )

def moco_cl_forward(cls,
    encoder,
    momentum_encoder,
    momentum_queue,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    mlm_input_ids=None,
    mlm_labels=None,
):
    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict
    ori_input_ids = input_ids
    # Number of sentences in one instance
    # 2: pair instance; 3: pair instance with a hard negative
    batch_size, num_sent, seq_length = input_ids.shape #(bs, num_sent, seq_length)
    raw_seq = input_ids[:, 0, :]
    momentum_seq = input_ids[:, 1, :]
    raw_attention_mask = attention_mask[:, 0, :]
    momentum_attention_mask = attention_mask[:, 1, :]
    if token_type_ids is not None:
        raw_token_type_ids = token_type_ids[:, 0, :]
        momentum_token_type_ids = token_type_ids[:, 1, :]

    mlm_outputs = None
    # Flatten input for encoding
    # Get raw embeddings
    raw_outputs = encoder(
        raw_seq,
        attention_mask=raw_attention_mask,
        token_type_ids=raw_token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=True if cls.model_args.pooler_type in ['avg_top2', 'avg_first_last'] else False,
        return_dict=True,
    )
    with torch.no_grad():
        momentum_outputs = momentum_encoder(
            momentum_seq,
            attention_mask=momentum_attention_mask,
            token_type_ids=momentum_token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True if cls.model_args.pooler_type in ['avg_top2', 'avg_first_last'] else False,
            return_dict=True,
        )
        momentum_pooler_output = cls.momentum_encoder_pooler(momentum_attention_mask, momentum_outputs)
        momentum_pooler_output = momentum_pooler_output.view(batch_size, -1) # (bs, hidden)

    pooler_output = cls.pooler(attention_mask, raw_outputs)
    pooler_output = pooler_output.view(batch_size, -1) # (bs, hidden)

    # If using "cls", we add an extra MLP layer
    # (same as BERT's original implementation) over the representation.
    if cls.pooler_type == "cls":
        pooler_output = cls.mlp(pooler_output)
        with torch.no_grad():
            momentum_pooler_output = cls.momentum_mlp(momentum_pooler_output)

    momentum_queue.insert(0, momentum_pooler_output)
    momentum_pooler_output = torch.cat(momentum_queue, dim = 0)

    cos_sim = cls.sim(pooler_output.unsqueeze(1), momentum_pooler_output.unsqueeze(0))

    labels = torch.arange(cos_sim.size(0)).long().to(cls.device)
    loss_fct = nn.CrossEntropyLoss()

    loss = loss_fct(cos_sim, labels)

    if not return_dict:
        output = (cos_sim,) + raw_outputs[2:]
        return ((loss,) + output) if loss is not None else output

    return SequenceClassifierOutput(
        loss=loss,
        logits=cos_sim,
        hidden_states=raw_outputs.hidden_states,
        attentions=raw_outputs.attentions,
    )

class BertForPolyMoCo(BertPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]
    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs['model_args']
        self.bert = BertModel(config, add_pooling_layer = False)
        self.momentum_encoder = copy.deepcopy(self.bert)
        cl_init(self, config)

        self.polyEmbedding = nn.Embedding(self.model_args.poly_m, config.hidden_size)
        self.momentumPolyEmbedding = copy.deepcopy(self.polyEmbedding)
        self.polyAttn = BertSelfAttention(config)
        self.momentumPolyAttn = copy.deepcopy(self.polyAttn)

        self.momentum_encoder_pooler = copy.deepcopy(self.pooler)
        self.momentum_mlp = copy.deepcopy(self.mlp)
        self.momentum_queue = []

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False,
        mlm_input_ids=None,
        mlm_labels=None,
    ):
        if sent_emb:
            if self.model_args.return_poly:
                return polysentemb_forward(self, self.bert,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds,
                    labels=labels,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )

            return sentemb_forward(self, self.bert,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        else:
            if len(self.momentum_queue) > self.model_args.queue_size:
                self.momentum_queue.pop()

            return poly_cl_forward(self, self.bert, self.momentum_encoder,
                momentum_queue = self.momentum_queue,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                mlm_input_ids=mlm_input_ids,
                mlm_labels=mlm_labels,
            )

class BertForMoCo(BertPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]
    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs['model_args']
        self.bert = BertModel(config, add_pooling_layer = False)
        self.momentum_encoder = copy.deepcopy(self.bert)
        cl_init(self, config)
        self.momentum_encoder_pooler = copy.deepcopy(self.pooler)
        self.momentum_mlp = copy.deepcopy(self.mlp)
        self.momentum_queue = []

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False,
        mlm_input_ids=None,
        mlm_labels=None,
    ):
        if sent_emb:
            return sentemb_forward(self, self.bert,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            if len(self.momentum_queue) > self.model_args.queue_size:
                self.momentum_queue.pop()

            return moco_cl_forward(self, self.bert, self.momentum_encoder,
                momentum_queue = self.momentum_queue,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                mlm_input_ids=mlm_input_ids,
                mlm_labels=mlm_labels,
            )