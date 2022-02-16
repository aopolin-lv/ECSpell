import os

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.models.bert.modeling_bert import BertModel, BertEncoder
from transformers.models.bert.configuration_bert import BertConfig

from glyce.layers.glyph_position_embed import GlyphPositionEmbedder
from glyce.layers.classifier import SingleLinearClassifier, MultiNonLinearClassifier


class ECSpell(nn.Module):
    def __init__(self, config, py_vocab_len, num_labels, train_state=False):
        super(ECSpell, self).__init__()
        self.num_labels = num_labels
        self.config = config
        self.config.id2label = {i: f"LABEL_{i}" for i in range(self.num_labels)}
        self.config.label2id = {v: k for k, v in self.config.id2label.items()}

        self.glyph_transformer = GlyceTransformer(self.config, py_vocab_len, self.num_labels, train_state)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.loss_fct = CrossEntropyLoss()
        if self.config.classifier_sign == "single_linear":
            self.classifier = SingleLinearClassifier(config.hidden_size, self.num_labels)
        elif self.config.classifier_sign == "multi_nonlinear":
            self.classifier = MultiNonLinearClassifier(config.hidden_size, self.num_labels)
        else:
            raise ValueError

    def forward(
        self,
        input_ids,
        pinyin_ids=None,
        token_type_ids=None,
        attention_mask=None,
        labels=None,
        past_key_values=None
    ):
        last_hidden_state, glyph_loss = self.glyph_transformer(input_ids=input_ids,
                                                               pinyin_ids=pinyin_ids,
                                                               token_type_ids=token_type_ids,
                                                               attention_mask=attention_mask,
                                                               past_key_values=past_key_values)
        logits = self.classifier(self.dropout(last_hidden_state))

        sum_loss = None
        if labels is not None:
            loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            sum_loss = loss + self.config.glyph_ratio * glyph_loss

        return TokenClassifierOutput(
            loss=sum_loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )


class GlyceTransformer(nn.Module):
    def __init__(self, config, py_vocab_len, num_labels=4, train_state=False):
        super(GlyceTransformer, self).__init__()
        self.num_labels = num_labels
        # glyph_embedding and pho_embedding
        self.glyph_embedding = GlyphPositionEmbedder(config.glyph_config)
        bert_config = BertConfig.from_pretrained(config.glyph_config.bert_model)
        self.pho_embedding = PinyinEmbedding(bert_config, py_vocab_len)

        # glyce_bert layer
        if config.glyph_config.bert_model and os.path.exists(config.glyph_config.bert_model) and train_state:
            self.bert_model = BertModel(bert_config).from_pretrained(config.glyph_config.bert_model)
        else:
            self.bert_model = BertModel(bert_config)

        # transformer layer
        self.map_fc = nn.Linear(bert_config.hidden_size * 2, bert_config.hidden_size)
        self.LayerNorm = nn.LayerNorm(bert_config.hidden_size, eps=bert_config.layer_norm_eps)
        self.dropout = nn.Dropout(bert_config.hidden_dropout_prob)

        transformer_config = {k: v for k, v in config.transformer_config.__dict__.items()}
        transformer_config = BertConfig.from_dict(transformer_config)
        self.transformer_layer = BertEncoder(transformer_config)

        if config.bert_frozen == "true":
            print("!=!" * 20)
            print("Please notice that the bert model if frozen")
            print("the loaded weights of models is ")
            print(config.glyph_config.bert_model)
            print("!-!" * 20)
            for param in self.bert_model.parameters():
                param.requires_grad = False

    def forward(self,
                input_ids=None,
                pinyin_ids=None,
                token_type_ids=None,
                attention_mask=None,
                past_key_values=None
                ):
        glyph_embeddings, glyph_cls_loss = self.glyph_embedding(input_ids, token_type_ids=token_type_ids)
        pho_embeddings = self.pho_embedding(pinyin_ids)
        context_bert_output = self.bert_model(input_ids=input_ids,
                                              token_type_ids=token_type_ids,
                                              attention_mask=attention_mask,
                                              past_key_values=past_key_values
                                              )[0]

        sup_embeddings = torch.cat((pho_embeddings, glyph_embeddings), 2)
        sup_embeddings = self.dropout(self.LayerNorm(self.map_fc(sup_embeddings)))
        input_features = torch.cat((context_bert_output, sup_embeddings), 2)
        # input_features = torch.cat((glyph_embeddings, context_bert_output), 2)

        attention_mask = torch.ones_like(input_ids)
        token_type_ids = torch.zeros_like(input_ids)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * - 10000.0
        last_hidden_state = self.transformer_layer(input_features, extended_attention_mask)[0]

        return last_hidden_state, glyph_cls_loss


class PinyinEmbedding(nn.Module):
    def __init__(self, config, py_vocab_len):
        super(PinyinEmbedding, self).__init__()
        self.pho_config = config
        # pho_config.num_hidden_layers = 4
        # self.pho_embedding = nn.Embedding(py_vocab_len, pho_config.hidden_size, padding_idx=0)
        # self.transformer = TransformerLayer(pho_config)
        self.out_dim = self.pho_config.hidden_size

        embedding_size = 128
        self.pho_embedding = nn.Embedding(py_vocab_len, embedding_size)
        self.conv = nn.Conv1d(in_channels=embedding_size, out_channels=self.out_dim, kernel_size=2, stride=1, padding=0)
        # self.conv = nn.Conv1d(in_channels=embedding_size, out_channels=self.out_dim, kernel_size=1, stride=1, padding=0)

    def forward(self, pinyin_ids):
        # pinyin_ids = pinyin_ids.unsqueeze(-1)
        embed = self.pho_embedding(pinyin_ids)
        bs, seq_len, pinyin_locs, embed_size = embed.shape
        view_embed = embed.view(-1, pinyin_locs, embed_size)

        # transformer + max_pooling
        # out = self.transformer(view_embed)[0]
        # out = out.permute(0, 2, 1)
        # out = F.max_pool1d(out, out.shape[-1])

        # conv + max_pooling
        input_embed = view_embed.permute(0, 2, 1)
        pinyin_conv = self.conv(input_embed)
        out = F.max_pool1d(pinyin_conv, pinyin_conv.shape[-1])
        out = out.view(bs, seq_len, self.out_dim)
        return out
