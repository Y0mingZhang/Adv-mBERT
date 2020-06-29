import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import BertConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_bert import BertEncoder, BertPooler, BertPreTrainedModel

class FC_Discriminator(nn.Module):
    def __init__(self, seq_len, input_dim, n_langs):
        self.fc_in = seq_len * input_dim
        super(FC_Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(self.fc_in, 256),
            nn.ReLU(),
            nn.Linear(256, n_langs),
        )

    def forward(self, x):
        x = x.view(-1, self.fc_in) 
        return self.net(x)

class Larger_MLP(nn.Module):
    def __init__(self, input_dim, n_langs):
        super(Larger_MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.LeakyReLU(),
            nn.Linear(2048, 2048),
            nn.LeakyReLU(),
            nn.Linear(2048, n_langs),
        )

    def forward(self, x):
        return self.net(x)

class StackedTransformerEncoder(nn.Module):
    def __init__(self, num_layers, hidden_dim, n_head, n_langs):
        super(StackedTransformerEncoder, self).__init__()
        te_layer = nn.TransformerEncoderLayer(1024, n_head, dim_feedforward=hidden_dim, activation='gelu')
        self.model = nn.TransformerEncoder(te_layer, num_layers)
        self.activation = nn.LeakyReLU()
        self.linear = nn.Linear(hidden_dim, n_langs)
    def forward(self, x, mask):
        output = self.model(x, mask=mask)
        return self.activation(self.linear(output))

class MeanPoolingDiscriminator(nn.Module):
    def __init__(self):
        super(MeanPoolingDiscriminator, self).__init__()
        self.linear = nn.Linear(768, 32)
        self.relu = nn.PReLU()
        self.out = nn.Linear(32, 1)
    
    def forward(self, inputs_embeds, attention_mask,
             labels):
        
        inputs_embeds[attention_mask==0.0] = 0
        sum_ = inputs_embeds.sum(1)
        lengths = attention_mask.sum(1).view(-1,1).repeat(1, sum_.shape[1])

        mean_pool = sum_ / lengths
        return self.out(self.relu(self.linear(mean_pool))).squeeze()
        
# class BertEncoderWrapper(BertPreTrainedModel):
#     def __init__(self, config, n_langs, pool=False):
#         super(BertEncoderWrapper, self).__init__(config)
#         self.config = config
#         self.encoder = BertEncoder(config)
#         self.pool = pool
#         if self.pool:
#             self.pooler = BertPooler(config)
            
#         self.pred_layer = nn.Linear(config.hidden_size, n_langs)
#         self.init_weights()

#     def forward(self, x, mask):
#         head_mask = [None] * self.config.num_hidden_layers
#         extended_attention_mask = mask.unsqueeze(1).unsqueeze(2)
#         extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

#         encoder_outputs = self.encoder(x,
#                                        extended_attention_mask,
#                                        head_mask=head_mask)

#         sequence_output = encoder_outputs[0]

#         if self.pool:
#             pooled_output = self.pooler(sequence_output)
#             return self.pred_layer(pooled_output)
#         else:
#             return self.pred_layer(sequence_output)

        


# class BertEncoderWrapper(BertPreTrainedModel):
#     def __init__(self, config):
#         super().__init__(config)
#         self.num_labels = config.num_labels

#         self.encoder = BertEncoder(config)
#         self.pooler = BertPooler(config)

#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.classifier = nn.Linear(config.hidden_size, config.num_labels)

#         self.init_weights()

#     def forward(
#         self,
#         encoder_input=None,
#         attention_mask=None,
#         token_type_ids=None,
#         position_ids=None,
#         head_mask=None,
#         inputs_embeds=None,
#         labels=None,
#     ):  
#         device = encoder_input.device if input_ids is not None else inputs_embeds.device

#         if token_type_ids is None:
#             token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
#         encoder_outputs = self.encoder(
#             encoder_input,
#             attention_mask=attention_mask,
#             head_mask=None,
#             encoder_hidden_states=None,
#             encoder_attention_mask=None,
#         )
#         sequence_output = encoder_outputs[0]
#         pooled_output = self.pooler(sequence_output)
#         pooled_output = self.dropout(pooled_output)
#         logits = self.classifier(pooled_output)

#         outputs = (logits,)

#         if labels is not None:
#             if self.num_labels == 1:
#                 #  We are doing regression
#                 loss_fct = MSELoss()
#                 loss = loss_fct(logits.view(-1), labels.view(-1))
#             else:
#                 loss_fct = CrossEntropyLoss()
#                 loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
#             outputs = (loss,) + outputs

#         return outputs  # (loss), logits, (hidden_states), (attentions)