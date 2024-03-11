from transformers.modeling_outputs import SequenceClassifierOutput
from torch import nn


class ModelWithCustomLossFunction(nn.Module):
    def __init__(self, pretrained_model, pretrained_model_label=None, num_labels=None, dropout=0.3, linear_layer_dim=768, output_attentions=False, output_hidden_states=True):
        super(ModelWithCustomLossFunction, self).__init__()
        self.pretrained_model = pretrained_model
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(linear_layer_dim, num_labels)
        self.num_labels = num_labels
        self.output_attentions = output_attentions
        self.num_labels = num_labels

    def forward(self, ids, attention_mask, labels=None, class_weights=None):
        outputs = self.pretrained_model(ids, attention_mask=attention_mask)

        if hasattr(outputs, 'pooler_output'):
            pooled_output = outputs.pooler_output
        else:
            last_hidden_state = outputs[0]
            pooled_output = last_hidden_state[:, 0]

        output = self.dropout(pooled_output)
        logits = self.classifier(output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fn = nn.MSELoss()
                loss = loss_fn(logits.view(-1), labels.view(-1))
            else:
                if class_weights is None:
                    loss_fn = nn.CrossEntropyLoss()
                else:
                    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
                # you can define any loss function here yourself
                # see https://pytorch.org/docs/stable/nn.html#loss-functions for an overview
                # next, compute the loss based on logits + ground-truth labels
                loss = loss_fn(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
        )
