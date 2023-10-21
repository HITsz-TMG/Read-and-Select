import torch
from torch import nn
from transformers import RobertaModel, RobertaConfig



class ExtractInfoEncoder(nn.Module):
    def __init__(self, pretrained_model, device, args):
        super(ExtractInfoEncoder, self).__init__()
        self.bert_config = RobertaConfig.from_pretrained(pretrained_model)
        model = RobertaModel(self.bert_config).from_pretrained(pretrained_model)
        model.resize_token_embeddings(self.bert_config.vocab_size + 20)
        self.model = model
        self.tokenizer = args.tokenizer
        self.device = device
        self.info_token_num = args.info_token_num
        self.net = nn.Sequential(
            torch.nn.LayerNorm(self.bert_config.hidden_size),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(self.bert_config.hidden_size, 128),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(128),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(128, 1),
        )

    def get_extended_attention_mask(
            self, attention_mask
    ):

        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:

            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                f"Wrong shape for input_ids  or attention_mask "
            )

        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask


    def get_extracted_candidates(self, candidates_input_ids, candidates_attention_masks):
        B, can_num, seq_len = candidates_input_ids.shape
        hidden_states = []
        candidates_input_ids = candidates_input_ids.reshape((B * can_num), seq_len)
        candidates_attention_masks = candidates_attention_masks.reshape((B * can_num), seq_len)

        step = 8
        for i in range(0, B * can_num, step):
            output = self.model(candidates_input_ids[i:i + step],
                                candidates_attention_masks[i:i + step])[0]
            hidden_states.append(output[:, :self.info_token_num, :])
        hidden_states = torch.cat(hidden_states, dim=0)
        hidden_states = hidden_states.reshape(B, -1, self.bert_config.hidden_size)
        return hidden_states

    def extract_text_info(self, input_ids, attention_mask, token_positions):
        last_hidden_states = self.model(input_ids, attention_mask)[0]

        return last_hidden_states


    def forward(self, text_input_ids, text_attention_mask, candidates_input_ids, candidates_attention_mask,
                token_positions, labels, op="train"):

        text_extracted_info = self.extract_text_info(text_input_ids, text_attention_mask, token_positions)
        candidates_extracted_info = self.get_extracted_candidates(candidates_input_ids, candidates_attention_mask,
                                                                 )

        _, text_seq = text_input_ids.shape

        candidates_attention_mask = torch.ones((candidates_extracted_info.shape[0],
                                                candidates_extracted_info.shape[1])).to(self.device)
        attention_mask = torch.cat((text_attention_mask, candidates_attention_mask), -1)
        extened_attention_mask = self.get_extended_attention_mask(attention_mask)
        combined_info = torch.cat((text_extracted_info, candidates_extracted_info), dim=1)

        combined_info = self.model.encoder(combined_info, extened_attention_mask)[0]

        candidate_logits = self.net(combined_info[:, text_seq:, :]).squeeze(-1)
        if op == "train":
            return torch.nn.functional.binary_cross_entropy_with_logits(candidate_logits, labels)
        else:
            return torch.sigmoid(candidate_logits)




