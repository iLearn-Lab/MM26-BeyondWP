import gzip
import json
import torch
import torch.nn as nn
from habitat import Config


class InstructionEncoder(nn.Module):

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        rnn = nn.GRU if self.config.rnn_type == "GRU" else nn.LSTM
        self.encoder_rnn = rnn(
            input_size=config.embedding_size,
            hidden_size=config.hidden_size,
            bidirectional=config.bidirectional,
        )
        if config.sensor_uuid == "instruction":
            if self.config.use_pretrained_embeddings:
                self.embedding_layer = nn.Embedding.from_pretrained(
                    embeddings=self._load_embeddings(), freeze=not self.config.fine_tune_embeddings
                )
            else:
                self.embedding_layer = nn.Embedding(
                    num_embeddings=config.vocab_size,
                    embedding_dim=config.embedding_size,
                    padding_idx=0,
                )

    @property
    def output_size(self):
        return self.config.hidden_size * (1 + int(self.config.bidirectional))

    def _load_embeddings(self):
        with gzip.open(self.config.embedding_file, "rt") as f:
            embeddings = torch.tensor(json.load(f))
        return embeddings

    def forward(self, observations):
        if self.config.sensor_uuid == "instruction":
            instruction = observations["instruction"].long()
            lengths = (instruction != 0.0).long().sum(dim=1)
            instruction = self.embedding_layer(instruction)
        else:
            instruction = observations["rxr_instruction"]
        lengths = (instruction != 0.0).long().sum(dim=2)
        lengths = (lengths != 0.0).long().sum(dim=1)
        packed_seq = nn.utils.rnn.pack_padded_sequence(
            instruction, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        output, final_state = self.encoder_rnn(packed_seq)
        if self.config.rnn_type == "LSTM":
            final_state = final_state[0]
        if self.config.final_state_only:
            return final_state.squeeze(0)
        else:
            ctx = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)[0].permute(0, 2, 1)
            all_lang_masks = (ctx == 0.0).all(dim=1)
            ctx = ctx.permute(0, 2, 1)
            return (ctx, all_lang_masks)
