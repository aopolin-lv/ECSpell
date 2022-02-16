from transformers.configuration_utils import PretrainedConfig


class TransformerConfig(PretrainedConfig):
    def __init__(
            self,
            vocab_size=21128,
            hidden_size=1536,
            num_hidden_layers=2,
            num_attention_heads=24,
            intermediate_size=6144,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=512,
            layer_norm_eps=1e-12,
            type_vocab_size=2,
            initializer_range=0.02,
            **kwargs,
    ):
        super().__init__(**kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.layer_norm_eps = layer_norm_eps
