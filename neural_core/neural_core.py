import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TokenInputLayer(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => num_heads x d_k
        query = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        key = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        value = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 2) Apply attention on all the projected vectors in batch.
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        x = torch.matmul(p_attn, value)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        return self.w_o(x)

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class ExpertRouter(nn.Module):
    def __init__(self, d_model, num_experts, top_k=2):
        super().__init__()
        self.top_k = top_k
        self.gate = nn.Linear(d_model, num_experts)
        self.experts = nn.ModuleList([PositionwiseFeedForward(d_model, d_model * 2) for _ in range(num_experts)])

    def forward(self, x):
        logits = self.gate(x)
        top_k_logits, top_k_indices = torch.topk(logits, self.top_k, dim=-1)
        zeros = torch.full_like(logits, -1e9)
        sparse_logits = zeros.scatter(-1, top_k_indices, top_k_logits)
        expert_weights = F.softmax(sparse_logits, dim=-1)

        output = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            # Select tokens routed to this expert
            idx = top_k_indices == i
            if idx.any():
                expert_output = expert(x[idx])
                output[idx] += expert_output * expert_weights[idx, i].unsqueeze(-1)
        return output

class EmotionNet(nn.Module):
    def __init__(self, d_model, num_emotions=7):
        super().__init__()
        self.emotion_embedding = nn.Linear(d_model, num_emotions)
        self.emotion_predictor = nn.Linear(num_emotions, d_model)

    def forward(self, x):
        # Simple emotion detection and injection
        emotion_logits = self.emotion_embedding(x.mean(dim=1))
        emotion_vector = self.emotion_predictor(F.softmax(emotion_logits, dim=-1))
        return x + emotion_vector.unsqueeze(1) # Add emotion vector to each token

class MemoryFusionLayer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.linear = nn.Linear(d_model * 2, d_model)

    def forward(self, x, memory_vector):
        # Concatenate current input with memory and project back to d_model
        # memory_vector is assumed to be broadcastable or of compatible shape
        if memory_vector.dim() == 2:
            memory_vector = memory_vector.unsqueeze(1).expand(-1, x.size(1), -1)
        fused_input = torch.cat([x, memory_vector], dim=-1)
        return self.linear(fused_input)

class KnowledgeFilter(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.filter_layer = nn.Linear(d_model, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Simple truth/knowledge filtering based on a learned gate
        gate = self.sigmoid(self.filter_layer(x))
        return x * gate # Element-wise multiplication to filter/emphasize

class OutputPredictor(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return self.linear(x)

class SathikNeuralCore(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, num_experts, top_k):
        super().__init__()
        self.token_input_layer = TokenInputLayer(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "self_attn": MultiHeadAttention(d_model, num_heads),
                "feed_forward": PositionwiseFeedForward(d_model, d_model * 4),
                "expert_router": ExpertRouter(d_model, num_experts, top_k),
                "emotion_net": EmotionNet(d_model),
                "memory_fusion": MemoryFusionLayer(d_model),
                "knowledge_filter": KnowledgeFilter(d_model),
                "norm1": nn.LayerNorm(d_model),
                "norm2": nn.LayerNorm(d_model),
                "norm3": nn.LayerNorm(d_model),
                "norm4": nn.LayerNorm(d_model),
                "norm5": nn.LayerNorm(d_model),
                "dropout1": nn.Dropout(0.1),
                "dropout2": nn.Dropout(0.1),
                "dropout3": nn.Dropout(0.1),
                "dropout4": nn.Dropout(0.1),
                "dropout5": nn.Dropout(0.1),
            })
            for _ in range(num_layers)
        ])
        self.output_predictor = OutputPredictor(d_model, vocab_size)

    def forward(self, src, memory_vector=None, src_mask=None):
        x = self.token_input_layer(src)
        x = self.positional_encoding(x)

        for layer in self.layers:
            # Multi-Head Attention
            x = layer["norm1"](x + layer["dropout1"](layer["self_attn"](x, x, x, src_mask)))

            # Expert Router (MoE)
            x = layer["norm2"](x + layer["dropout2"](layer["expert_router"](x)))

            # Emotion-Net
            x = layer["norm3"](x + layer["dropout3"](layer["emotion_net"](x)))

            # Memory Fusion Layer
            if memory_vector is not None:
                x = layer["norm4"](x + layer["dropout4"](layer["memory_fusion"](x, memory_vector)))

            # Knowledge Filter
            x = layer["norm5"](x + layer["dropout5"](layer["knowledge_filter"](x)))

            # Deep Feedforward (after all other transformations)
            x = layer["norm5"](x + layer["dropout5"](layer["feed_forward"](x)))

        return self.output_predictor(x)

# Example usage:
if __name__ == "__main__":
    vocab_size = 30000
    d_model = 512
    num_heads = 8
    num_layers = 6
    num_experts = 8
    top_k = 2

    model = SathikNeuralCore(vocab_size, d_model, num_heads, num_layers, num_experts, top_k)
    print(model)

    # Create a dummy input (batch_size, sequence_length)
    batch_size = 2
    sequence_length = 10
    dummy_input = torch.randint(0, vocab_size, (batch_size, sequence_length))

    # Dummy memory vector (e.g., from a memory system)
    dummy_memory_vector = torch.randn(batch_size, d_model)

    output = model(dummy_input, memory_vector=dummy_memory_vector)
    print(f"Output shape: {output.shape}") # Expected: (batch_size, sequence_length, vocab_size)


