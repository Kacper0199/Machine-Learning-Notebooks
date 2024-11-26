import torch
import torch.nn as nn


class TransformerEncoder(nn.Module):
    def __init__(self, embedding_size, num_heads, hidden_layer_dim, dropout_rate):
        super(TransformerEncoder, self).__init__()
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=embedding_size,
            num_heads=num_heads
        )

        self.mlp_1 = nn.Linear(embedding_size, hidden_layer_dim)
        self.mlp_2 = nn.Linear(hidden_layer_dim, embedding_size)
        self.layer_norm_1 = nn.LayerNorm(embedding_size)
        self.layer_norm_2 = nn.LayerNorm(embedding_size)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x_norm = self.layer_norm_1(x)
        x_attention, _ = self.multihead_attention(
            query=x_norm,
            key=x_norm,
            value=x_norm
        )
        x = x + x_attention

        x_2 = self.layer_norm_2(x)
        x_2 = self.mlp_1(x_2)
        x_2 = self.activation(x_2)
        x_2 = self.dropout(x_2)
        x_2 = self.mlp_2(x_2)
        x_2 = self.dropout(x_2)
        x = x + x_2

        return x


class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, input_channels_size, embedding_size):
        super().__init__()
        self.patch_size = patch_size
        self.mlp = nn.Linear(
            patch_size * patch_size * input_channels_size,
            embedding_size
        )

    def forward(self, x):
        batch_size, channels, height, width = x.shape

        # [batch_size, channels, num_patches_h, num_patches_w, patch_size, patch_size]
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(
            3, self.patch_size, self.patch_size)

        # [batch_size, num_patches, patch_size * patch_size * channels]
        patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(
            batch_size, -1, self.patch_size * self.patch_size * channels)

        patches = self.mlp(patches)
        return patches


class VisionTransformer(nn.Module):
    def __init__(self, image_size, patch_size, emb_input_channels_size, embedding_size, num_transformers, hidden_layer_dim_transformer, num_heads, num_classes, dropout_rate):
        super().__init__()
        self.num_patches = (image_size // patch_size) ** 2

        self.positional_encoding = nn.Parameter(
            torch.randn(
                1,
                self.num_patches + 1,  # +1 for class token
                embedding_size
            )
        )

        self.cls_token = nn.Parameter(
            torch.randn(
                1,
                1,
                embedding_size
            )
        )

        self.patch_embedding = PatchEmbedding(
            patch_size,
            emb_input_channels_size,
            embedding_size
        )

        transformers = [
            TransformerEncoder(
                embedding_size=embedding_size,
                num_heads=num_heads,
                hidden_layer_dim=hidden_layer_dim_transformer,
                dropout_rate=dropout_rate
            )
        ] * num_transformers

        self.transformers = nn.Sequential(*transformers)
        self.mlp = nn.Linear(embedding_size, num_classes)
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(embedding_size)  # for class token

    def forward(self, x):
        batch_size = x.shape[0]

        patches = self.patch_embedding(x)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, patches), dim=1)

        x += self.positional_encoding

        x = self.dropout(x)
        x = self.transformers(x)

        # classification only for the class token
        class_token_output = x[:, 0, :]

        class_token_output = self.layer_norm(class_token_output)
        x = self.mlp(class_token_output)
        return x

    # def forward(self, x):
    #     x = self.patch_embedding(x)
    #     x += self.positional_encoding
    #     x = self.dropout(x)
    #     x = self.transformers(x)
    #     x = x.mean(dim=1)
    #     x = self.mlp(x)
    #     return x
