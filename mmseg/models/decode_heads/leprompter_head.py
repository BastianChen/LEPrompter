from mmcv.cnn import ConvModule

from mmseg.models.builder import HEADS
from mmcv.runner import ModuleList
from torch import nn
from mmcv.cnn.bricks import build_activation_layer, build_norm_layer
from ..utils import nchw_to_nlc


class MLPBlock(nn.Module):
    def __init__(self, embedding_dim, mlp_dim, act_cfg):
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.activate = build_activation_layer(act_cfg)

    def forward(self, x):
        return self.lin2(self.activate(self.lin1(x)))


class ImagePromptTransformer(nn.Module):
    def __init__(self, depth, embedding_dim, num_heads, mlp_dim, sr_ratio=4,
                 activation=dict(type='GELU'),
                 norm_cfg=dict(type='LN')):
        """
        A transformer decoder that attends to an input image using
        queries whose positional embedding is supplied.

        Args:
          depth (int): number of layers in the transformer
          embedding_dim (int): the channel dimension for the input embeddings
          num_heads (int): the number of heads for multihead attention. Must
            divide embedding_dim
          mlp_dim (int): the channel dimension internal to the MLP block
          sr_ratio (int): The ratio of spatial reduction of Efficient Multi-head Attention. Default: 4.
          activation (nn.Module): the activation to use in the MLP block
          norm_cfg (dict): Config dict for normalization layer. Default: dict(type='LN')
        """
        super().__init__()
        self.layers = ModuleList()

        for i in range(depth):
            self.layers.append(
                ImagePromptAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    skip_first_layer_pe=(i == 0),
                )
            )

        self.attn = nn.MultiheadAttention(embedding_dim, num_heads)
        self.sr1 = ConvModule(
            in_channels=embedding_dim,
            out_channels=embedding_dim,
            kernel_size=sr_ratio,
            stride=sr_ratio)
        self.sr2 = ConvModule(
            in_channels=embedding_dim,
            out_channels=embedding_dim,
            kernel_size=sr_ratio,
            stride=sr_ratio)
        self.norm_sr1 = build_norm_layer(norm_cfg, embedding_dim)[1]
        self.norm_sr2 = build_norm_layer(norm_cfg, embedding_dim)[1]

    def forward(self, image_embedding, point_embedding, mask_embedding=None):
        """
        Args:
          image_embedding (torch.Tensor): image to attend to. Should be shape
            B x embedding_dim x h x w for any h and w.
          point_embedding (torch.Tensor): the embedding to add to the query points.
            Must have shape B x N_points x embedding_dim for any N_points.
          mask_embedding (torch.Tensor): the embedding of the mask prompt when only using the mask prompt.

        Returns:
          torch.Tensor: the processed image_embedding
        """

        image_embedding = self.sr1(image_embedding)
        image_embedding = self.norm_sr1(nchw_to_nlc(image_embedding))
        if point_embedding is None:
            mask_embedding = self.sr2(mask_embedding)
            point_embedding = self.norm_sr2(nchw_to_nlc(mask_embedding))

        # Prepare queries
        queries = point_embedding
        # efficient self-attention  point_embedding
        keys = image_embedding
        values = point_embedding

        # Apply transformer blocks and final layernorm
        for layer in self.layers:
            queries, keys, values = layer(
                queries=queries,
                keys=keys,
                values=values
            )

        return keys.permute(1, 0, 2)


class ImagePromptAttentionBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, mlp_dim, skip_first_layer_pe,
                 activation=dict(type='GELU'),
                 norm_cfg=dict(type='LN'), ):
        """
        A transformer block with four layers: (1) self-attention of sparse
        inputs, (2) cross attention of sparse inputs to dense inputs, (3) mlp
        block on sparse inputs, and (4) cross attention of dense inputs to sparse
        inputs.

        Arguments:
          embedding_dim (int): the channel dimension of the embeddings
          num_heads (int): the number of heads in the attention layers
          mlp_dim (int): the hidden dimension of the mlp block
          activation (nn.Module): the activation of the mlp block
          skip_first_layer_pe (bool): skip the PE on the first layer
          norm_cfg (dict): Config dict for normalization layer. Default: dict(type='LN')
        """
        super().__init__()
        self.attn = nn.MultiheadAttention(embedding_dim, num_heads)
        point_embeddings = [build_norm_layer(norm_cfg, embedding_dim)[1] for _ in range(4)]
        self.norm_layers = ModuleList(point_embeddings)
        self.mlp = MLPBlock(embedding_dim, mlp_dim, activation)
        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(self, queries, keys, values):
        if self.skip_first_layer_pe:
            queries = queries.transpose(0, 1)
            keys = keys.transpose(0, 1)
            values = values.transpose(0, 1)
        # Self attention block
        if self.skip_first_layer_pe:
            queries = self.attn(query=queries, key=queries, value=queries)[0]
        else:
            q = queries + values
            attn_out = self.attn(query=q, key=q, value=queries)[0]
            queries = queries + attn_out
        queries = self.norm_layers[0](queries)

        # Cross attention block, tokens attending to image embedding
        q = queries + values
        attn_out = self.attn(query=q, key=keys, value=keys)[0]
        queries = queries + attn_out
        queries = self.norm_layers[1](queries)

        # MLP block
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm_layers[2](queries)

        # Cross attention block, image embedding attending to tokens
        q = queries + values
        attn_out = self.attn(query=keys, key=q, value=queries)[0]
        keys = keys + attn_out
        keys = self.norm_layers[3](keys)

        return queries, keys, values


@HEADS.register_module()
class LEPrompterHead(nn.Module):
    def __init__(self, depth=2, embedding_dim=192, num_heads=4, mlp_dim=384, sr_ratio=4):
        """
        integrating image embedding with prompt tokens using a Transformer architecture.

        Arguments:
            depth (int): number of layers in the transformer
            embedding_dim (int): the channel dimension of the embeddings
            num_heads (int): the number of heads in the attention layers
            mlp_dim (int): the hidden dimension of the mlp block
            sr_ratio (int): The ratio of spatial reduction of Efficient Multi-head Attention.
        """
        super().__init__()
        self.transformer = ImagePromptTransformer(depth=depth, embedding_dim=embedding_dim, num_heads=num_heads,
                                                  mlp_dim=mlp_dim, sr_ratio=sr_ratio)

    def forward(
            self,
            image_embeddings,
            sparse_prompt_embeddings,
            dense_prompt_embeddings=None,
    ):
        """
        Predict masks given image and prompt embeddings.

        Arguments:
            image_embeddings (torch.Tensor): the embeddings from the image encoder
            sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
            dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs

        Returns:
          torch.Tensor: the output token after fusing image embedding with prompt tokens.
        """

        if dense_prompt_embeddings is not None and sparse_prompt_embeddings is not None:
            image_embeddings += dense_prompt_embeddings

        mask_tokens = self.transformer(image_embeddings, sparse_prompt_embeddings, dense_prompt_embeddings)

        return mask_tokens
