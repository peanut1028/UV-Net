import pytorch_lightning as pl
from typing import Optional, List
import torch
import admin_torch
from torch import nn
from torch.nn import LayerNorm as LN
from torch.nn import BatchNorm1d as BN
from torch_geometric.utils import unbatch_edge_index
from torch_geometric.utils import to_dense_batch
import torch.nn.functional as F
from utils import *



def get_adj_mask_from_edge_index_node(
    edge_index, batch_size, max_items, batch_mapping, xformers_or_torch_attn, use_bfloat16=True, device="cuda:0"
):
    if xformers_or_torch_attn in ["torch"]:
        empty_mask_fill_value = False
        mask_dtype = torch.bool
        edge_mask_fill_value = True
    else:
        empty_mask_fill_value = -99999
        mask_dtype = torch.bfloat16 if use_bfloat16 else torch.float32
        edge_mask_fill_value = 0

    adj_mask = torch.full(
        size=(batch_size, max_items, max_items),
        fill_value=empty_mask_fill_value,
        device=device,
        dtype=mask_dtype,
        requires_grad=False,
    )

    edge_index_unbatched = unbatch_edge_index(edge_index, batch_mapping)
    edge_batch_non_cumulative = torch.cat(edge_index_unbatched, dim=1)

    edge_batch_mapping = batch_mapping.index_select(0, edge_index[0, :])

    adj_mask[
        edge_batch_mapping, edge_batch_non_cumulative[0, :], edge_batch_non_cumulative[1, :]
    ] = edge_mask_fill_value

    if xformers_or_torch_attn in ["torch"]:
        adj_mask = ~adj_mask

    adj_mask = adj_mask.unsqueeze(1)
    return adj_mask


def create_edge_adjacency_mask(edge_index, num_edges):
    # Get all the nodes in the edge index (source and target separately for undirected graphs)
    source_nodes = edge_index[0]
    target_nodes = edge_index[1]

    # Create expanded versions of the source and target node tensors
    expanded_source_nodes = source_nodes.unsqueeze(1).expand(-1, num_edges)
    expanded_target_nodes = target_nodes.unsqueeze(1).expand(-1, num_edges)

    # Create the adjacency mask where an edge is adjacent if either node matches either node of other edges
    source_adjacency = expanded_source_nodes == expanded_source_nodes.t()
    target_adjacency = expanded_target_nodes == expanded_target_nodes.t()
    cross_adjacency = (expanded_source_nodes == expanded_target_nodes.t()) | (
        expanded_target_nodes == expanded_source_nodes.t()
    )

    adjacency_mask = source_adjacency | target_adjacency | cross_adjacency

    # Mask out self-adjacency by setting the diagonal to False
    adjacency_mask.fill_diagonal_(0)  # We use "0" here to indicate False in PyTorch boolean context

    return adjacency_mask


def get_first_unique_index(t):
    # This is taken from Stack Overflow :)
    unique, idx, counts = torch.unique(t, sorted=True, return_inverse=True, return_counts=True)
    _, ind_sorted = torch.sort(idx, stable=True)
    cum_sum = counts.cumsum(0)

    zero = torch.tensor([0], device=torch.device("cuda:0"))
    cum_sum = torch.cat((zero, cum_sum[:-1]))
    first_indicies = ind_sorted[cum_sum]

    return first_indicies


def generate_consecutive_tensor(input_tensor, final):
    # Calculate the length of each segment
    lengths = input_tensor[1:] - input_tensor[:-1]

    # Append the final length
    lengths = torch.cat((lengths, torch.tensor([final - input_tensor[-1]], device=torch.device("cuda:0"))))

    # Create ranges for each segment
    ranges = [torch.arange(0, length, device=torch.device("cuda:0")) for length in lengths]

    # Concatenate all ranges into a single tensor
    result = torch.cat(ranges)

    return result

# This is needed if the standard "nonzero" method from PyTorch fails
# This alternative is slower but allows bypassing the problem until 64-bit
# support is available
def nonzero_chunked(ten, num_chunks):
    # This is taken from this pull request
    # https://github.com/facebookresearch/segment-anything/pull/569/files
    b, w_h = ten.shape
    total_elements = b * w_h

    # Maximum allowable elements in one chunk - as torch is using 32 bit integers for this function
    max_elements_per_chunk = 2**31 - 1

    # Calculate the number of chunks needed
    if total_elements % max_elements_per_chunk != 0:
        num_chunks += 1

    # Calculate the actual chunk size
    chunk_size = b // num_chunks
    if b % num_chunks != 0:
        chunk_size += 1

    # List to store the results from each chunk
    all_indices = []

    # Loop through the diff tensor in chunks
    for i in range(num_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, b)
        chunk = ten[start:end, :]

        # Get non-zero indices for the current chunk
        indices = chunk.nonzero()

        # Adjust the row indices to the original tensor
        indices[:, 0] += start

        all_indices.append(indices)

    # Concatenate all the results
    change_indices = torch.cat(all_indices, dim=0)

    return change_indices


def get_adj_mask_from_edge_index_edge(
    edge_index,
    batch_size,
    max_items,
    batch_mapping,
    xformers_or_torch_attn,
    use_bfloat16=True,
    device="cuda:0",
):
    if xformers_or_torch_attn in ["torch"]:
        empty_mask_fill_value = False
        mask_dtype = torch.bool
        edge_mask_fill_value = True
    else:
        empty_mask_fill_value = -99999
        mask_dtype = torch.bfloat16 if use_bfloat16 else torch.float32
        edge_mask_fill_value = 0

    adj_mask = torch.full(
        size=(batch_size, max_items, max_items),
        fill_value=empty_mask_fill_value,
        device=device,
        dtype=mask_dtype,
        requires_grad=False,
    )

    edge_batch_mapping = batch_mapping.index_select(0, edge_index[0, :])

    edge_adj_matrix = create_edge_adjacency_mask(edge_index, edge_index.shape[1])

    edge_batch_index_to_original_index = generate_consecutive_tensor(
        get_first_unique_index(edge_batch_mapping), edge_batch_mapping.shape[0]
    )

    try:
        eam_nonzero = edge_adj_matrix.nonzero()
    except:
        # Adjust chunk size as desired
        eam_nonzero = nonzero_chunked(edge_adj_matrix, 3)

    adj_mask[
        edge_batch_mapping.index_select(0, eam_nonzero[:, 0]),
        edge_batch_index_to_original_index.index_select(0, eam_nonzero[:, 0]),
        edge_batch_index_to_original_index.index_select(0, eam_nonzero[:, 1]),
    ] = edge_mask_fill_value


    if xformers_or_torch_attn in ["torch"]:
        adj_mask = ~adj_mask

    adj_mask = adj_mask.unsqueeze(1)
    return adj_mask



class SABComplete(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        num_heads,
        dropout,
        idx,
        norm_type,
        use_mlp=False,
        mlp_hidden_size=64,
        mlp_type="standard",
        node_or_edge="edge",
        xformers_or_torch_attn="xformers",
        residual_dropout=0,
        set_max_items=0,
        use_bfloat16=True,
        num_mlp_layers=3,
        pre_or_post="pre",
        num_layers_for_residual=0,
        use_mlp_ln=False,
        mlp_dropout=0,
    ):
        super(SABComplete, self).__init__()

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.num_heads = num_heads
        self.use_mlp = use_mlp
        self.idx = idx
        self.mlp_hidden_size = mlp_hidden_size
        self.node_or_edge = node_or_edge
        self.xformers_or_torch_attn = xformers_or_torch_attn
        self.residual_dropout = residual_dropout
        self.set_max_items = set_max_items
        self.use_bfloat16 = use_bfloat16
        self.num_mlp_layers = num_mlp_layers
        self.pre_or_post = pre_or_post

        if self.pre_or_post == "post":
            self.residual_attn = admin_torch.as_module(num_layers_for_residual)
            self.residual_mlp = admin_torch.as_module(num_layers_for_residual)

        if dim_in != dim_out:
            self.proj_1 = nn.Linear(dim_in, dim_out)
    

        self.sab = SAB(dim_in, dim_out, num_heads, dropout, xformers_or_torch_attn)

        if self.idx != 2:
            bn_dim = self.set_max_items
        else:
            bn_dim = 32

        if norm_type == "LN":
            if self.pre_or_post == "post":
                self.norm = LN(dim_out)
            else:
                self.norm = LN(dim_in)
                    
        elif norm_type == "BN":
            self.norm = BN(bn_dim)

        self.mlp_type = mlp_type

        if self.use_mlp:
            if self.mlp_type == "standard":
                self.mlp = SmallMLP(
                    in_dim=dim_out,
                    out_dim=dim_out,
                    inter_dim=mlp_hidden_size,
                    dropout_p=mlp_dropout,
                    num_layers=num_mlp_layers,
                    use_ln=use_mlp_ln,
                )
            
            elif self.mlp_type == "gated_mlp":
                self.mlp = GatedMLPMulti(
                    in_dim=dim_out,
                    out_dim=dim_out,
                    inter_dim=mlp_hidden_size,
                    dropout_p=mlp_dropout,
                    num_layers=num_mlp_layers,
                    use_ln=use_mlp_ln,
                )

        if norm_type == "LN":
            if self.idx != 2:
                self.norm_mlp = LN(dim_out, num_elements=self.set_max_items)
            else:
                self.norm_mlp = LN(dim_out)
                
        elif norm_type == "BN":
            self.norm_mlp = BN(bn_dim)


    def forward(self, inp):
        X, edge_index, batch_mapping, max_items, adj_mask = inp

        if self.pre_or_post == "pre":
            X = self.norm(X)

        if self.idx == 1:
            out_attn = self.sab(X, adj_mask)
        else:
            out_attn = self.sab(X, None)

        if out_attn.shape[-1] != X.shape[-1]:
            X = self.proj_1(X)

        if self.pre_or_post == "pre":
            out = X + out_attn
        
        if self.pre_or_post == "post":
            out = self.residual_attn(X, out_attn)
            out = self.norm(out)

        if self.use_mlp:
            if self.pre_or_post == "pre":
                out_mlp = self.norm_mlp(out)
                out_mlp = self.mlp(out_mlp)
                if out.shape[-1] == out_mlp.shape[-1]:
                    out = out_mlp + out

            if self.pre_or_post == "post":
                out_mlp = self.mlp(out)
                if out.shape[-1] == out_mlp.shape[-1]:
                    out = self.residual_mlp(out, out_mlp)
                out = self.norm_mlp(out)

        if self.residual_dropout > 0:
            out = F.dropout(out, p=self.residual_dropout)

        return out, edge_index, batch_mapping, max_items, adj_mask


class PMAComplete(nn.Module):
    def __init__(
        self,
        dim_hidden,
        num_heads,
        num_outputs,
        norm_type,
        dropout=0,
        use_mlp=False,
        mlp_hidden_size=64,
        mlp_type="standard",
        xformers_or_torch_attn="xformers",
        set_max_items=0,
        use_bfloat16=True,
        num_mlp_layers=3,
        pre_or_post="pre",
        num_layers_for_residual=0,
        residual_dropout=0,
        use_mlp_ln=False,
        mlp_dropout=0,
    ):
        super(PMAComplete, self).__init__()

        self.use_mlp = use_mlp
        self.mlp_hidden_size = mlp_hidden_size
        self.num_heads = num_heads
        self.xformers_or_torch_attn = xformers_or_torch_attn
        self.set_max_items = set_max_items
        self.use_bfloat16 = use_bfloat16
        self.residual_dropout = residual_dropout
        self.num_mlp_layers = num_mlp_layers
        self.pre_or_post = pre_or_post

        if self.pre_or_post == "post":
            self.residual_attn = admin_torch.as_module(num_layers_for_residual)
            self.residual_mlp = admin_torch.as_module(num_layers_for_residual)

        self.pma = PMA(dim_hidden, num_heads, num_outputs, dropout, xformers_or_torch_attn)

        if norm_type == "LN":
            self.norm = LN(dim_hidden)
        elif norm_type == "BN":
            self.norm = BN(self.set_max_items)

        self.mlp_type = mlp_type

        if self.use_mlp:
            if self.mlp_type == "standard":
                self.mlp = SmallMLP(
                    in_dim=dim_hidden,
                    out_dim=dim_hidden,
                    inter_dim=mlp_hidden_size,
                    dropout_p=mlp_dropout,
                    num_layers=num_mlp_layers,
                    use_ln=use_mlp_ln,
                )

            elif self.mlp_type == "gated_mlp":
                self.mlp = GatedMLPMulti(
                    in_dim=dim_hidden,
                    out_dim=dim_hidden,
                    inter_dim=mlp_hidden_size,
                    dropout_p=mlp_dropout,
                    num_layers=num_mlp_layers,
                    use_ln=use_mlp_ln,
                )

        if norm_type == "LN":
            self.norm_mlp = LN(dim_hidden)
        elif norm_type == "BN":
            self.norm_mlp = BN(32)


    def forward(self, inp):
        X, edge_index, batch_mapping, max_items, adj_mask = inp

        if self.pre_or_post == "pre":
            X = self.norm(X)

        out_attn = self.pma(X)

        if self.pre_or_post == "pre" and out_attn.shape[-2] == X.shape[-2]:
            out = X + out_attn
        
        elif self.pre_or_post == "post" and out_attn.shape[-2] == X.shape[-2]:
            out = self.residual_attn(X, out_attn)
            out = self.norm(out)
        
        else:
            out = out_attn

        if self.use_mlp:
            if self.pre_or_post == "pre":
                out_mlp = self.norm_mlp(out)
                out_mlp = self.mlp(out_mlp)
                if out.shape[-2] == out_mlp.shape[-2]:
                    out = out_mlp + out

            if self.pre_or_post == "post":
                out_mlp = self.mlp(out)
                if out.shape[-2] == out_mlp.shape[-2]:
                    out = self.residual_mlp(out, out_mlp)
                out = self.norm_mlp(out)

        if self.residual_dropout > 0:
            out = F.dropout(out, p=self.residual_dropout)

        return out, edge_index, batch_mapping, max_items, adj_mask



class ESA(nn.Module):
    def __init__(
        self,
        num_outputs,
        dim_output,
        dim_hidden,
        num_heads,
        layer_types,
        node_or_edge="edge",
        xformers_or_torch_attn="xformers",
        pre_or_post="pre",
        norm_type="LN",
        sab_dropout=0.0,
        mab_dropout=0.0,
        pma_dropout=0.0,
        residual_dropout=0.0,
        pma_residual_dropout=0.0,
        use_mlps=False,
        mlp_hidden_size=64,
        num_mlp_layers=2,
        mlp_type="gated_mlp",
        mlp_dropout=0.0,
        use_mlp_ln=False,
        set_max_items=0,
        use_bfloat16=True,
    ):
        super(ESA, self).__init__()

        assert len(layer_types) == len(dim_hidden) and len(layer_types) == len(num_heads)

        self.dim_hidden = dim_hidden
        self.num_heads = num_heads
        self.layer_types = layer_types
        self.node_or_edge = node_or_edge
        self.xformers_or_torch_attn = xformers_or_torch_attn
        self.pre_or_post = pre_or_post
        self.norm_type = norm_type
        self.sab_dropout = sab_dropout
        self.mab_dropout = mab_dropout
        self.pma_dropout = pma_dropout
        self.residual_dropout = residual_dropout
        self.pma_residual_dropout = pma_residual_dropout
        self.use_mlps = use_mlps
        self.mlp_hidden_size = mlp_hidden_size
        self.num_mlp_layers = num_mlp_layers
        self.mlp_type = mlp_type
        self.mlp_dropout = mlp_dropout
        self.use_mlp_ln = use_mlp_ln
        self.set_max_items = set_max_items
        self.use_bfloat16 = use_bfloat16
        
        layer_tracker = 0

        self.encoder = []

        pma_encountered = False
        dim_pma = -1

        has_pma = "P" in self.layer_types

        for lt in self.layer_types:
            layer_in_dim = dim_hidden[layer_tracker]
            layer_num_heads = num_heads[layer_tracker]
            if lt != "P":
                if has_pma:
                    layer_out_dim = dim_hidden[layer_tracker + 1]
                else:
                    layer_out_dim = dim_hidden[layer_tracker]
            else:
                layer_out_dim = -1

            if lt == "S" and not pma_encountered:
                self.encoder.append(
                    SABComplete(
                        layer_in_dim,
                        layer_out_dim,
                        layer_num_heads,
                        idx=0,
                        dropout=sab_dropout,
                        node_or_edge=node_or_edge,
                        xformers_or_torch_attn=xformers_or_torch_attn,
                        pre_or_post=pre_or_post,
                        residual_dropout=residual_dropout,
                        use_mlp=use_mlps,
                        mlp_hidden_size=mlp_hidden_size,
                        mlp_dropout=mlp_dropout,
                        num_mlp_layers=num_mlp_layers,
                        use_mlp_ln=use_mlp_ln,
                        norm_type=norm_type,
                        mlp_type=mlp_type,
                        set_max_items=set_max_items,
                        use_bfloat16=use_bfloat16,
                        num_layers_for_residual=len(dim_hidden) * 2,
                    )
                )
                
                print(f"Added encoder SAB ({layer_in_dim}, {layer_out_dim}, {layer_num_heads})")

            if lt == "M" and not pma_encountered:
                self.encoder.append(
                    SABComplete(
                        layer_in_dim,
                        layer_out_dim,
                        layer_num_heads,
                        idx=1,
                        dropout=mab_dropout,
                        node_or_edge=node_or_edge,
                        xformers_or_torch_attn=xformers_or_torch_attn,
                        pre_or_post=pre_or_post,
                        residual_dropout=residual_dropout,
                        use_mlp=use_mlps,
                        mlp_hidden_size=mlp_hidden_size,
                        mlp_dropout=mlp_dropout,
                        num_mlp_layers=num_mlp_layers,
                        use_mlp_ln=use_mlp_ln,
                        norm_type=norm_type,
                        mlp_type=mlp_type,
                        set_max_items=set_max_items,
                        use_bfloat16=use_bfloat16,
                        num_layers_for_residual=len(dim_hidden) * 2,
                    )
                )
                
                print(f"Added encoder MAB ({layer_in_dim}, {layer_out_dim}, {layer_num_heads})")
                
            if lt == "P":
                pma_encountered = True
                dim_pma = layer_in_dim
                self.decoder = [
                    PMAComplete(
                        layer_in_dim,
                        layer_num_heads,
                        num_outputs,
                        dropout=pma_dropout,
                        residual_dropout=pma_residual_dropout,
                        xformers_or_torch_attn=xformers_or_torch_attn,
                        pre_or_post=pre_or_post,
                        use_mlp=use_mlps,
                        mlp_hidden_size=mlp_hidden_size,
                        mlp_dropout=mlp_dropout,
                        num_mlp_layers=num_mlp_layers,
                        use_mlp_ln=use_mlp_ln,
                        norm_type=norm_type,
                        mlp_type=mlp_type,
                        set_max_items=set_max_items,
                        use_bfloat16=use_bfloat16,
                        num_layers_for_residual=len(dim_hidden) * 2,
                    )
                ]

                print(f"Added decoder PMA ({layer_in_dim}, {layer_num_heads})")

            if lt == "S" and pma_encountered:
                self.decoder.append(
                    SABComplete(
                        layer_in_dim,
                        layer_out_dim,
                        layer_num_heads,
                        idx=2,
                        dropout=sab_dropout,
                        node_or_edge=node_or_edge,
                        xformers_or_torch_attn=xformers_or_torch_attn,
                        pre_or_post=pre_or_post,
                        residual_dropout=residual_dropout,
                        use_mlp=use_mlps,
                        mlp_hidden_size=mlp_hidden_size,
                        mlp_dropout=mlp_dropout,
                        num_mlp_layers=num_mlp_layers,
                        use_mlp_ln=use_mlp_ln,
                        norm_type=norm_type,
                        mlp_type=mlp_type,
                        set_max_items=set_max_items,
                        use_bfloat16=use_bfloat16,
                        num_layers_for_residual=len(dim_hidden) * 2,
                    )
                )

                print(f"Added decoder SAB ({layer_in_dim}, {layer_out_dim}, {layer_num_heads})")

            if lt != "P":
                layer_tracker += 1

        self.encoder = nn.Sequential(*self.encoder)
        if pma_encountered:
            self.decoder = nn.Sequential(*self.decoder)

        self.decoder_linear = nn.Linear(dim_hidden[-1], dim_output, bias=True)

        if has_pma and dim_hidden[0] != dim_pma:
            self.out_proj = nn.Linear(dim_hidden[0], dim_pma)

            self.dim_pma = dim_pma


    def forward(self, X, edge_index, batch_mapping, num_max_items):

        if self.node_or_edge == "node":
            adj_mask = get_adj_mask_from_edge_index_node(
                edge_index=edge_index,
                batch_mapping=batch_mapping,
                batch_size=X.shape[0],
                max_items=self.set_max_items,
                xformers_or_torch_attn=self.xformers_or_torch_attn,
                use_bfloat16=self.use_bfloat16,
            )
        elif self.node_or_edge == "edge":
            adj_mask = get_adj_mask_from_edge_index_edge(
                edge_index=edge_index,
                batch_mapping=batch_mapping,
                batch_size=X.shape[0],
                max_items=self.set_max_items,
                xformers_or_torch_attn=self.xformers_or_torch_attn,
                use_bfloat16=self.use_bfloat16,
            )

        enc, _, _, _, _ = self.encoder((X, edge_index, batch_mapping, num_max_items, adj_mask))
        if hasattr(self, "dim_pma") and self.dim_hidden[0] != self.dim_pma:
            X = self.out_proj(X)

        enc = enc + X

        if hasattr(self, "decoder"):
            out, _, _, _, _ = self.decoder((enc, edge_index, batch_mapping, num_max_items, adj_mask))
            out = out.mean(dim=1)
        else:
            out = enc

        return F.mish(self.decoder_linear(out))



class ESARegressor(nn.Module):
    def __init__(
        self,
        num_features,
        graph_dim,
        edge_dim,
        xformers_or_torch_attn="xformers",
        hidden_dims=None,
        num_heads=None,
        sab_dropout=0.0,
        mab_dropout=0.0,
        pma_dropout=0.0,
        use_mlps=False,
        mlp_hidden_size=64,
        mlp_type="standard",
        norm_type="LN",
        layer_types: List[str] = None,
        attn_residual_dropout=0.0,
        set_max_items=0,
        num_mlp_layers=3,
        use_bfloat16=True,
        pre_or_post="pre",
        pma_residual_dropout=0.0,
        use_mlp_ln=False,
        mlp_dropout=0.0,
        linear_output_size=1,
    ):
        super().__init__()
        self.lap_encoder = None
        self.rwse_encoder = None
        self.num_features = num_features
        self.graph_dim = graph_dim
        self.edge_dim = edge_dim
        self.hidden_dims = hidden_dims
        self.num_heads = num_heads
        self.sab_dropout = sab_dropout
        self.mab_dropout = mab_dropout
        self.pma_dropout = pma_dropout
        self.set_max_items = set_max_items
        self.use_mlps = use_mlps
        self.mlp_hidden_size = mlp_hidden_size
        self.mlp_type = mlp_type
        self.norm_type = norm_type
        self.num_mlp_layers = num_mlp_layers
        self.xformers_or_torch_attn = xformers_or_torch_attn
        self.attn_residual_dropout = attn_residual_dropout
        in_dim = self.num_features
        in_dim = in_dim * 2
        if self.edge_dim is not None:
            in_dim += self.edge_dim

        self.node_edge_mlp = SmallMLP(
            in_dim=in_dim, 
            inter_dim=128,
            out_dim=self.hidden_dims[0],
            use_ln=False,
            dropout_p=0,
            num_layers=self.num_mlp_layers,
        )
        st_args = dict(
            num_outputs=32, # this is the k for PMA
            dim_output=self.graph_dim,
            xformers_or_torch_attn=self.xformers_or_torch_attn,
            dim_hidden=self.hidden_dims,
            num_heads=self.num_heads,
            sab_dropout=self.sab_dropout,
            mab_dropout=self.mab_dropout,
            pma_dropout=self.pma_dropout,
            use_mlps=self.use_mlps,
            mlp_hidden_size=self.mlp_hidden_size,
            mlp_type=self.mlp_type,
            norm_type=self.norm_type,
            node_or_edge="edge",
            residual_dropout=self.attn_residual_dropout,
            set_max_items=nearest_multiple_of_8(self.set_max_items + 1),
            use_bfloat16=use_bfloat16,
            layer_types=layer_types,
            num_mlp_layers=self.num_mlp_layers,
            pre_or_post=pre_or_post,
            pma_residual_dropout=pma_residual_dropout,
            use_mlp_ln=use_mlp_ln,
            mlp_dropout=mlp_dropout,
        )
        self.st_fast = ESA(**st_args)
        self.output_mlp = nn.Linear(self.graph_dim, linear_output_size)

    def forward(self, 
                x, 
                edge_index, 
                edge_attr, 
                batch_mapping,
                num_max_items, 
                vars=None):
        x = x.float()
        # if self.lap_encoder is not None:
        #     lap_pos_enc = self.lap_encoder(batch.EigVals, batch.EigVecs)
        #     x = torch.cat((x, lap_pos_enc), 1)
        # if self.rwse_encoder is not None:
        #     rwse_pos_enc = self.rwse_encoder(batch.pestat_RWSE)
        #     x = torch.cat((x, rwse_pos_enc), 1)

        # ESA
        source = x[edge_index[0, :], :]
        target = x[edge_index[1, :], :]
        h = torch.cat((source, target), dim=1)

        if self.edge_dim is not None and edge_attr is not None:
            h = torch.cat((h, edge_attr.float()), dim=1)

        h = self.node_edge_mlp(h)

        edge_batch_index = batch_mapping.index_select(0, edge_index[0, :])
        h, _ = to_dense_batch(h, edge_batch_index, fill_value=0, max_num_nodes=num_max_items)
        h = self.st_fast(h, edge_index, batch_mapping, num_max_items=num_max_items)

        predictions = torch.flatten(self.output_mlp(h))

        return predictions


class Regression(pl.LightningModule):
    """
    PyTorch Lightning module to train/test the regressor.
    """

    def __init__(self, argsdict):
        """
        Args:
        """
        super().__init__()
        self.save_hyperparameters()
        self.model = ESARegressor(**argsdict)

    def forward(self, batch):
        x, edge_index, y, batch_mapping, edge_attr = batch.x, batch.edge_index, batch.y, batch.batch, batch.edge_attr
        max_node, max_edge = batch.max_node_global, batch.max_edge_global
        inputs = batch["graph"].to(self.device)
        num_max_items = max_edge
        num_max_items = torch.max(num_max_items).item()
        num_max_items = nearest_multiple_of_8(num_max_items + 1)
        # inputs.ndata["x"] = inputs.ndata["x"].permute(0, 3, 1, 2)
        # inputs.edata["x"] = inputs.edata["x"].permute(0, 2, 1)
        labels = batch["label"].to(self.device)
        vars = batch["vars"].to(self.device) 
        logits = self.model(x=x, 
                            edge_index=edge_index, 
                            edge_attr=edge_attr, 
                            batch_mapping=batch_mapping, 
                            num_max_items=num_max_items, 
                            vars=vars)
        logits = torch.squeeze(logits)
        loss = F.l1_loss(logits, labels, reduction="mean")
        preds = logits
        acc = 1 - torch.mean(torch.abs(preds - labels) / labels)
        return {"loss": loss, "acc": acc}
    
    def training_step(self, batch, batch_idx):
        return self.forward(batch)

    def validation_step(self, batch, batch_idx):
        return self.forward(batch)

    def predict_step(self, batch, batch_idx):
        inputs = batch["graph"].to(self.device)
        inputs.ndata["x"] = inputs.ndata["x"].permute(0, 3, 1, 2)
        inputs.edata["x"] = inputs.edata["x"].permute(0, 2, 1)
        labels = batch["label"].to(self.device)
        vars = batch["vars"].to(self.device)
        logits = self.model(inputs, vars)
        logits = torch.squeeze(logits)
        return {"labels": labels, "preds": logits}

    def training_epoch_end(self, outputs):
        '''
        This function is called at the end of each epoch to log the training loss and accuracy.

        Args:
            outputs (list[dict]): A list of dictionaries, where each dictionary corresponds to the output of a training_step.
        '''
        train_loss = torch.stack([x['loss'] for x in outputs]).mean()
        train_acc = torch.stack([x['acc'] for x in outputs]).mean()
        self.log("train_loss", train_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train_acc", train_acc, on_step=False, on_epoch=True, sync_dist=True)    

    def validation_epoch_end(self, outputs):
        val_loss = torch.stack([x['loss'] for x in outputs]).mean()
        val_acc = torch.stack([x['acc'] for x in outputs]).mean()
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_acc", val_acc, on_step=False, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        # scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, 1, 0.001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
        # return [optimizer], [scheduler]
        return optimizer



