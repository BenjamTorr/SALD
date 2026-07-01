import torch.nn as nn
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv

class GraphDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, 
                 down_sample=False, num_heads=4, num_layers=2, 
                 use_attention=True):
        super(GraphDownBlock, self).__init__()
        self.num_layers = num_layers
        self.use_attention = use_attention
        self.down_sample = down_sample
        
        self.res_gcn_layers = nn.ModuleList()
        self.res_linear_proj = nn.ModuleList()
        
        self.attn_norms = nn.ModuleList()
        self.attn_layers = nn.ModuleList()

        for i in range(num_layers):
            in_dim = in_channels if i == 0 else out_channels
            self.res_gcn_layers.append(GCNConv(in_dim, out_channels))
            self.res_linear_proj.append(nn.Linear(in_dim, out_channels))
            
            if use_attention:
                self.attn_norms.append(nn.LayerNorm(out_channels))
                self.attn_layers.append(GATConv(out_channels, out_channels // num_heads, heads=num_heads, concat=True, edge_dim= 1))

        if down_sample:
            self.down_proj = nn.Linear(out_channels, out_channels)
        else:
            self.down_proj = nn.Identity()

    def forward(self, x, edge_index, edge_weight, edge_attr):
        for i in range(self.num_layers):
            residual = self.res_linear_proj[i](x)
            x = F.silu(self.res_gcn_layers[i](x, edge_index, edge_weight = edge_weight))
            x = x + residual

            if self.use_attention:
                x = self.attn_norms[i](x)
                x = F.silu(self.attn_layers[i](x, edge_index, edge_attr = edge_attr)) + x

        x = self.down_proj(x)  # Optional linear projection
        return x

    
class SCGraphModel1D(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.in_channels = args['in_channels']
        self.down_channels = args['down_channels']
        self.out_channels = args['out_channels']
        self.latent_dim = args['latent_dim']
        self.num_heads = args['num_heads']
        self.num_down_layers = args['num_down_layers']
        self.down_sample = args['down_sample']
        self.attns = args['attn_down']
        self.latent_dim = args['latent_dim']
        
        self.encoder_layers = nn.ModuleList([])
        self.enconder_conv_in = GCNConv(self.in_channels, self.down_channels[0])
        for i in range(len(self.down_channels) - 1):
            self.encoder_layers.append(GraphDownBlock(in_channels = self.down_channels[i], out_channels=self.down_channels[i + 1],
                                                 down_sample=self.down_sample[i],
                                                 num_heads=self.num_heads,
                                                 num_layers=self.num_down_layers,
                                                 use_attention=self.attns[i]))
        #self.encoder_conv_out = nn.Identity()  #GCNConv(self.down_channels[-1], out_channels=self.out_channels)
        #self.projector = nn.Linear(self.in_channels, self.latent_dim)

        self.row_queries = nn.Parameter(torch.randn(self.out_channels, self.down_channels[-1]))  # (4, node_dim)
        self.to_latent = nn.Linear(self.down_channels[-1], self.latent_dim) 

    def forward(self, x, edge_index, edge_weight, edge_attr):  #   
        x = self.enconder_conv_in(x, edge_index, edge_weight=edge_weight)
        for layer in self.encoder_layers:
            x = layer(x, edge_index, edge_weight, edge_attr)
        #x = self.encoder_conv_out(x, edge_index, edge_weight=edge_weight)

        # Reshape per graph: assume you know B
        B = x.shape[0] // self.in_channels  # B is the batch size, which is the number of graphs
        x = x.view(B, self.in_channels, self.down_channels[-1])             # [B, in_channels, node_dim]
        #x = x.view(B, 128, self.in_channels)             # [B, out_channels, in_channels]
        #x = self.projector(x)            # [B, out_channels, latent_dim * latent_dim]
        #x = x.view(B, self.out_channels, self.latent_dim)         # [B, out_channels,  latent_dim]
        Q = self.row_queries.unsqueeze(0).expand(B, -1, -1)     # (B, 4, node_dim)
        attn = torch.softmax(Q @ x.transpose(1,2) / (self.down_channels[-1]**0.5), dim=-1)  # (B,4,N)
        rows = attn @ x                                   # (B,4,node_dim)

        z = self.to_latent(rows)                                # (B,4,643)
        return z
    




"""
class SCGraphModel(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.in_channels = args['in_channels']
        self.down_channels = args['down_channels']
        self.out_channels = args['out_channels']
        self.latent_dim = args['latent_dim']
        self.num_heads = args['num_heads']
        self.num_down_layers = args['num_down_layers']
        self.down_sample = args['down_sample']
        self.attns = args['attn_down']
        self.latent_dim = args['latent_dim']
        
        self.encoder_layers = nn.ModuleList([])
        self.enconder_conv_in = GCNConv(self.in_channels, self.down_channels[0])
        for i in range(len(self.down_channels) - 1):
            self.encoder_layers.append(GraphDownBlock(in_channels = self.down_channels[i], out_channels=self.down_channels[i + 1],
                                                 down_sample=self.down_sample[i],
                                                 num_heads=self.num_heads,
                                                 num_layers=self.num_down_layers,
                                                 use_attention=self.attns[i]))
        self.encoder_conv_out = GCNConv(self.down_channels[-1], out_channels=self.out_channels)

        self.projector = nn.Linear(self.in_channels, self.latent_dim * self.latent_dim)

    def forward(self, x, edge_index):  #   
        x = self.enconder_conv_in(x, edge_index)
        for layer in self.encoder_layers:
            x = layer(x, edge_index)
        x = self.encoder_conv_out(x, edge_index)

        # Reshape per graph: assume you know B
        B = x.shape[0] // self.in_channels  # B is the batch size, which is the number of graphs
        x = x.view(B, self.out_channels, self.in_channels)             # [B, out_channels, in_channels]
        x = self.projector(x)            # [B, out_channels, latent_dim * latent_dim]
        x = x.view(B, self.out_channels, self.latent_dim, self.latent_dim)         # [B, out_channels, latent_dim, latent_dim]
        return x
"""