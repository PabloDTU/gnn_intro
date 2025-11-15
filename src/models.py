import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, AttentionalAggregation, JumpingKnowledge, NNConv
# from torch_geometric.nn import GlobalAttention

class GCN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels=64, dropout=0.1):
        super(GCN, self).__init__()

        # Graph convolutions
        self.num_layers = 2  # adjust when adding more layers later
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

        # NEW: Normalisation layers (one per convolution) & dropout
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.dropout = dropout
        
        # NEW: Attention-based pooling
        self.att_gate = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels * self.num_layers, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, 1)
        )
        # self.att_pool = GlobalAttention(gate_nn=self.att_gate)
        self.att_pool = AttentionalAggregation(gate_nn=self.att_gate)

        # NEW: Jumping Knowledge to combine multi-layer representations
        self.jk = JumpingKnowledge(mode='cat', channels=hidden_channels, num_layers=self.num_layers)

        # Linear readout
        # self.linear = torch.nn.Linear(hidden_channels, 1) # old: linear layer input dim = hidden_channels
        self.linear = torch.nn.Linear(hidden_channels * self.num_layers, 1) # Update linear layer input dimension (because we concatenate in JK)



    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        layer_outs = []  # store each layer output for Jumping Knowledge

        # ---- Layer 1 ----
        h = self.conv1(x, edge_index)
        h = self.bn1(h)                      # normalise features (keep node embeddings stable)
        h = F.relu(h)                        # non-linear activation
        h = F.dropout(h, p=self.dropout, training=self.training)  # regularise
        x = h                                # update x for next layer
        layer_outs.append(x)                 # store output of layer 1

        # ---- Layer 2 ----
        h = self.conv2(x, edge_index)
        h = self.bn2(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        x = x + h  # residual
        layer_outs.append(x)

        # Residual connection (if same dimension)
        # x = x + h # old: residual 
        x = self.jk(layer_outs) # JK: combine multi-layer features

        # ---- Pooling ----
        # x = global_mean_pool(x, batch) # old: mean pooling
        x = self.att_pool(x, batch) # new: attention-based pooling


        # ---- Final prediction ----
        x = self.linear(x)
        return x


# Edge-aware GCN model
class EdgeAwareGCNPlus(nn.Module):
    """
    Edge-aware 2-layer GCN with:
      - NNConv (edge-conditioned) message passing
      - BN + ReLU + Dropout + Residual
      - Jumping Knowledge ('cat') across layers
      - Attention-based global pooling
    Assumes data carries: x, edge_index, edge_attr, batch
    """

    def __init__(self,
                 num_node_features: int,
                 num_edge_features: int,
                 hidden_channels: int = 64,
                 dropout: float = 0.1):
        super().__init__()

        self.num_layers = 2
        self.hidden = hidden_channels
        self.dropout = dropout

        # --- Edge networks (produce per-edge weight matrices) ---
        # conv1: in=num_node_features, out=hidden -> weights of size (in*out)
        self.edge_mlp1 = nn.Sequential(
            nn.Linear(num_edge_features, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, num_node_features * hidden_channels)
        )
        # conv2: in=hidden, out=hidden -> weights of size (hidden*hidden)
        self.edge_mlp2 = nn.Sequential(
            nn.Linear(num_edge_features, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels * hidden_channels)
        )

        # --- Convolutions ---
        self.conv1 = NNConv(num_node_features, hidden_channels, self.edge_mlp1, aggr='mean')
        self.conv2 = NNConv(hidden_channels, hidden_channels, self.edge_mlp2, aggr='mean')

        # --- Normalisation ---
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)

        # --- Jumping Knowledge over layer outputs (we'll 'cat' them) ---
        self.jk = JumpingKnowledge(mode='cat', channels=hidden_channels, num_layers=self.num_layers)
        jk_dim = hidden_channels * self.num_layers  # 64 * 2 = 128 by default

        # --- Attention pooling (learned gate over nodes) ---
        self.att_gate = nn.Sequential(
            nn.Linear(jk_dim, jk_dim // 2),
            nn.ReLU(),
            nn.Linear(jk_dim // 2, 1)
        )
        self.att_pool = AttentionalAggregation(gate_nn=self.att_gate)

        # --- Prediction head ---
        self.head = nn.Sequential(
            nn.Linear(jk_dim, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1)
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        layer_outs = []

        # ----- Layer 1 -----
        h = self.conv1(x, edge_index, edge_attr)
        h = self.bn1(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        layer_outs.append(h)  # store 1-hop features

        # ----- Layer 2 + residual -----
        h2 = self.conv2(h, edge_index, edge_attr)
        h2 = self.bn2(h2)
        h2 = F.relu(h2)
        h2 = F.dropout(h2, p=self.dropout, training=self.training)
        h = h + h2                    # residual keeps earlier signal
        layer_outs.append(h)          # store 2-hop (post-residual) features

        # ----- Jumping Knowledge combine -----
        x = self.jk(layer_outs)       # [N, hidden * num_layers] when mode='cat'

        # ----- Attention pooling to graph embedding -----
        x = self.att_pool(x, batch)   # [B, hidden * num_layers]

        # ----- Head -----
        out = self.head(x)            # [B, 1]
        return out
