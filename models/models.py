import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d
from torch_geometric.nn import PNAConv, global_mean_pool
import joblib
from ops.loss import AdaptiveRelativeLoss



class Encoder(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 dropout=0.2):
        super().__init__()
        self.conv1 = Sequential(
            Linear(in_channels, out_channels),
            ReLU(),
            BatchNorm1d(out_channels),
        )

    def forward(self, x):
        x = self.conv1(x)
        return x


class PNA(nn.Module):
    def __init__(self, 
                 node_in_channels, 
                 var_in_channels,
                 edge_in_channels, 
                 hidden_channels, 
                 out_channels,
                 aggregators=['mean', 'min', 'max', 'std'],
                 scalers=['identity', 'amplification', 'attenuation'],
                 deg=None, 
                 num_layers=3, 
                 dropout=0.2):
        super().__init__()
        # Input projection for node features
        self.node_encoder = Linear(node_in_channels, hidden_channels)

        # Input projection for edge features
        self.edge_encoder = Linear(edge_in_channels, hidden_channels)

        # Input projection for variables
        self.vars_encoder = Linear(var_in_channels, hidden_channels)

        # Build PNA layers
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(
                PNAConv(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    aggregators=aggregators,
                    scalers=scalers,
                    deg=deg,
                    edge_dim=hidden_channels  # edge feature dimension after encoding
                )
            )
            self.batch_norms.append(BatchNorm1d(hidden_channels))
        # Output MLP for graph regression
        self.mlp = Sequential(
            Linear(hidden_channels+hidden_channels, hidden_channels),
            ReLU(),
            Linear(hidden_channels, out_channels)
        )
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr, vars, batch):
        # Encode node and edge features
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)
        vars = self.vars_encoder(vars)

        for conv, bn in zip(self.convs, self.batch_norms):
            x = conv(x, edge_index, edge_attr)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Global pooling (mean) for graph-level output
        x = global_mean_pool(x, batch['graph'].batch)

        # Final regression MLP
        out = self.mlp(torch.cat([x, vars], dim=1))
        return out



class Regression(pl.LightningModule):
    """
    PyTorch Lightning module to train/test the regressor.
    """

    def __init__(self, 
                num_classes=1, 
                vars_dim=11, 
                node_feat_dim=6, 
                edge_feat_dim=6,
                hidden_channels=64,
                num_layers=3,
                deg=None,
                lossfn='L1',
                scheduler=None,
                init_lr=1e-2,
                scaler_file=None):
        """
        Args:
            num_classes (int): Number of output dimensions
        """
        super().__init__()
        self.save_hyperparameters()
        self.model = PNA(node_in_channels=node_feat_dim, 
                         edge_in_channels=edge_feat_dim, 
                         var_in_channels=vars_dim, 
                         hidden_channels=hidden_channels,
                         out_channels=num_classes,
                         num_layers=num_layers, 
                         deg=deg,
                         )
        if lossfn == 'L1':
            self.loss = F.l1_loss
        elif lossfn == 'MSE':
            self.loss = F.mse_loss
        elif lossfn == 'Adaptive':
            self.loss = AdaptiveRelativeLoss()
        self.scheduler = scheduler
        self.lr = init_lr
        self.scaler = None
        if scaler_file:
            faceScaler, edgeScaler, varScaler, labelScaler = joblib.load(scaler_file)
            self.scaler = labelScaler

    def elastic_penalty(self, model, l1_ratio=0.5):
        l1_norm = sum(p.abs().sum() for p in model.parameters())
        l2_norm = sum(p.pow(2).sum() for p in model.parameters())
        return l1_ratio * l1_norm + (1 - l1_ratio) * l2_norm

    def forward(self, batch):
        inputs = batch["graph"].to(self.device)
        node_feat = inputs.x.reshape(-1, 18).float()
        edge_feat = inputs.edge_attr.reshape(-1, 18).float()
        vars = batch["vars"].to(self.device)
        labels = batch["label"].to(self.device)
        logits = self.model(node_feat, 
                            inputs.edge_index, 
                            edge_feat, 
                            vars, 
                            batch)
        logits = torch.squeeze(logits)
        if isinstance(self.loss, AdaptiveRelativeLoss):
            self.loss.set_y_var(labels) 
        loss = self.loss(logits, labels, reduction="mean") #+ 0.001 * self.elastic_penalty(self.model, 0.5)
        preds = logits
        if self.scaler:
            preds = self.scaler.inverse_transform(preds.cpu().detach().numpy().reshape(-1, 1))
            labels = self.scaler.inverse_transform(labels.cpu().numpy().reshape(-1, 1))
            preds = torch.from_numpy(preds)
            labels = torch.from_numpy(labels)
        acc = 1 - torch.mean(torch.abs(preds - labels) / labels)
        return {"loss": loss, "acc": acc}

    def training_step(self, batch, batch_idx):
        return self.forward(batch)

    def validation_step(self, batch, batch_idx):
        return self.forward(batch)

    def predict_step(self, batch, batch_idx):
        inputs = batch["graph"].to(self.device)
        node_feat = inputs.x.reshape(-1, 18).float()
        edge_feat = inputs.edge_attr.reshape(-1, 18).float()
        labels = batch["label"].to(self.device)
        vars = batch["vars"].to(self.device)
        logits = self.model(node_feat,
                            inputs.edge_index, 
                            edge_feat,
                            vars,
                            batch)
        logits = torch.squeeze(logits)
        preds = logits
        if self.scaler:
            preds = self.scaler.inverse_transform(logits.cpu().detach().numpy().reshape(-1, 1))
            labels = self.scaler.inverse_transform(labels.cpu().numpy().reshape(-1, 1))
            preds = torch.from_numpy(preds)
            labels = torch.from_numpy(labels)
        return {"labels": labels, "preds": preds}

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
        val_loss = 0.0
        val_loss = torch.stack([x['loss'] for x in outputs]).mean()
        val_acc = torch.stack([x['acc'] for x in outputs]).mean()
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_acc", val_acc, on_step=False, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), self.lr)
        scheduler = None
        monitor = None
        if self.scheduler == 'reducelronplateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
            monitor = 'val_loss'
            scheduler = {"scheduler": scheduler, 
                         "strict": False, 
                         "monitor": monitor}
        elif self.scheduler == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
        elif self.scheduler == 'linear':
            scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, 1, 0.001)
        elif self.scheduler == 'multistep':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[300, 700], gamma=0.1)
        if self.scheduler == None:
            return optimizer
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": monitor}

