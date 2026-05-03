# Study: peft_gend_pilot12

- Total trials: 12
- Completed: 7
- Pruned: 5
- Failed: 0
- Wall-clock: 43.27 h
- Best trial: 11
- Best val_auc: 0.8339

## Top 10

| Rank | Trial | Value | Params |
| --- | --- | --- | --- |
| 1 | 11 | 0.8339 | clip.feature_layer=pre_proj, clip.l2_normalize_features=False, loss.ua_enabled=False, optimizer.lr=3.526898898291157e-05, optimizer.weight_decay=0.0001, temporal.num_layers=2, temporal.dim_feedforward=1024, temporal.attn_dropout=0.0, temporal.mlp_dropout=0.10262369664587076, temporal.mlp_hidden_dim=256 |
| 2 | 5 | 0.8151 | clip.feature_layer=pre_proj, clip.l2_normalize_features=False, loss.ua_enabled=False, optimizer.lr=3.289828520266678e-05, optimizer.weight_decay=0.0001, temporal.num_layers=2, temporal.dim_feedforward=2048, temporal.attn_dropout=0.0, temporal.mlp_dropout=0.14534341680430196, temporal.mlp_hidden_dim=256 |
| 3 | 7 | 0.8099 | clip.feature_layer=pre_proj, clip.l2_normalize_features=False, loss.ua_enabled=False, optimizer.lr=3.981609797764976e-05, optimizer.weight_decay=0.0001, temporal.num_layers=2, temporal.dim_feedforward=1024, temporal.attn_dropout=0.0, temporal.mlp_dropout=0.11157283262237103, temporal.mlp_hidden_dim=256 |
| 4 | 3 | 0.7919 | clip.feature_layer=pre_proj, clip.l2_normalize_features=False, loss.ua_enabled=False, optimizer.lr=2e-05, optimizer.weight_decay=0.0001, temporal.num_layers=1, temporal.dim_feedforward=1024, temporal.attn_dropout=0.0, temporal.mlp_dropout=0.25, temporal.mlp_hidden_dim=256 |
| 5 | 1 | 0.7767 | clip.feature_layer=pre_proj, clip.l2_normalize_features=True, loss.ua_enabled=False, optimizer.lr=2e-05, optimizer.weight_decay=0.0001, temporal.num_layers=1, temporal.dim_feedforward=1024, temporal.attn_dropout=0.0, temporal.mlp_dropout=0.25, temporal.mlp_hidden_dim=256 |
| 6 | 0 | 0.7734 | clip.feature_layer=pre_proj, clip.l2_normalize_features=True, loss.ua_enabled=True, optimizer.lr=2e-05, optimizer.weight_decay=0.0001, temporal.num_layers=1, temporal.dim_feedforward=1024, temporal.attn_dropout=0.0, temporal.mlp_dropout=0.25, temporal.mlp_hidden_dim=256 |
| 7 | 2 | 0.7729 | clip.feature_layer=pre_proj, clip.l2_normalize_features=False, loss.ua_enabled=True, optimizer.lr=2e-05, optimizer.weight_decay=0.0001, temporal.num_layers=1, temporal.dim_feedforward=1024, temporal.attn_dropout=0.0, temporal.mlp_dropout=0.25, temporal.mlp_hidden_dim=256 |
