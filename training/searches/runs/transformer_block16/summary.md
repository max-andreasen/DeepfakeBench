# Study: transformer_block16

- Total trials: 251
- Completed: 17
- Pruned: 233
- Failed: 1
- Wall-clock: 2.26 h

## Top 5

| Rank | Trial | Value | Params |
| --- | --- | --- | --- |
| 1 | 138 | 0.9544 | optimizer_type=adam, lr=0.0003240099597356709, weight_decay=9.62035565367408e-05, lr_scheduler=constant, num_layers=10, n_heads=8, dim_feedforward=1024, attn_dropout=0.2633531458161589, mlp_dropout=0.16771171324811882, mlp_hidden_dim=128 |
| 2 | 179 | 0.9541 | optimizer_type=adam, lr=0.00026777645380258905, weight_decay=7.079781318529977e-06, lr_scheduler=cosine_warmup, warmup_epochs=10, num_layers=6, n_heads=16, dim_feedforward=1024, attn_dropout=0.17368010398336428, mlp_dropout=0.44438898266377325, mlp_hidden_dim=1024 |
| 3 | 99 | 0.9522 | optimizer_type=adam, lr=4.6433791749394055e-05, weight_decay=0.0012599794317644558, lr_scheduler=cosine_warmup, warmup_epochs=7, num_layers=10, n_heads=4, dim_feedforward=2048, attn_dropout=0.05481700533402252, mlp_dropout=0.27237182638257634, mlp_hidden_dim=512 |
| 4 | 148 | 0.9521 | optimizer_type=adam, lr=0.0005570290677887123, weight_decay=7.073195054568212e-05, lr_scheduler=cosine, num_layers=9, n_heads=4, dim_feedforward=1024, attn_dropout=0.17931809462688963, mlp_dropout=0.39841602051756453, mlp_hidden_dim=128 |
| 5 | 96 | 0.9513 | optimizer_type=adam, lr=0.0003708215956268205, weight_decay=0.002954339465196583, lr_scheduler=cosine_warmup, warmup_epochs=7, num_layers=7, n_heads=4, dim_feedforward=1024, attn_dropout=0.34879672907200454, mlp_dropout=0.39164842211844514, mlp_hidden_dim=1024 |
