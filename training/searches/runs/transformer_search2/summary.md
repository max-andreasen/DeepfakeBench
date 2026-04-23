# Study: transformer_search2

- Total trials: 853
- Completed: 92
- Pruned: 759
- Failed: 2
- Wall-clock: 8.26 h

## Top 10

| Rank | Trial | Value | Params |
| --- | --- | --- | --- |
| 1 | 542 | 0.9541 | clip_layer=block_16, optimizer_type=adam, lr_scheduler=cosine, lr=0.0004618440618566113, weight_decay=0.00617363204815352, num_layers=10, n_heads=16, dim_feedforward=1024, attn_dropout=0.5554856888671009, mlp_dropout=0.20241314743859784, mlp_hidden_dim=1024 |
| 2 | 4 | 0.9534 | clip_layer=block_16, optimizer_type=adam, lr_scheduler=constant, lr=0.0003244160088734161, weight_decay=8.22607494622104e-06, num_layers=6, n_heads=8, dim_feedforward=512, attn_dropout=0.5355353990939866, mlp_dropout=0.3696711209578254, mlp_hidden_dim=128 |
| 3 | 376 | 0.9533 | clip_layer=block_16, optimizer_type=adam, lr_scheduler=cosine, lr=0.0004373112221639367, weight_decay=0.007510479825431071, num_layers=8, n_heads=8, dim_feedforward=1024, attn_dropout=0.3698745029608919, mlp_dropout=0.11179789547153351, mlp_hidden_dim=1024 |
| 4 | 677 | 0.9531 | clip_layer=block_16, optimizer_type=adam, lr_scheduler=cosine, lr=0.00036686177147205037, weight_decay=0.00700201220607947, num_layers=13, n_heads=16, dim_feedforward=1024, attn_dropout=0.5919763754198194, mlp_dropout=0.15879928091226836, mlp_hidden_dim=1024 |
| 5 | 838 | 0.9529 | clip_layer=block_16, optimizer_type=adam, lr_scheduler=cosine, lr=0.00022052158594474124, weight_decay=0.007543352132214297, num_layers=9, n_heads=8, dim_feedforward=1024, attn_dropout=0.11322570541959184, mlp_dropout=0.10068654628628879, mlp_hidden_dim=1024 |
| 6 | 280 | 0.9529 | clip_layer=block_12, optimizer_type=adam, lr_scheduler=cosine_warmup, warmup_epochs=9, lr=4.136358085113368e-05, weight_decay=7.425998408672037e-06, num_layers=7, n_heads=8, dim_feedforward=1024, attn_dropout=0.12415941097858406, mlp_dropout=0.449763776993488, mlp_hidden_dim=512 |
| 7 | 273 | 0.9528 | clip_layer=block_12, optimizer_type=adam, lr_scheduler=cosine_warmup, warmup_epochs=10, lr=3.3069747681083765e-05, weight_decay=2.4133007826267158e-06, num_layers=7, n_heads=8, dim_feedforward=1024, attn_dropout=0.03005332539815607, mlp_dropout=0.48438562379480093, mlp_hidden_dim=512 |
| 8 | 761 | 0.9526 | clip_layer=block_16, optimizer_type=adam, lr_scheduler=cosine, lr=0.00020057571065968882, weight_decay=0.008591885777391891, num_layers=10, n_heads=8, dim_feedforward=1024, attn_dropout=0.26674855749031123, mlp_dropout=0.10165738371254543, mlp_hidden_dim=1024 |
| 9 | 358 | 0.9525 | clip_layer=block_16, optimizer_type=adam, lr_scheduler=cosine, lr=6.527998412802627e-05, weight_decay=0.005490689444684897, num_layers=13, n_heads=2, dim_feedforward=256, attn_dropout=0.5255156075217686, mlp_dropout=0.42030705144256675, mlp_hidden_dim=512 |
| 10 | 357 | 0.9524 | clip_layer=block_16, optimizer_type=adam, lr_scheduler=cosine, lr=0.00016701770030388434, weight_decay=0.003016192925936962, num_layers=11, n_heads=2, dim_feedforward=256, attn_dropout=0.47316851230473755, mlp_dropout=0.43588387679697393, mlp_hidden_dim=512 |
