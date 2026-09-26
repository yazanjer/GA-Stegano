# Models

`srnet_pretrained_lsbm05.pt` is the SRNet state dict pre-trained for 30 epochs on the
1,400 training covers against synthetic +-1 embedding at 0.5 bpp. It is the starting
point of every per-method fine-tuning in `revision/srnet_v2.py`. Copy it to
`<run>/srnet_pretrained.pt` to skip the pre-training step.

The file is stored in parts of 3 MB. Join the parts before use:

```bash
cat models/srnet_pretrained_lsbm05.pt.part* > models/srnet_pretrained_lsbm05.pt
sha256sum models/srnet_pretrained_lsbm05.pt   # 5a3f4c33c455912b93025c63183a3e9cd71b4f3a1bb24b43598cbf2bed0f82cc
cat results/revision2/stego.csv.gz.part* > results/revision2/stego.csv.gz
sha256sum results/revision2/stego.csv.gz      # 05e21727b9c43b3a83c6f6f989a9ef1593676659162ae5c862e493d1dc7c38e1
```
