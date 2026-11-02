# Runtime model assets

Copy only the approved server assets into this directory during deployment:

- `decoder_scmr_v5_10_100.pth`, SHA256 `338f2c0b8db90851f733ed31caa2c99914605fed3f7215ec2f69f19eba59583b`;
- `b6_model.pt`, SHA256 `15981fbe93d2c70e14755ea756a02856c9383481e7d92d239619268a628a29b5`.

The binaries are intentionally ignored by Git. Run `scripts/verify_runtime_contract.py` after copying. Never place the historical `[50,150]` decoder here.

