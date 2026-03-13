# Local Briefing Studio

一个全新搭建的本地私有财经周报写作台，面向单人使用，覆盖：

- 历史语料导入与风格画像
- 本周新闻候选池与证据包
- 本地训练/评测流水线管理
- 周报草稿生成、逐段反馈、偏好样本沉淀
- Markdown / HTML 导出

## 当前定位

这是第一阶段的强原型：

- 后台、数据库、接口、训练资产闭环已经搭好
- 提供真实可运行的抓源、聚类、证据包、草稿生成和反馈回灌
- 对 `MLX / MLX-LM` 保留清晰接入点
- 若本机尚未安装模型运行时，会自动退回到**规则驱动的本地占位生成器**

## 快速启动

```bash
cd /Users/yuwan/Documents/New\ project/local-briefing-studio
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
uvicorn app.main:app --reload
```

打开：

- 后台首页：`http://127.0.0.1:8000/`
- 设置中心：`http://127.0.0.1:8000/settings`
- 本周候选主题：`http://127.0.0.1:8000/topics-board`

## 导入加密 PDF 语料

如果你的历史周报是带密码的 PDF，可以直接批量导入，无需先手工另存：

```bash
cd /Users/yuwan/Documents/New\ project/local-briefing-studio
source .venv/bin/activate
python scripts/import_local_corpus.py /Users/yuwan/Desktop/订阅 --password 654321 --train-sft
```

也支持通过环境变量注入默认密码：

```bash
export BRIEFING_PDF_PASSWORD=654321
python scripts/import_local_corpus.py /Users/yuwan/Desktop/订阅 --train-sft
```

## 启动本机真实训练

历史语料导入完后，可以直接在本机启动 `MLX-LM LoRA` 微调：

```bash
cd /Users/yuwan/Documents/New\ project/local-briefing-studio
source .venv/bin/activate
python scripts/train_writer_mlx.py --model mlx-community/Qwen2.5-7B-Instruct-4bit
```

如果你希望更稳、更凉快一些，可以改用 3B：

```bash
python scripts/train_writer_mlx.py --model mlx-community/Qwen2.5-3B-Instruct-4bit
```

训练数据会被自动切成：

- `train.jsonl`
- `valid.jsonl`
- `test.jsonl`

对应目录位于：

- `data/training/mlx_sft/<artifact_id>/`
- `data/models/writer/<artifact_id>/`

日志会写到：

- `data/training/mlx_sft/<artifact_id>.log`

## 主要接口

- `POST /corpus/import`
- `POST /sources/sync`
- `GET /topics?week=YYYY-WW`
- `POST /issues`
- `POST /issues/{id}/generate`
- `POST /issues/{id}/feedback`
- `POST /training/sft`
- `POST /training/preferences`
- `POST /eval/run`
- `POST /models/promote`
- `POST /models/rollback`

## MLX 接入说明

默认实现已经为 Apple Silicon 本地微调预留适配器位。若你准备启用真实模型链路，可优先参考：

- [MLX](https://github.com/ml-explore/mlx)
- [MLX-LM](https://github.com/ml-explore/mlx-lm)
- [mlx-retrieval](https://github.com/jina-ai/mlx-retrieval)

当前代码中的 `app/services/training.py` 与 `app/services/runtime.py` 已经定义好：

- 训练数据集输出目录
- 适配器产物登记
- 生产模型切换与回滚
- 缺少运行时时的安全降级行为

## 目录

```text
local-briefing-studio/
├── app/
│   ├── main.py
│   ├── config.py
│   ├── db.py
│   ├── schemas.py
│   ├── services/
│   ├── static/
│   └── templates/
├── data/
└── tests/
```

## 测试

```bash
cd /Users/yuwan/Documents/New\ project/local-briefing-studio
python3 -m unittest discover tests
```
