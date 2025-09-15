# SAR-RNTR (Semi-Autoregressive Road Network Transformer) 使用指南

## 概述

SAR-RNTR 是基于论文《将图像翻译为道路网络：一个序列到序列的视角》中半自回归方法的实现。相比于原始的AR-RNTR（自回归），SAR-RNTR将道路网络分解为多个并行子序列，每个子序列内部保持自回归依赖，不同子序列间可以并行生成，从而显著提升推理速度（论文中提到6倍加速）和准确率。

## 核心特性

- **并行子序列生成**：将完整的RoadNet序列拆分为多个以关键点为起始的子序列
- **序列内因果性**：每个子序列内部仍保持自回归的因果依赖关系
- **共享BEV记忆**：所有子序列共享相同的BEV特征表示
- **关键点检测**：自动检测交叉口、端点等关键位置作为子序列起点
- **兼容现有框架**：完全兼容现有的AR-RNTR数据格式和评测指标

## 文件结构

```
rntr/
├── sar_rntr.py                    # SAR-RNTR主模型
├── sar_rntr_head.py               # 半自回归解码头
├── parallel_seq_transformer.py    # 并行序列Transformer
└── transforms/
    └── parallel_seq_transform.py  # 数据预处理管道

configs/sar_rntr_roadseq/
└── sar_rntr_r50_704x256_24e_2key.py  # 配置文件

test_sar_rntr.py                   # 测试脚本
```

## 快速开始

### 1. 环境准备

确保已按照 `get_started.md` 完成基础环境配置：

```bash
# 基础依赖（与AR-RNTR相同）
conda create -n roadnet python=3.8
conda activate roadnet
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
pip install mmengine==0.10.1 mmcv==2.1.0 mmdet==3.3.0 mmsegmentation==1.2.2
pip install mmdet3d==1.4.0
```

### 2. 数据准备

使用与AR-RNTR相同的数据准备流程：

```bash
# 下载nuScenes数据集
# 运行数据预处理
python tools/create_data_pon_centerline.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes
```

### 3. 训练

```bash
# 单GPU训练
python tools/train.py configs/sar_rntr_roadseq/sar_rntr_r50_704x256_24e_2key.py

# 多GPU训练
bash tools/dist_train.sh configs/sar_rntr_roadseq/sar_rntr_r50_704x256_24e_2key.py 8
```

### 4. 测试

```bash
# 测试
python tools/test.py configs/sar_rntr_roadseq/sar_rntr_r50_704x256_24e_2key.py work_dirs/sar_rntr_r50_704x256_24e_2key/latest.pth --eval bbox

# 可视化
python tools/test.py configs/sar_rntr_roadseq/sar_rntr_r50_704x256_24e_2key.py work_dirs/sar_rntr_r50_704x256_24e_2key/latest.pth --show --show-dir ./vis_results
```

## 核心组件详解

### 1. SAR_RNTR 模型

主模型类，继承自 `MVXTwoStageDetector`：

```python
model = dict(
    type='SAR_RNTR',
    img_backbone=dict(type='ResNet', depth=50, ...),
    img_neck=dict(type='CPFPN', ...),
    pts_bbox_head=dict(type='SARRNTRHead', ...)
)
```

**关键方法**：
- `forward_pts_train()`: 训练时的前向传播
- `parse_predictions()`: 解析Transformer输出为各组件预测
- `simple_test_pts()`: 推理时的前向传播

### 2. SARRNTRHead 解码头

半自回归序列解码头：

```python
pts_bbox_head=dict(
    type='SARRNTRHead',
    max_parallel_seqs=16,      # 最大并行子序列数
    max_seq_len=128,           # 每个子序列最大长度
    embed_dims=256,            # 嵌入维度
    transformer=dict(
        type='LssParallelSeqLineTransformer',
        decoder=dict(
            type='ParallelSeqTransformerDecoderLayer',
            ...
        )
    )
)
```

**关键功能**：
- 关键点检测：`extract_keypoints()`
- 并行序列准备：`prepare_parallel_sequences()`
- 自回归生成：推理时逐步生成每个子序列

### 3. 并行序列Transformer

专门设计的Transformer架构：

```python
transformer=dict(
    type='LssParallelSeqLineTransformer',
    decoder=dict(
        type='ParallelSeqTransformerDecoderLayer',
        attn_cfgs=[
            dict(type='MultiheadAttention', ...),  # 序列内自注意力（带因果mask）
            dict(type='CustomMSDeformableAttention', ...)  # 与BEV的交叉注意力
        ],
        operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')
    )
)
```

**核心特性**：
- 序列内因果mask：`generate_square_subsequent_mask()`
- BEV记忆共享：所有子序列共享相同的BEV特征
- 并行处理：将 `[B, M, L, D]` reshape为 `[B*M, L, D]` 进行批处理

### 4. 数据预处理管道

两个关键Transform：

```python
train_pipeline = [
    # ... 其他transforms
    dict(type='RoadNetSequenceParser',      # 将中心线转换为RoadNet序列
         pc_range=point_cloud_range,
         discretization_resolution=0.5),
    dict(type='ParallelSeqTransform',       # 将序列拆分为并行子序列
         max_parallel_seqs=16,
         max_seq_len=128,
         keypoint_strategy='intersection',   # 关键点策略
         min_seq_len=3,
         noise_ratio=0.1),
    # ...
]
```

## 配置参数详解

### 模型参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `max_parallel_seqs` | 16 | 最大并行子序列数量 |
| `max_seq_len` | 128 | 每个子序列的最大长度 |
| `embed_dims` | 256 | 嵌入维度 |
| `num_center_classes` | 576 | 词汇表大小 |

### 数据预处理参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `keypoint_strategy` | 'intersection' | 关键点选择策略：intersection/endpoint/grid/random |
| `min_seq_len` | 3 | 有效子序列的最小长度 |
| `noise_ratio` | 0.1 | 噪声子序列的比例 |
| `discretization_resolution` | 0.5 | 坐标离散化分辨率 |

### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `lr` | 2e-4 | 学习率 |
| `total_epochs` | 24 | 训练轮数 |
| `samples_per_gpu` | 1 | 每GPU批大小 |

## 性能对比

根据论文结果，SAR-RNTR相比AR-RNTR的优势：

| 指标 | AR-RNTR | SAR-RNTR | 提升 |
|------|---------|----------|------|
| Landmark F-score | 54.1 | 56.0 | +1.9 |
| Reachability F-score | 61.3 | 65.7 | +4.4 |
| 推理速度 (FPS) | 0.1 | 0.6 | **6.0×** |

## 故障排除

### 常见问题

1. **显存不足**
   - 减少 `max_parallel_seqs` 或 `max_seq_len`
   - 使用梯度累积：`optimizer_config=dict(grad_clip=dict(max_norm=35, norm_type=2))`

2. **关键点检测效果差**
   - 调整 `keypoint_strategy`，尝试 'grid' 或 'endpoint'
   - 增加 `min_seq_len` 过滤短序列

3. **训练不稳定**
   - 检查学习率设置
   - 确保数据预处理正确生成了 `parallel_seqs_*` 字段

### 调试技巧

1. **可视化并行序列**：
```python
# 在训练脚本中添加
print("Parallel sequences shape:", parallel_seqs_in.shape)
print("Valid sequences:", parallel_seqs_mask.sum(dim=-1))
```

2. **检查关键点质量**：
```python
# 在SARRNTRHead.extract_keypoints()中添加
print("Detected keypoints:", len(keypoints_list[0]))
```

## 扩展开发

### 添加新的关键点策略

在 `ParallelSeqTransform.extract_keypoints_from_sequence()` 中添加：

```python
elif self.keypoint_strategy == 'custom':
    # 实现自定义关键点检测逻辑
    keypoints = custom_keypoint_detection(sequence)
```

### 优化并行序列长度

可以根据数据集统计动态调整：

```python
# 分析序列长度分布
seq_lengths = [len(seq) for seq in all_sequences]
optimal_max_len = np.percentile(seq_lengths, 95)
```

## 引用

如果使用此实现，请引用原论文：

```bibtex
@article{roadnet2024,
  title={将图像翻译为道路网络：一个序列到序列的视角},
  author={...},
  journal={...},
  year={2024}
}
```
