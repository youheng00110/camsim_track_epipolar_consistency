# OpenDWM preview 自动读取

本版本可以直接递归读取 OpenDWM 导出的 `stflow_manifest.jsonl`，不需要生成额外的 `frames.jsonl`，也不会复制图片。

默认预览根目录：

```text
/inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/output/eval/nuplanhard1000
```

## 1. 仅扫描并检查图片

```bash
cd /inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/sam3-eval/sam3-eval
bash scan_preview.sh
```

也可以临时覆盖目录：

```bash
bash scan_preview.sh config.yaml /path/to/another/preview_root
```

程序会递归搜索：

```text
**/stflow_manifest.jsonl
```

每个 `video × timestep × camera` 会被展开为一张 SAM 输入图。默认跳过 manifest 中标记为 `is_reference_frame=true` 的参考帧。

## 2. 小规模测试

```bash
CUDA_VISIBLE_DEVICES=0 python run_eval.py \
  --config config.yaml \
  --limit-frames 16
```

保留参考帧时增加：

```bash
--keep-reference-frames
```

## 3. 完整运行

单卡：

```bash
bash run.sh config.yaml 1
```

四卡：

```bash
bash run.sh config.yaml 4
```

第三个参数可以覆盖预览根目录：

```bash
bash run.sh config.yaml 4 /path/to/preview_root
```

## 4. 筛选方法

在 `config.yaml` 中使用 shell 通配符：

```yaml
preview:
  include_methods:
    - "*token*"
    - "*plucker*"
  exclude_methods:
    - "*debug*"
```

空列表表示不筛选。方法名取 manifest 相对 `preview_root` 的父目录路径；结果在 `summary.json` 中按方法分别汇总。

## 当前限制

当前自动读取阶段把 `boxes` 设为空，只验证图片发现、SAM 文本检测、mask 输出和可视化。此时 recall、precision、F1 和 IoU 还不能作为正式指标。接入 NuPlan 条件 3D box 后再启用几何匹配指标。
