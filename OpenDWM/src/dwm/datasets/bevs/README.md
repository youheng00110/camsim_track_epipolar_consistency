# OpenDWM BEV datasets

目标目录

```text
/inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/
camsim_lyh/OpenDWM/src/dwm/datasets/bevs
```

文件结构

```text
src/dwm/datasets/bevs/
├── __init__.py
├── common.py
├── nuscenes.py
├── waymo.py
├── argoverse.py
├── nuplan.py
└── test_compare_original.py
```

安装

在压缩包所在目录执行

```bash
OPEN_DWM=/inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/OpenDWM
unzip -o OpenDWM_bevs_patch.zip -d /tmp/OpenDWM_bevs_patch
mkdir -p "$OPEN_DWM/src/dwm/datasets/bevs"
cp -a /tmp/OpenDWM_bevs_patch/src/dwm/datasets/bevs/. \
  "$OPEN_DWM/src/dwm/datasets/bevs/"
```

测试使用现有配置

```bash
cd /inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/OpenDWM
export PYTHONPATH="$PWD/src:$PYTHONPATH"

python -m dwm.datasets.bevs.test_compare_original \
  --config configs/ctsd/unimlvg/camsim/nuplan/nuplancamtoken.json \
  --samples-per-entry 1 \
  --report /tmp/bevs_vs_original_report.json
```

测试脚本会检查配置中的全部条目。训练配置含三个 Argoverse 相机布局，所以默认实际比较 Waymo、nuScenes、三个 Argoverse 布局和 NuPlan。

测试判定

- 必须一致
  - dataset 长度
  - RGB 图像变换后的 `vae_images`
  - 相机内参、图像尺寸、相机外参、ego 位姿
  - FPS、文本、stub 字段
  - 3D box 投影的非零区域
- 允许变化并逐通道报告
  - box slot 顺序
  - 统一后的类别编号
  - 被过滤的未知类别
  - Waymo 的静态 BEV
  - 由新类别和稳定 slot 生成的动态 BEV
- 新版必须满足
  - 同一 slot 跨时间类别不变
  - 同一时刻不同视角中的 box 角点一致
  - 新版有效 box 均能在原版当前帧 box 集合中按角点匹配

Waymo 静态 BEV 的前三个通道固定为

```text
R crossing
G lane
B drivable
```

实例 ID 缺失时，测试立即停止并打印具体数据集和样本位置。代码没有候选 key 或自动回退。

指定测试索引

```bash
python -m dwm.datasets.bevs.test_compare_original \
  --indices 0,10 \
  --datasets waymo,nuscenes,argoverse,nuplan
```

退出码

```text
0 只有预期条件差异
2 存在非预期差异
其他 Python 异常 实例 ID、文件或数据读取失败
```
