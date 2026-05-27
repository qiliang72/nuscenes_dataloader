# nuScenes Dataset/DataLoader 基础框架

这个项目用于学习 PyTorch `Dataset` 和 `DataLoader` 的基本使用方式。当前版本以 nuScenes 数据集为命名示例，只搭建数据读取框架，不解析真实 nuScenes 文件内容。

## 环境准备

本项目使用 `uv` 管理 Python 版本、虚拟环境和依赖。Windows 可参考官方文档安装：

- https://docs.astral.sh/uv/getting-started/installation/

建议使用 Python 3.12，项目已在 `pyproject.toml` 中固定为：

```toml
requires-python = ">=3.12,<3.13"
```

初始化和安装依赖：

```powershell
uv python install 3.12
uv sync --dev
```

项目默认使用 CPU 版 PyTorch。当前依赖版本为 `torch==2.4.0` 和 `numpy==1.26.4`，后续如果需要 GPU/CUDA，可以再根据机器环境调整 `pyproject.toml` 中的 PyTorch index。

## 项目结构

```text
src/
  datasets/
    nuscenes_dataset.py
  train_loop.py
tests/
  test_nuscenes_dataset.py
```

`src` 下面已经去掉额外的 `dist_train` 目录层级，便于先专注学习 Dataset 和 DataLoader 的核心用法。

## 运行 demo

```powershell
uv run python src/train_loop.py
```

你会看到 `DataLoader` 按 batch 循环输出 mock 样本。这个过程展示了：

1. `Dataset.__len__` 告诉框架有多少条样本。
2. `Dataset.__getitem__` 定义如何按索引取出一条样本。
3. `DataLoader` 负责批量读取、打乱顺序、组织 batch。

## 运行测试

```powershell
uv run pytest
```

## 当前数据格式

`NuScenesDataset` 当前返回的每条样本是一个字典：

```python
{
    "sample_token": "train-sample-000000",
    "scene_name": "scene-0000",
    "timestamp": 0,
    "split": "train",
}
```

后续版本可以在这个结构上逐步加入真实 nuScenes 的图像、点云、标注、标定参数和时序信息读取逻辑。
