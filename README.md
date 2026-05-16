# project-constellation-stability
运动稳定性大作业：摄动力和外力破坏下的星座构型稳定性分析

## 项目初始结构

```text
project_constellation_stability/
  __init__.py
  plotting.py
  stability_analysis.py
tests/
  test_plotting.py
  test_stability_analysis.py
```

- `stability_analysis.py`：稳定性分析模块，提供扰动指标与稳定性判定。
- `plotting.py`：绘图模块，提供轨道半径序列的绘图数据生成功能。
- `tests/`：测试模块，覆盖稳定性分析与绘图数据生成的核心行为。
