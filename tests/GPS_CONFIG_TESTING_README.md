# GPS配置测试数据结构设计

本文档介绍了为`test_fp8.py`中的`test_image_configs()`函数设计的数据结构，用于存储和分析不同M、N配置下的测试结果。

## 数据结构设计

### 1. GemmTestResult
存储单次GEMM测试的完整结果：

```python
@dataclass
class GemmTestResult:
    # 测试配置
    m: int                    # 矩阵M维度
    n: int                    # 矩阵N维度  
    k: int                    # 矩阵K维度
    kernel_type: str          # 内核类型 ('1D1D' or '1D2D')
    layout: str               # 矩阵布局 ('NT', 'TN', 'NN', 'TT')
    out_dtype: str            # 输出数据类型 ('FP32' or 'BF16')
    accumulate: bool          # 是否累加
    
    # 性能指标
    launch_time_us: float     # 启动时间 (微秒)
    execution_time_us: float  # 执行时间 (微秒)
    tflops: float            # TFLOPS性能
    bandwidth_gb_s: float    # 带宽 (GB/s)
    
    # 正确性验证
    diff: float              # 与参考结果的差异
```

### 2. GPSConfigTestResult
存储特定GPS配置下的所有测试结果：

```python
@dataclass
class GPSConfigTestResult:
    gps_m: int                           # GPS配置的M值
    gps_n: int                           # GPS配置的N值
    test_results: List[GemmTestResult]   # 该配置下的所有测试结果
```

### 3. GPSTestResultManager
管理所有GPS配置的测试结果：

```python
class GPSTestResultManager:
    def add_config_result(gps_m, gps_n, test_results)  # 添加配置结果
    def get_config_result(gps_m, gps_n)                # 获取配置结果
    def save_to_file(filename)                         # 保存到JSON文件
    def load_from_file(filename)                       # 从JSON文件加载
    def print_summary_table()                          # 打印性能摘要表格
```

## 使用方法

### 1. 运行GPS配置测试

```bash
# 运行所有GPS配置测试
python test_fp8.py --gps-configs

# 指定输出文件
python test_fp8.py --gps-configs --output my_results.json
```

### 2. 分析测试结果

```bash
# 分析默认结果文件
python test_fp8.py --analyze

# 分析指定结果文件
python test_fp8.py --analyze --file my_results.json
```

### 3. 比较两个配置

```bash
# 比较GPS配置 (M=64,N=16) 和 (M=64,N=32)
python test_fp8.py --compare 64 16 64 32

# 使用指定结果文件比较
python test_fp8.py --compare 64 16 64 32 my_results.json
```

### 4. 编程方式使用

```python
from test_fp8 import test_image_configs, analyze_gps_results, compare_gps_configs

# 运行测试并获取管理器
manager = test_image_configs(save_results=True, results_file="results.json")

# 获取特定配置的结果
config_result = manager.get_config_result(64, 16)
summary = config_result.get_summary()

# 分析结果
analyze_gps_results("results.json")

# 比较配置
compare_gps_configs((64, 16), (64, 32), "results.json")
```

## 输出格式

### 1. 控制台输出
测试过程中会显示：
- 当前测试进度
- 每个配置的简要统计
- 性能摘要表格
- 最佳配置推荐

### 2. JSON文件格式
```json
{
  "configs": [
    {
      "gps_config": {"m": 64, "n": 16},
      "test_results": [
        {
          "config": {
            "m": 128, "n": 2112, "k": 7168,
            "kernel_type": "1D2D",
            "layout": "NT",
            "out_dtype": "FP32",
            "accumulate": false
          },
          "performance": {
            "launch_time_us": 45.2,
            "execution_time_us": 123.4,
            "tflops": 856.7,
            "bandwidth_gb_s": 1234.5
          },
          "correctness": {
            "diff": 0.000123
          }
        }
      ],
      "summary": {
        "total_tests": 24,
        "avg_tflops": 845.2,
        "avg_bandwidth_gb_s": 1205.3,
        "avg_launch_time_us": 42.1,
        "avg_execution_time_us": 118.7,
        "max_diff": 0.000456
      }
    }
  ]
}
```

## 性能分析功能

### 1. 摘要表格
显示所有GPS配置的关键性能指标：
- 平均TFLOPS
- 平均带宽
- 平均启动时间
- 平均执行时间
- 最大误差

### 2. 最佳配置识别
自动识别：
- TFLOPS最高的配置
- 带宽最高的配置

### 3. 配置比较
详细比较两个配置的：
- 性能差异百分比
- 各项指标的改进情况

## 环境变量控制

- `DG_GPS_CONFIG_N`: 设置GPS配置的N值
- `DG_GPS_CONFIG_M`: 设置GPS配置的M值  
- `DG_PRINT_CONFIGS`: 启用配置打印 (设置为1)

## 示例脚本

运行 `example_gps_testing.py` 查看完整的使用示例：

```bash
python tests/example_gps_testing.py
```

这个脚本演示了：
1. 如何运行GPS配置测试
2. 如何分析结果
3. 如何比较不同配置

## 注意事项

1. **测试时间**: 完整的GPS配置测试需要较长时间（90个配置 × 每个配置多个测试用例）
2. **存储空间**: JSON结果文件可能较大，包含详细的性能数据
3. **错误处理**: 如果某个配置测试失败，会跳过该配置并继续测试其他配置
4. **环境变量**: 测试过程中会临时修改环境变量，测试完成后会自动恢复