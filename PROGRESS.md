# GPT Orion调度器集成进展

## 环境信息
- CUDA版本: 12.4.1
- Python版本: 3.8.20
- 环境脚本: `/lihongliang/fangzl/envs/use-cuda-12.4.sh`

## 1. Profiling GPT模型

### 1.1 创建虚拟环境并安装依赖
```bash
cd /lihongliang/fangzl/orion-work/orion_gpt/gpt-example
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### 1.2 运行NSYS Profiling
```bash
source /lihongliang/fangzl/envs/use-cuda-12.4.sh
cd /lihongliang/fangzl/orion-work/orion_gpt/gpt-example

nsys profile \
    --trace=cuda \
    --output=profiling_results/output_nsys \
    --force-overwrite=true \
    ./venv/bin/python profile_gpt.py --mode nsys --batchsize 4 --seq_len 512 --warmup 2 --iters 1
```

### 1.3 导出GPU Trace为CSV
```bash
nsys stats \
    --report gputrace \
    --format csv \
    --output profiling_results/output_nsys \
    profiling_results/output_nsys.nsys-rep
```

### 1.4 处理NSYS输出生成kernel信息文件
```bash
cd /lihongliang/fangzl/orion-work/orion_gpt/gpt-example
./venv/bin/python process_nsys_gpt.py
```

生成的kernel信息文件: `profiling_results/gpt_kernel_info.csv`
- 总kernel数: 2034
- GEMM: 997, Elementwise: 614, Memcpy: 192, Softmax: 192

## 2. 编译Orion调度器和拦截库

### 2.1 编译cuda_capture拦截库
```bash
source /lihongliang/fangzl/envs/use-cuda-12.4.sh
cd /lihongliang/fangzl/orion-work/orion_gpt/src/cuda_capture
make clean
make -j1 libinttemp.so
```

### 2.2 编译scheduler调度器
```bash
source /lihongliang/fangzl/envs/use-cuda-12.4.sh
cd /lihongliang/fangzl/orion-work/orion_gpt/src/scheduler
make clean
make -j1 scheduler_eval.so
```

注意: 使用`-j1`单线程编译避免32GB内存机器OOM

## 3. 运行GPT拦截调度样例

### 3.1 完整运行命令（当前可成功运行）
```bash
source /lihongliang/fangzl/envs/use-cuda-12.4.sh
cd /lihongliang/fangzl/orion-work/orion_gpt/gpt-example

# 设置库路径
export LD_LIBRARY_PATH="/lihongliang/fangzl/orion-work/orion_gpt/gpt-example/venv/lib/python3.8/site-packages/nvidia/cudnn/lib:/lihongliang/fangzl/cuda-12.4.1/lib64:$LD_LIBRARY_PATH"

# 设置Orion拦截库路径
export ORION_LIB_PATH="/lihongliang/fangzl/orion-work/orion_gpt/src/cuda_capture/libinttemp.so"

# 运行GPT调度样例
LD_PRELOAD='/lihongliang/fangzl/orion-work/orion_gpt/src/cuda_capture/libinttemp.so' \
    ./venv/bin/python launch_gpt.py --config gpt_config.json
```

### 3.2 一键运行脚本
```bash
source /lihongliang/fangzl/envs/use-cuda-12.4.sh && \
cd /lihongliang/fangzl/orion-work/orion_gpt/gpt-example && \
export LD_LIBRARY_PATH="/lihongliang/fangzl/orion-work/orion_gpt/gpt-example/venv/lib/python3.8/site-packages/nvidia/cudnn/lib:/lihongliang/fangzl/cuda-12.4.1/lib64:$LD_LIBRARY_PATH" && \
export ORION_LIB_PATH="/lihongliang/fangzl/orion-work/orion_gpt/src/cuda_capture/libinttemp.so" && \
LD_PRELOAD='/lihongliang/fangzl/orion-work/orion_gpt/src/cuda_capture/libinttemp.so' ./venv/bin/python launch_gpt.py --config gpt_config.json
```

### 3.3 预期输出
```
==================================================
GPT Orion Scheduler Launch
==================================================
Number of clients: 1
Loading scheduler from: /lihongliang/fangzl/orion-work/orion_gpt/src/scheduler/scheduler_eval.so
[GPT Client 0] Starting, batchsize=4, num_iters=5
...
Loading interception library from: /lihongliang/fangzl/orion-work/orion_gpt/src/cuda_capture/libinttemp.so
KERNEL_INFO_FILE IS /lihongliang/fangzl/orion-work/orion_gpt/gpt-example/profiling_results/gpt_kernel_info_v4.csv
----------- SIZE: 1806
...
[GPT Client 0] Iteration 0 done, took 260.82 ms
[GPT Client 0] Passed final barrier
[GPT Client 0] Completed 1 iterations in 1.26 sec
...
==================================================
All threads completed!
==================================================
```

## 4. 配置文件说明

### gpt_config.json（当前配置）
```json
[
    {
        "arch": "gpt",
        "kernel_file": "/lihongliang/fangzl/orion-work/orion_gpt/gpt-example/profiling_results/gpt_kernel_info_v4.csv",
        "num_kernels": 1806,
        "num_iters": 5,
        "args": {
            "model_name": "gpt",
            "batchsize": 4,
            "rps": 0,
            "uniform": true,
            "dummy_data": true,
            "train": false
        }
    }
]
```

## 5. 代码修改说明

### 5.1 调度器修改 (scheduler_eval.cpp)
- 添加500ms空闲超时检测
- 超时后自动设置`request_status`和`stops`信号，让客户端线程退出
- 支持`ORION_LIB_PATH`环境变量指定拦截库路径

### 5.2 CUBLAS拦截修改 (intercept_cublas.cpp)
- 非客户端线程直接返回成功，避免dlsym失败

### 5.3 Makefile修改
- 更新CUDA路径为12.4.1
- 添加cudnn和cublas库链接

## 6. Kernel数量对齐

### 问题分析
- 原始NSYS profile得到2929个GPU事件（包括memcpy）
- 非memcpy kernel: 1843个
- 实际拦截到的kernel: 1806个

### 解决方案
1. 重新用NSYS profiling获取单次迭代的kernel trace
2. 过滤掉memcpy事件（Orion拦截库不计入memcpy）
3. 取前1806个kernel作为profile文件

### Profile文件说明
| 文件 | 说明 |
|------|------|
| gpt_kernel_info.csv | 原始2034 kernel (包含memcpy) |
| gpt_kernel_info_v3.csv | 1843 kernel (不含memcpy) |
| gpt_kernel_info_v4.csv | 1806 kernel (对齐实际拦截数量) |

### 当前配置
- kernel_file: gpt_kernel_info_v4.csv
- num_kernels: 1806

## 7. Core dump文件

调试过程中生成的core文件可以删除：
```bash
rm -f gpt-example/core.*
```

## 8. Git版本

初始版本commit: `a15abc8`

回滚命令:
```bash
git reset --hard a15abc8
```
