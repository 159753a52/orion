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

### 3.1 完整运行命令
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

### 3.2 预期输出
```
==================================================
GPT Orion Scheduler Launch
==================================================
Number of clients: 1
Loading scheduler from: /lihongliang/fangzl/orion-work/orion_gpt/src/scheduler/scheduler_eval.so
[GPT Client 0] Starting, batchsize=4, num_iters=10
...
Loading interception library from: /lihongliang/fangzl/orion-work/orion_gpt/src/cuda_capture/libinttemp.so
KERNEL_INFO_FILE IS /lihongliang/fangzl/orion-work/orion_gpt/gpt-example/profiling_results/gpt_kernel_info.csv
----------- SIZE: 2034
...
[GPT Client 0] Iteration 0 done, took 243.55 ms
[GPT Client 0] Received stop signal
Total loop took 0.243000 sec
...
==================================================
All threads completed!
==================================================
```

## 4. 配置文件说明

### gpt_config.json
```json
{
    "clients": [{
        "arch": "gpt",
        "kernel_file": "/lihongliang/fangzl/orion-work/orion_gpt/gpt-example/profiling_results/gpt_kernel_info.csv",
        "num_kernels": 2034,
        "num_iters": 10,
        "args": {
            "model_name": "gpt",
            "batchsize": 4,
            "sequence_len": 512
        }
    }]
}
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

## 6. 已知问题

1. **Kernel数量不匹配**: 预期2034个kernel，实际捕获1806个
   - 可能原因: profiling和实际运行时代码路径不同
   - 解决方案: 500ms超时后自动结束

2. **Core dump文件**: 调试过程中生成的core文件可以删除
   ```bash
   rm -f gpt-example/core.*
   ```

## 7. Git版本

当前可用版本commit: `a15abc8`

回滚命令:
```bash
git reset --hard a15abc8
```
