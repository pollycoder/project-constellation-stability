#!/bin/bash
#SBATCH --job-name=big_project   # 任务名
#SBATCH --partition=compute      # 分区名
#SBATCH --nodes=1                # 单节点
#SBATCH --ntasks=1               # 1个主进程
#SBATCH --cpus-per-task=8       # 给32个核做数据采样 
#SBATCH --mem=32G                # 给64G内存 (原先可能默认很小)
#SBATCH --time=48:00:00          # 跑48小时
#SBATCH --output=logs/BIG_PROJECT_%j.log


# 2. 激活环境
source ~/.bashrc

# 3. 运行
python -u walker_uub_sim.py