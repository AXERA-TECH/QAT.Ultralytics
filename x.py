#!/usr/bin/env python3
"""
GPU显存占据脚本
功能：在指定的GPU上占据指定比例的显存，按Ctrl+C后自动释放
"""

import torch
import time
import signal
import sys
import argparse
from typing import List, Dict

class GPUMemoryOccupier:
    def __init__(self):
        self.occupied_tensors = {}
        self.is_running = True
        self.setup_signal_handlers()
    
    def setup_signal_handlers(self):
        """设置信号处理器，用于优雅退出"""
        def signal_handler(sig, frame):
            print(f"\n接收到信号 {sig}，正在释放显存并退出...")
            self.cleanup()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
        signal.signal(signal.SIGTERM, signal_handler) # kill命令
    
    def get_gpu_info(self) -> Dict:
        """获取GPU信息"""
        gpu_info = {}
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                total_memory = props.total_memory
                free_memory = torch.cuda.memory_reserved(i)  # 已预留内存
                gpu_info[i] = {
                    'name': props.name,
                    'total_memory_gb': total_memory / (1024**3),
                    'free_memory_gb': (total_memory - free_memory) / (1024**3),
                    'total_memory': total_memory
                }
        return gpu_info
    
    def occupy_memory(self, gpu_ids: List[int], memory_ratio: float):
        """
        占据指定GPU的显存
        
        Args:
            gpu_ids: GPU ID列表
            memory_ratio: 要占据的显存比例 (0.0-1.0)
        """
        print("开始占据显存...")
        print(f"目标GPU: {gpu_ids}")
        print(f"目标显存比例: {memory_ratio * 100:.1f}%")
        
        for gpu_id in gpu_ids:
            if gpu_id not in self.occupied_tensors:
                self.occupied_tensors[gpu_id] = []
            
            # 设置当前GPU
            torch.cuda.set_device(gpu_id)
            
            # 获取GPU信息
            gpu_info = self.get_gpu_info()[gpu_id]
            total_memory = gpu_info['total_memory']
            target_memory = int(total_memory * memory_ratio)
            
            print(f"\nGPU {gpu_id} ({gpu_info['name']}):")
            print(f"  总显存: {gpu_info['total_memory_gb']:.2f} GB")
            print(f"  目标占据: {target_memory / (1024**3):.2f} GB")
            
            try:
                # 计算要分配的tensor大小
                # 每个float32元素占4字节
                element_size = 4  # bytes
                num_elements = target_memory // element_size
                
                # 分批分配以避免单次分配过大
                batch_size = min(num_elements // 10, 100000000)  # 每次最多分配1亿元素
                allocated_memory = 0
                
                while allocated_memory < target_memory and self.is_running:
                    remaining = target_memory - allocated_memory
                    current_batch_elements = min(batch_size, remaining // element_size)
                    
                    if current_batch_elements > 0:
                        # 在GPU上分配tensor
                        tensor = torch.zeros(
                            current_batch_elements, 
                            dtype=torch.float32, 
                            device=f'cuda:{gpu_id}'
                        )
                        self.occupied_tensors[gpu_id].append(tensor)
                        allocated_memory += tensor.element_size() * tensor.numel()
                        
                        current_ratio = (allocated_memory / total_memory) * 100
                        print(f"  GPU {gpu_id}: 已分配 {allocated_memory / (1024**3):.2f} GB ({current_ratio:.1f}%)", end='\r')
                    
                    time.sleep(0.1)  # 短暂延迟，避免过于频繁分配
                
                print(f"\n  GPU {gpu_id}: 显存占据完成!")
                
            except RuntimeError as e:
                print(f"\n  GPU {gpu_id}: 分配显存时出错: {e}")
                break
    
    def cleanup(self):
        """释放所有占据的显存"""
        print("\n开始释放显存...")
        self.is_running = False
        
        for gpu_id, tensors in self.occupied_tensors.items():
            print(f"释放 GPU {gpu_id} 的显存...")
            for tensor in tensors:
                del tensor
            self.occupied_tensors[gpu_id] = []
        
        # 强制垃圾回收
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                torch.cuda.empty_cache()
                torch.cuda.synchronize(i)
        
        print("所有显存已释放!")
    
    def monitor_memory(self, gpu_ids: List[int]):
        """监控显存使用情况"""
        while self.is_running:
            print("\n当前显存使用情况:", end=' ')
            for gpu_id in gpu_ids:
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated(gpu_id) / (1024**3)
                    reserved = torch.cuda.memory_reserved(gpu_id) / (1024**3)
                    print(f"GPU{gpu_id}: {allocated:.2f}GB/{reserved:.2f}GB", end=' | ')
            print("按 Ctrl+C 退出", end='\r')
            time.sleep(10)

def main():
    parser = argparse.ArgumentParser(description='GPU显存占据工具')
    parser.add_argument('--ratio', type=float, default=0.5, 
                       help='要占据的显存比例 (0.0-1.0)，默认0.5')
    parser.add_argument('--gpus', type=str, default='0,1,2,3',
                       help='要占据的GPU ID，用逗号分隔，默认0,1,2,3')
    parser.add_argument('--no-monitor', action='store_true',
                       help='不显示显存监控信息')
    
    args = parser.parse_args()
    
    # 解析GPU ID
    try:
        gpu_ids = [int(x.strip()) for x in args.gpus.split(',')]
    except ValueError:
        print("错误: GPU ID格式不正确，请使用逗号分隔的数字，如 '0,1,2,3'")
        sys.exit(1)
    
    # 验证参数
    if args.ratio <= 0 or args.ratio > 1:
        print("错误: 显存比例必须在0.0到1.0之间")
        sys.exit(1)
    
    # 检查GPU可用性
    if not torch.cuda.is_available():
        print("错误: 未检测到CUDA设备")
        sys.exit(1)
    
    available_gpus = torch.cuda.device_count()
    for gpu_id in gpu_ids:
        if gpu_id >= available_gpus:
            print(f"错误: GPU {gpu_id} 不存在，可用GPU: 0-{available_gpus-1}")
            sys.exit(1)
    
    # 显示初始GPU信息
    occupier = GPUMemoryOccupier()
    gpu_info = occupier.get_gpu_info()
    
    print("=" * 60)
    print("GPU显存占据工具")
    print("=" * 60)
    print("初始GPU状态:")
    for gpu_id, info in gpu_info.items():
        print(f"GPU {gpu_id}: {info['name']} - {info['total_memory_gb']:.2f} GB")
    print("=" * 60)
    
    try:
        # 占据显存
        occupier.occupy_memory(gpu_ids, args.ratio)
        
        # 监控模式
        if not args.no_monitor:
            print("\n进入监控模式...")
            occupier.monitor_memory(gpu_ids)
        else:
            print("\n显存占据完成，保持运行中...")
            while occupier.is_running:
                time.sleep(1)
                
    except KeyboardInterrupt:
        print("\n用户中断")
    except Exception as e:
        print(f"\n发生错误: {e}")
    finally:
        occupier.cleanup()

if __name__ == "__main__":
    main()
