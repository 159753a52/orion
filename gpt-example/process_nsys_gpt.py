"""
Process NSYS GPU trace output to generate kernel info file for Orion.
This is a simplified version that works without NCU profiling.

Usage:
    python process_nsys_gpt.py --input profiling_results/output_nsys_gputrace.csv \
                               --output profiling_results/gpt_kernel_info.csv
"""

import argparse
import pandas as pd
import math


def estimate_sm_usage(grid_x, grid_y, grid_z, block_x, block_y, block_z, max_sms=56):
    """
    Estimate SM usage based on grid dimensions.
    This is a simplified estimation - actual SM usage depends on many factors.
    
    For A30 GPU: 56 SMs
    For V100 GPU: 80 SMs
    For A100 GPU: 108 SMs
    """
    if pd.isna(grid_x) or pd.isna(block_x):
        return 1  # memcpy or other non-kernel operations
    
    total_blocks = int(grid_x) * int(grid_y) * int(grid_z)
    # Estimate: each block uses at least one SM, but we cap at max_sms
    sm_used = min(total_blocks, max_sms)
    return max(1, sm_used)


def simplify_kernel_name(name):
    """Simplify kernel name for easier reading."""
    if pd.isna(name):
        return "Unknown"
    
    name = str(name)
    
    # Handle common patterns
    if 'sgemm' in name.lower():
        return 'GEMM'
    elif 'softmax' in name.lower():
        return 'Softmax'
    elif 'layer_norm' in name.lower() or 'layernorm' in name.lower():
        return 'LayerNorm'
    elif 'elementwise' in name.lower():
        return 'Elementwise'
    elif 'indexSelect' in name:
        return 'IndexSelect'
    elif 'memcpy' in name.lower():
        return 'Memcpy'
    elif 'masked_fill' in name:
        return 'MaskedFill'
    elif 'arange' in name:
        return 'Arange'
    else:
        # Extract first meaningful part
        tokens = name.split('<')[0].split('::')
        return tokens[-1] if tokens else name[:50]


def process_nsys_trace(input_file, output_file, max_sms=56):
    """Process NSYS gputrace CSV and generate kernel info file."""
    
    print(f"Reading {input_file}...")
    df = pd.read_csv(input_file)
    
    print(f"Found {len(df)} kernel records")
    
    results = []
    
    for idx, row in df.iterrows():
        name = row.get('Name', 'Unknown')
        duration_ns = row.get('Duration (ns)', 0)
        
        # Get grid and block dimensions
        grid_x = row.get('GrdX', 1)
        grid_y = row.get('GrdY', 1)
        grid_z = row.get('GrdZ', 1)
        block_x = row.get('BlkX', 1)
        block_y = row.get('BlkY', 1)
        block_z = row.get('BlkZ', 1)
        
        # Estimate SM usage
        sm_used = estimate_sm_usage(grid_x, grid_y, grid_z, 
                                     block_x, block_y, block_z, max_sms)
        
        # Simplify kernel name
        simple_name = simplify_kernel_name(name)
        
        # Profile: -1 means unknown (would need NCU for accurate roofline analysis)
        # For GEMM operations, we can assume compute-bound (1)
        # For memcpy, assume memory-bound (0)
        if 'gemm' in str(name).lower() or 'sgemm' in str(name).lower():
            profile = 1  # compute-bound
        elif 'memcpy' in str(name).lower():
            profile = 0  # memory-bound
        else:
            profile = -1  # unknown
        
        results.append({
            'Name': simple_name,
            'Profile': profile,
            'Memory_footprint': 0,
            'SM_usage': sm_used,
            'Duration': duration_ns
        })
    
    # Create DataFrame and save
    result_df = pd.DataFrame(results)
    result_df.to_csv(output_file, index=False)
    
    print(f"\nGenerated kernel info file: {output_file}")
    print(f"Total kernels: {len(result_df)}")
    print(f"\nKernel type distribution:")
    print(result_df['Name'].value_counts().head(10))
    print(f"\nTotal execution time: {result_df['Duration'].sum() / 1e6:.2f} ms")
    
    return result_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process NSYS trace for Orion')
    parser.add_argument('--input', type=str, required=True,
                        help='Input NSYS gputrace CSV file')
    parser.add_argument('--output', type=str, required=True,
                        help='Output kernel info CSV file')
    parser.add_argument('--max_sms', type=int, default=56,
                        help='Maximum SMs on the GPU (A30=56, V100=80, A100=108)')
    
    args = parser.parse_args()
    process_nsys_trace(args.input, args.output, args.max_sms)
