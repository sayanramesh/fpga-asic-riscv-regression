#!/usr/bin/env python3

import pandas as pd

def load_metrics(csv_path):
   
    df = pd.read_csv(csv_path, header=[0,1])
    
    df.columns = [
        'core',
        'asic_area', 'asic_fmax', 'asic_power', 'asic_nets', 'asic_registers',
        'asic45_area', 'asic45_fmax', 'asic45_power', 'asic45_nets', 'asic45_registers',
        'fpga_luts', 'fpga_fmax', 'fpga_power', 'fpga_nets', 'fpga_ffs',
        'fpga_distram', 'fpga_lookahead8', 'fpga_unicontrolsets'
    ]

    # drop empty rows
    df = df.dropna(subset=['core'])
    df = df[df['core'].str.strip() != '']
    df = df.reset_index(drop=True)

    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

if __name__ == "__main__":
    import sys
    df = load_metrics(sys.argv[1] if len(sys.argv) > 1 else 'metrics.csv')
    print(f"Loaded {len(df)} cores:")
    print(df[['core','fpga_luts','fpga_fmax','asic_area','asic_fmax']].to_string(index=False))