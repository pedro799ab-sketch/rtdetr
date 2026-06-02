#!/usr/bin/env python3
"""
Convert the detailed CSV to a simplified version with only:
- K values
- Target Time
- Actual Decoder Time
- Decoder Power
"""

import csv
import sys

def convert_csv(input_file, output_file):
    """Extract specific columns from the detailed CSV."""
    
    with open(input_file, 'r') as infile:
        reader = csv.DictReader(infile)
        
        with open(output_file, 'w', newline='') as outfile:
            writer = csv.writer(outfile)
            
            # Write header
            writer.writerow(['K', 'Target_Time_s', 'Actual_Decoder_Time_s', 'Decoder_Power_W'])
            
            # Write data rows
            for row in reader:
                writer.writerow([
                    row['K'],
                    row['Target_Time_s'],
                    row['Decoder_Time_s'],
                    row['Decoder_Power_W']
                ])
    
    print(f"Converted {input_file} -> {output_file}")
    print(f"Columns: K, Target_Time_s, Actual_Decoder_Time_s, Decoder_Power_W")

if __name__ == "__main__":
    input_csv = "power_vs_K_different_times.csv"
    output_csv = "power_vs_K_simplified.csv"
    
    if len(sys.argv) > 1:
        input_csv = sys.argv[1]
    if len(sys.argv) > 2:
        output_csv = sys.argv[2]
    
    convert_csv(input_csv, output_csv)
