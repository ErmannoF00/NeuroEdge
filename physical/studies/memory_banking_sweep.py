#!/usr/bin/env python3
"""existing analytical storage/banking inventory, *not* mapped macro power/area.
Input macro geometry must come from authorized IHP/physical library data.
"""
import argparse,json,math
from pathlib import Path

STORES={
 'weights':{'depth':44224,'width':8,'ports':'1RW','read_latency_cycles':1},
 'metadata':{'depth':361,'width':37,'ports':'1RW','read_latency_cycles':1},
 'offsets':{'depth':512,'width':16,'ports':'1R+1W logically gated', 'read_latency_cycles':1},
 'offset_valid_mask':{'depth':512,'width':1,'ports':'resettable flags'},
}

def candidates(rows: int, width: int, bank_depth: int, macro_width: int):
    if min(rows,width,bank_depth,macro_width)<=0:
        raise ValueError('dimensions must be positive')
    naddr=math.ceil(rows/bank_depth)
    nwide=math.ceil(width/macro_width)
    return {'macro_depth':bank_depth,'macro_width':macro_width,
            'macro_count':naddr*nwide,'macro_rows':naddr,'macro_columns':nwide,
            'capacity_bits':naddr*nwide*bank_depth*macro_width,
            'logical_bits':rows*width,
            'capacity_utilization':(rows*width)/(naddr*nwide*bank_depth*macro_width),
            'requires_collision_semantics_review':True}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--depths',nargs='+',type=int,default=[256,512,1024,2048])
    ap.add_argument('--widths',nargs='+',type=int,default=[8,16,32])
    a=ap.parse_args()
    data={'basis':'LOGICAL CAPACITY ONLY, NO PDK MACRO LIBRARY OR PPA CLAIM',
          'logical_bit_inventory':STORES,
          'total_bits_excluding_valid':sum(v['depth']*v['width'] for k,v in STORES.items() if k!='offset_valid_mask'),
          'hypothetical_macro_geometries':{k:[candidates(v['depth'],v['width'],d,w)
              for d in a.depths for w in a.widths] for k,v in STORES.items() if k!='offset_valid_mask'},
          'external_dependencies':['verified foundry macro catalog','Liberty+LEF','read-old collision contract',
             'scan/reset policy for valid mask','post-route SDC/STA and power activity']}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(data,indent=2)+'\n')
    print('LOGICAL_MEMORY_INVENTORY_GENERATED (NOT SILICON MEASUREMENT)')
if __name__=='__main__': main()
