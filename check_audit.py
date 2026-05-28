import json, os

path = 'Exp/reports/data_audit_report.json'
if not os.path.exists(path):
    print("File not found — audit was run on Kaggle, not locally")
else:
    d = json.load(open(path))
    t = d['train']
    b = t['binary_distribution']
    total = b['change_pixels'] + b['no_change_pixels']
    print(f"Change pixels:    {b['change_pixels']:,}")
    print(f"No-change pixels: {b['no_change_pixels']:,}")
    print(f"Change %:         {b['change_pixels']/total*100:.2f}%")
    print(f"No-Change %:      {b['no_change_pixels']/total*100:.2f}%")
    print(f"Imbalance ratio:  {b['imbalance_ratio']}:1")
    print()
    print("4-class breakdown:")
    for cls, info in t['class_distribution_original'].items():
        print(f"  {cls}: {info['pixels']:>14,}  ({info['fraction']*100:.2f}%)")
