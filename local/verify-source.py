from pathlib import Path
import hashlib,json,sys
root=Path(__file__).resolve().parents[1]
m=json.loads((root/'local/source.lock.json').read_text())
for f,h in m['product_files'].items():
 if hashlib.sha256((root/f).read_bytes()).hexdigest()!=h:raise SystemExit('Source identity mismatch: '+f)
print('PASS committed product source identities')
