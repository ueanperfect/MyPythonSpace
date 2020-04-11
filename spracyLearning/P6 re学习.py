import re
match = re.search(r'[1‐9]\d{5}','BIT 100081')
if match:
    print(match.group(0))
match = re.match(r'[1‐9]\d{5}','100081 BIT')
if match:
    match.group(0)
ls = re.findall(r'[1‐9]\d{5}', 'BIT 100081 TSU100084')
re.split(r'[1‐9]\d{5}', 'BIT 100081 TSU100084')
re.split(r'[1‐9]\d{5}', 'BIT 100081 TSU100084', maxsplit=1)
for m in re.finditer(r'[1‐9]\d{5}', 'BIT100081 TSU100084'):
    if m: print(m.group(0))
    re.sub(r'[1‐9]\d{5}', ':zipcode','BIT100081 TSU100084')