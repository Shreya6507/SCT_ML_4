path = 'data/gestures.csv'
with open(path, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        parts = line.rstrip('\n').split(',')
        # show last 5 tokens to inspect label placement
        print(i, 'tokens=', len(parts), 'last5=', parts[-6:])
