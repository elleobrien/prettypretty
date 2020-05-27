import json
import sys

j = json.load(sys.stdin)
print(list(j.values())[0])
