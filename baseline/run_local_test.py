import json, sys, time
from pathlib import Path
from myalgorithm import algorithm
from utils import check_feasibility

if len(sys.argv) < 2:
    print('Usage: python run_local_test.py path/to/prob.json [timelimit]')
    raise SystemExit(1)

path = Path(sys.argv[1])
timelimit = float(sys.argv[2]) if len(sys.argv) > 2 else 60.0
prob = json.load(open(path))
t0 = time.time()
sol = algorithm(prob, timelimit)
elapsed = time.time() - t0
res = check_feasibility(prob, sol)
print('Instance :', prob.get('name', path.name))
print('Elapsed  :', round(elapsed, 3), 'sec')
print('Feasible :', res.get('feasible'), 'stage=', res.get('stage'))
if res.get('feasible'):
    print('Objective:', round(res['objective'], 2), 'obj1=', res['obj1'], 'obj2=', res['obj2'], 'obj3=', res['obj3'])
else:
    print('Violations:', res.get('violations', [])[:5])
