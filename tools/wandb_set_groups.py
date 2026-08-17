#!/usr/bin/env python3
"""Set the WandB `group` (and `job_type`) of runs from a batch job-id TSV.

Runs are logged to WandB with no group, so seeds of the same arm are scattered
across the dashboard.  This walks a `job_logs/*.tsv` batch file, maps each
experiment tag to its arm's display name, and writes `group` = arm and
`job_type` = task onto the matching WandB run.  With both set you can group the
dashboard by arm and still split the two tasks apart (group by group, job_type),
which matters because one arm spans several tasks.

Only the two fields are touched: the mutation is a partial update, so config,
summary and history are left alone (verified on e549 -- 337 summary keys and
433 config keys unchanged).

Uses stdlib + the GraphQL endpoint directly, so it runs under the system python
on the login node; the `wandb` package and dreamerv3_env are not needed.
WANDB_API_KEY must be set (source_dreamerv3_env.sh exports it).

    tools/wandb_set_groups.py job_logs/e510_e557_goal_ae_comparison.tsv
    DRY_RUN=1 tools/wandb_set_groups.py <tsv>          # show, change nothing

The TSV columns are those written by the submit_* scripts:
    ts  jobid  tag  task  arm  seed  name  partition  steps
"""
import base64
import json
import os
import re
import sys
import urllib.request

ENTITY = os.environ.get('WANDB_ENTITY', 'vasilache')
PROJECT = os.environ.get('WANDB_PROJECT', 'dreamerv3')
URL = 'https://api.wandb.ai/graphql'

# arm (as written in the batch TSV) -> group name shown on the dashboard.
ARM2GROUP = {
    'director': 'Director',
    'som_line': 'SOM(line)',
    'som_orig_line': 'SOM(OG,line)',
    'lipvq_prod': 'LipVQ',
    'som_lipvq_line_prod': 'SOM(line)+LipVQ',
    'som_orig_lipvq_line_prod': 'SOM(OG,line)+LipVQ',
}


def gql(query, variables=None):
    body = json.dumps({'query': query, 'variables': variables or {}}).encode()
    req = urllib.request.Request(URL, data=body, method='POST')
    key = os.environ['WANDB_API_KEY']
    req.add_header(
        'Authorization',
        'Basic ' + base64.b64encode(('api:' + key).encode()).decode())
    req.add_header('Content-Type', 'application/json')
    with urllib.request.urlopen(req, timeout=60) as resp:
        out = json.loads(resp.read().decode())
    if out.get('errors'):
        raise RuntimeError(json.dumps(out['errors'])[:800])
    return out['data']


LIST = '''
query Runs($entity: String!, $project: String!, $cursor: String) {
  project(name: $project, entityName: $entity) {
    runs(first: 500, after: $cursor) {
      pageInfo { hasNextPage endCursor }
      edges { node { id name displayName group jobType state } }
    }
  }
}
'''

UPDATE = '''
mutation SetGroup($id: String!, $groupName: String, $jobType: String) {
  upsertBucket(input: {id: $id, groupName: $groupName, jobType: $jobType}) {
    bucket { id group jobType }
  }
}
'''


def list_runs():
    runs, cursor = [], None
    while True:
        page = gql(LIST, {
            'entity': ENTITY, 'project': PROJECT, 'cursor': cursor,
        })['project']['runs']
        runs += [e['node'] for e in page['edges']]
        if not page['pageInfo']['hasNextPage']:
            return runs
        cursor = page['pageInfo']['endCursor']


def main(tsv):
    meta = {}
    with open(tsv) as f:
        for line in f:
            cols = line.rstrip('\n').split('\t')
            if len(cols) >= 5:
                meta[cols[2]] = (cols[3].replace('dmc_', ''), cols[4])

    unknown = sorted({a for _, a in meta.values()} - set(ARM2GROUP))
    if unknown:
        sys.exit('no group name for arm(s): %s -- add them to ARM2GROUP'
                 % ', '.join(unknown))

    todo = []
    for run in list_runs():
        m = re.search(r'/(e[0-9]+)_', run['displayName'] or '')
        if m and m.group(1) in meta:
            task, arm = meta[m.group(1)]
            todo.append((m.group(1), run, ARM2GROUP[arm], task))
    todo.sort(key=lambda t: t[0])

    missing = sorted(set(meta) - {t[0] for t in todo})
    if missing:
        print('warning: no WandB run found for %d tag(s): %s'
              % (len(missing), ' '.join(missing)))

    dry = os.environ.get('DRY_RUN') == '1'
    done = skipped = failed = 0
    for tag, run, group, task in todo:
        if run['group'] == group and run['jobType'] == task:
            skipped += 1
            continue
        print('%s%-6s -> group=%-20s job_type=%s'
              % ('[dry-run] ' if dry else '', tag, group, task))
        if dry:
            continue
        try:
            got = gql(UPDATE, {
                'id': run['id'], 'groupName': group, 'jobType': task,
            })['upsertBucket']['bucket']
            assert got['group'] == group and got['jobType'] == task, got
            done += 1
        except Exception as e:  # keep going; report at the end
            failed += 1
            print('  %s FAILED: %s' % (tag, e), file=sys.stderr)
    print('updated=%d already-correct=%d failed=%d' % (done, skipped, failed))
    return 1 if failed else 0


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1]))
