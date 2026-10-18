import numpy as np

def search_metrics(result, optimum_val=None, target_val=None):
    rows=result.get('rows',[])
    best_curve=np.asarray([float(r['best_val']) for r in rows],dtype=float)
    if not len(best_curve): best_curve=np.asarray([result['best_val_acc']],dtype=float)
    out={'best_val_acc':float(result['best_val_acc']),'test_acc':float(result['test_acc']),
         'search_auc':float(best_curve.mean()),'params':float(result['params']),'flops':float(result['flops'])}
    out['final_regret']=None if optimum_val is None else float(optimum_val-result['best_val_acc'])
    if target_val is None: out['evaluations_to_target']=None
    else:
        hits=np.flatnonzero(best_curve>=target_val)
        out['evaluations_to_target']=int(hits[0]+1) if len(hits) else None
    return out
