"""Pure CPU parameter-packet composition; no simulator or experiment launcher."""
import numpy as np
from dr_factor_plan import GROUPS,validate

NAMES=('body_mass','body_inertia','body_ipos','body_iquat','geom_friction','actuator_gainprm','actuator_biasprm')
ALLOWED=('object_dynamics','friction','finger_gains','finger_offsets')

def mix_packets(base,donor,base_context,donor_context,group,object_body,hand_geoms,floor_geom,finger_ids):
 if group not in (*ALLOWED,None):raise ValueError('Unapproved factor')
 a=validate(base_context);b=validate(donor_context)
 if set(base)!=set(NAMES) or set(donor)!=set(NAMES):raise ValueError('Incomplete original model packet')
 result={}
 for k in NAMES:
  x,y=np.asarray(base[k]),np.asarray(donor[k])
  if x.shape!=y.shape or x.dtype!=y.dtype or not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('Model shape,dtype or finite-value mismatch')
  result[k]=x.copy()
 ctx=a.copy()
 if group is not None:ctx[list(GROUPS[group])]=b[list(GROUPS[group])]
 if group=='object_dynamics':
  for k in ('body_mass','body_inertia','body_ipos'):result[k][object_body]=donor[k][object_body]
 elif group=='friction':
  for ids in (hand_geoms,[floor_geom]):result['geom_friction'][ids,0]=donor['geom_friction'][ids,0]
 elif group=='finger_gains':
  result['actuator_gainprm'][finger_ids,0]=donor['actuator_gainprm'][finger_ids,0]
  result['actuator_biasprm'][finger_ids,1:3]=donor['actuator_biasprm'][finger_ids,1:3]
 validate(ctx)
 return result,ctx
