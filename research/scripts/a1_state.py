"""Full-A1 causal state contract. World poses; two hand slots, object, presence masks.

Per hand (137): q20,dq20,p3,quat_wxyz4,v_world3,w_world3,
three requested absolute controls26 (oldest first), servo pose6.
Object (13): p3,quat4,v3,w3. Total287; masks and physical context separate.
Commands are complete base+residual targets after limit, BEFORE delay. This
resolves the source document's ambiguity between normalized residual and control.
No future true state/contact labels enter any prediction.
"""
import torch
from world_model_mini import qmatrix,qmul,qconj,qexp,qlog,qnormalize
H=137; STATE=287; DELTA=116; ENCODED=337; DYNAMICS=125; COMMAND_ENCODED=58
Q=slice(0,20);DQ=slice(20,40);P=slice(40,43);R=slice(43,47);V=slice(47,50);W=slice(50,53);HISTORY=slice(53,131);SERVO=slice(131,137)
def euler_matrix(x):
    e=torch.eye(3,dtype=x.dtype,device=x.device)
    r=qmatrix(qexp(x[...,0,None]*e[0]))
    for j in [1,2]:r=r@qmatrix(qexp(x[...,j,None]*e[j]))
    return r
def euler_quaternion(x):
    e=torch.eye(3,dtype=x.dtype,device=x.device)
    return qmul(qmul(qexp(x[...,0,None]*e[0]),qexp(x[...,1,None]*e[1])),qexp(x[...,2,None]*e[2]))
def matrix_euler(r):
    # Intrinsic XYZ, the same convention as MuJoCo's chained x/y/z hinges.
    return torch.stack([torch.atan2(-r[...,1,2],r[...,2,2]),torch.asin(r[...,0,2].clamp(-1+1e-7,1-1e-7)),torch.atan2(-r[...,0,1],r[...,0,0])],-1)
def mat6(r):return r[...,:,:2].reshape(*r.shape[:-2],6)
def yaw_frame(s,mask):
    hands=s[...,:274].reshape(*s.shape[:-1],2,H)
    primary=torch.where((mask[...,0]>0)[...,None],hands[...,0,:],hands[...,1,:])
    r=qmatrix(primary[...,R]);yaw=torch.atan2(r[...,1,0],r[...,0,0])
    c=yaw.cos();sn=yaw.sin();z=torch.zeros_like(c);one=torch.ones_like(c)
    world_from_local=torch.stack([c,-sn,z,sn,c,z,z,z,one],-1).reshape(*c.shape,3,3)
    origin=torch.cat([primary[...,40:42],z[...,None]],-1)
    return world_from_local,origin
def local_vec(v,frame):return (frame.transpose(-1,-2)@v[...,None]).squeeze(-1)
def world_vec(v,frame):return (frame@v[...,None]).squeeze(-1)
def encode_command(u,frame,origin):
    pos=local_vec(u[...,:3]-origin,frame)
    rot=frame.transpose(-1,-2)@euler_matrix(u[...,3:6])
    return torch.cat([pos,mat6(rot),u[...,6:]],-1)
def encode(s,mask):
    frame,origin=yaw_frame(s,mask);hands=s[...,:274].reshape(*s.shape[:-1],2,H);obj=s[...,274:];features=[]
    for h in range(2):
        a=hands[...,h,:];wr=qmatrix(a[...,R])
        dyn=torch.cat([a[...,:40],local_vec(a[...,P]-origin,frame),mat6(frame.transpose(-1,-2)@wr),local_vec(a[...,V],frame),local_vec(a[...,W],frame)],-1)
        hist=a[...,HISTORY].reshape(*s.shape[:-1],3,26)
        controls=torch.cat([encode_command(hist[...,j,:],frame,origin) for j in range(3)],-1)
        servo=encode_command(torch.cat([a[...,SERVO],torch.zeros_like(a[...,:20])],-1),frame,origin)[...,:9]
        features.append(torch.cat([dyn,controls,servo],-1)*mask[...,h,None])
    features.append(torch.cat([local_vec(obj[...,:3]-origin,frame),mat6(frame.transpose(-1,-2)@qmatrix(obj[...,3:7])),local_vec(obj[...,7:10],frame),local_vec(obj[...,10:13],frame)],-1))
    for h in range(2):
        a=hands[...,h,:];wr=qmatrix(a[...,R])
        relative=torch.cat([local_vec(obj[...,:3]-a[...,P],wr),mat6(wr.transpose(-1,-2)@qmatrix(obj[...,3:7]))],-1)
        features.append(relative*mask[...,h,None])
    features.append(mask)
    out=torch.cat(features,-1);assert out.shape[-1]==ENCODED
    return out
def encode_dynamics(s):
    hs=s[...,:274].reshape(*s.shape[:-1],2,H);parts=[]
    for h in range(2):
        a=hs[...,h,:];parts.append(torch.cat([a[...,:43],mat6(qmatrix(a[...,R])),a[...,V],a[...,W]],-1))
    o=s[...,274:];parts.append(torch.cat([o[...,:3],mat6(qmatrix(o[...,3:7])),o[...,7:]],-1))
    out=torch.cat(parts,-1);assert out.shape[-1]==DYNAMICS;return out
def delta_between(s,target,mask):
    frame,_=yaw_frame(s,mask);ha=s[...,:274].reshape(*s.shape[:-1],2,H);hb=target[...,:274].reshape(*s.shape[:-1],2,H);parts=[]
    for h in range(2):
        a,b=ha[...,h,:],hb[...,h,:]
        # Differences use the SAME start-state frame. Never subtract two moving canonical states.
        parts.append(torch.cat([b[...,:40]-a[...,:40],local_vec(b[...,P]-a[...,P],frame),qlog(qmul(qconj(a[...,R]),b[...,R])),local_vec(b[...,V]-a[...,V],frame),local_vec(b[...,W]-a[...,W],frame)],-1)*mask[...,h,None])
    a,b=s[...,274:],target[...,274:]
    parts.append(torch.cat([local_vec(b[...,:3]-a[...,:3],frame),qlog(qmul(qconj(a[...,3:7]),b[...,3:7])),local_vec(b[...,7:10]-a[...,7:10],frame),local_vec(b[...,10:]-a[...,10:],frame)],-1))
    out=torch.cat(parts,-1);assert out.shape[-1]==DELTA;return out
def analytic_control(s,requested,mask,delay,lag_tau,zero_offset,dt=.02):
    """Isotropic position and geodesic SO(3) lag; stored as six pose coordinates.

    delay[...,hand,0] is wrist delay0..3, delay[...,hand,1] finger delay0..3.
    lag_tau[...,hand,6] and zero_offset[...,hand,20] are fixed known episode context.
    This is a declared simulated actuator assumption until measured xArm parameters exist.
    """
    hs=s[...,:274].reshape(*s.shape[:-1],2,H);u=requested.reshape(*s.shape[:-1],2,26);applied=[];servo=[]
    for h in range(2):
        a=hs[...,h,:];hist=a[...,HISTORY].reshape(*s.shape[:-1],3,26)
        queue=torch.cat([hist,u[...,h,None,:]],-2)
        index=(3-delay[...,h,:].long()).clamp(0,3)
        wrist=torch.gather(queue,-2,index[...,0,None,None].expand(*index.shape[:-1],1,26)).squeeze(-2)[...,:6]
        finger=torch.gather(queue,-2,index[...,1,None,None].expand(*index.shape[:-1],1,26)).squeeze(-2)[...,6:]
        alpha=dt/(lag_tau[...,h,:].clamp_min(0)+dt)
        position=a[...,131:134]+alpha[...,:3]*(wrist[...,:3]-a[...,131:134])
        current_q=euler_quaternion(a[...,134:137]);target_q=euler_quaternion(wrist[...,3:6])
        rotation=qmul(current_q,qexp(alpha[...,3,None]*qlog(qmul(qconj(current_q),target_q))))
        pose=torch.cat([position,matrix_euler(qmatrix(rotation))],-1)
        servo.append(pose);applied.append(torch.cat([pose,finger+zero_offset[...,h,:]],-1)*mask[...,h,None])
    return torch.stack(applied,-2),torch.stack(servo,-2)
def compose(s,delta,requested,mask,delay,lag_tau,zero_offset):
    frame,_=yaw_frame(s,mask);hs=s[...,:274].reshape(*s.shape[:-1],2,H);u=requested.reshape(*s.shape[:-1],2,26)
    _,servo=analytic_control(s,requested,mask,delay,lag_tau,zero_offset);parts=[]
    for h in range(2):
        a=hs[...,h,:];d=delta[...,h*52:(h+1)*52]
        dyn=torch.cat([a[...,:40]+d[...,:40],a[...,P]+world_vec(d[...,40:43],frame),qnormalize(qmul(a[...,R],qexp(d[...,43:46]))),a[...,V]+world_vec(d[...,46:49],frame),a[...,W]+world_vec(d[...,49:52],frame)],-1)
        hist=torch.cat([a[...,79:131],u[...,h,:]],-1)
        out=torch.cat([dyn,hist,servo[...,h,:]],-1)
        parts.append(torch.where(mask[...,h,None]>0,out,a))
    a=s[...,274:];d=delta[...,104:]
    parts.append(torch.cat([a[...,:3]+world_vec(d[...,:3],frame),qnormalize(qmul(a[...,3:7],qexp(d[...,3:6]))),a[...,7:10]+world_vec(d[...,6:9],frame),a[...,10:]+world_vec(d[...,9:12],frame)],-1))
    return torch.cat(parts,-1)
def empty_state(batch,device='cpu',dtype=torch.float32):
    s=torch.zeros(batch,STATE,device=device,dtype=dtype)
    s[:,43]=1;s[:,H+43]=1;s[:,277]=1
    return s
