"""Fail if a fllm_* function defined in rust/engine/cuda/*.cu (HIP stubs included) does not match its prototype in
fllm_cuda_internal.h / fllm_cuda.h. C++ treats a different parameter list as a new overload, so the compilers accept it and
only the link of the AMD or Windows build breaks."""
import re,sys,glob,os
d=os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','rust','engine','cuda')+os.sep
def strip(s):
    s=re.sub(r'//.*','',s); return re.sub(r'/\*.*?\*/','',s,flags=re.S)
def norm_params(p):
    out=[]
    for a in p.split(','):
        a=a.strip()
        if not a or a=='void': continue
        a=re.sub(r'=.*','',a).strip()
        m=re.match(r'(.*?)([A-Za-z_]\w*)?(\[\])?$',a)
        # drop trailing identifier if there is a type before it
        toks=re.findall(r'[A-Za-z_]\w*|\*|&',a)
        if len(toks)>1 and toks[-1] not in('int','float','char','void','bool','double','t','long') and re.match(r'[A-Za-z_]',toks[-1]) and toks[-2] not in ('unsigned','const','struct'):
            toks=toks[:-1]
        out.append(' '.join(toks).replace('* ','*').replace(' *','*'))
    return out
pat=re.compile(r'((?:[\w:]+\s+)+\**)\b(fllm_\w+)\s*\(([^()]*(?:\([^()]*\)[^()]*)*)\)\s*(;|\{)',re.S)
def scan(path):
    s=strip(open(path).read()); r=[]
    for m in pat.finditer(s):
        if m.group(1).strip().startswith(('return','else')): continue
        r.append((m.group(2),norm_params(m.group(3)),m.group(4)))
    return r
decl={}
for h in ('fllm_cuda_internal.h','fllm_cuda.h'):
    for n,p,t in scan(d+h):
        if t==';': decl.setdefault(n,[]).append((h,p))
bad=0
for f in sorted(glob.glob(d+'*.cu')):
    for n,p,t in scan(f):
        if t!='{' or n not in decl: continue
        if not any(p==q for _,q in decl[n]):
            bad+=1; print(f'{os.path.basename(f)}: {n} has {len(p)} params, header has {[len(q) for _,q in decl[n]]}')
print('mismatches',bad,'of decl',len(decl))
sys.exit(1 if bad else 0)
