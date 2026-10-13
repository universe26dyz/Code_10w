"""Authoritative TRAD/HHZ dictionary matching for the Phase-2 baseline."""
from __future__ import annotations
import numpy as np
from ..simulation.simulate_hhz_signals import generate_signal_field
from .contracts import FROZEN_GRID

def frozen_axis(spec: tuple[float, float, float, float, float, float, float, float, float]) -> np.ndarray:
    a,b,c,d,e,f,g,h,i = spec
    return np.concatenate((np.arange(a, c + 0.1, b), np.arange(d, f + 0.1, e), np.arange(g, i + 0.1, h)))

def build_dictionary(t1_ms: np.ndarray, t2_ms: np.ndarray, b1: np.ndarray, timing9_ms: np.ndarray, tr_ms: float, vps: int) -> dict[str, np.ndarray]:
    t1,t2,b1=np.asarray(t1_ms,float).reshape(-1),np.asarray(t2_ms,float).reshape(-1),np.asarray(b1,float).reshape(-1)
    if not (t1.shape == t2.shape == b1.shape): raise ValueError("dictionary parameter arrays must match")
    raw=generate_signal_field(t1.astype(np.float32),t2.astype(np.float32),b1.astype(np.float32),np.asarray(timing9_ms,float),tr_ms=tr_ms,vps=vps)
    return {"t1_ms":t1,"t2_ms":t2,"b1":b1,"fingerprints":raw/np.linalg.norm(raw,axis=1,keepdims=True).clip(1e-12)}

def frozen_dictionary(timing9_ms: np.ndarray, tr_ms: float, vps: int) -> dict[str,np.ndarray]:
    if (tr_ms,vps)!=(2.61,87): raise ValueError("frozen primary dictionary requires TR=2.61/VPS=87")
    t1,t2,b1=frozen_axis(FROZEN_GRID["t1_ms"]),frozen_axis(FROZEN_GRID["t2_ms"]),np.arange(.1,1.2001,.05)
    x,y,z=np.meshgrid(t1,t2,b1,indexing="ij"); keep=x>y
    return build_dictionary(x[keep],y[keep],z[keep],timing9_ms,tr_ms,vps)

def match_fingerprints(dictionary_fingerprints: np.ndarray, measured: np.ndarray, pixel_chunk: int = 256, dictionary_chunk: int = 32768) -> np.ndarray:
    d=np.asarray(dictionary_fingerprints,float); m=np.asarray(measured,float)
    m=m.reshape(-1,10); m=m/np.linalg.norm(m,axis=1,keepdims=True).clip(1e-12)
    result=np.empty(len(m),dtype=np.int64)
    for begin in range(0,len(m),pixel_chunk):
        block=m[begin:begin+pixel_chunk]; best=np.full(len(block),-np.inf); index=np.zeros(len(block),dtype=np.int64)
        for offset in range(0,len(d),dictionary_chunk):
            score=block @ d[offset:offset+dictionary_chunk].T; local=np.argmax(score,axis=1); values=score[np.arange(len(block)),local]; update=values>best
            best[update]=values[update]; index[update]=offset+local[update]
        result[begin:begin+len(block)]=index
    return result
