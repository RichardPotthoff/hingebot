---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.17.0
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

<!-- #region jupyter={"outputs_hidden": false, "source_hidden": false} pythonista={"disabled": false} -->
# Hingebot
<!-- #endregion -->

<!-- #region jupyter={"outputs_hidden": false, "source_hidden": false} pythonista={"disabled": false} -->
## Common Subroutines
<!-- #endregion -->

<!-- #region jupyter={"outputs_hidden": false, "source_hidden": false} pythonista={"disabled": false} -->
### plotArc, plotArcchain, calcTangent, pipe, iterize
<!-- #endregion -->

```python jupyter={"outputs_hidden": false, "source_hidden": false} pythonista={"disabled": false}
import numpy as np
from matplotlib import pyplot as plt
from cmath import pi,acos,exp,sqrt
def plotArc(ax,P0,n0,l,da,*args,tol=0.001,**kwargs):
  if l==0:
    return
  x=np.linspace(0,l,max(2,int(abs(6*(da/(2*pi)))),int(l//(2*abs(2*l/da*tol)**0.5)+1))if (da!=0) and (l!=0) else 2)
  phi2=x/l*da/2
  p=P0+x*np.sinc(phi2/pi)*n0*np.exp(1j*phi2)
  ax.plot(p.real,p.imag,*args,**kwargs)
    
def plotArcchain(ax,P0,n0,arcs,*args,**kwargs):
    p=P0
    n=n0
    for l,da in arcs:
        plotArc(ax,p,n,l,da,*args,**kwargs)
        p+=l*np.sinc(da/(2*pi))*n*exp(1j*da/2)
        n*=exp(1j*da)
        
def calcTangent(c1,r1,c2,r2):
    c2_c1=(c2-c1)
    lcc=abs(c2_c1) 
    ec2_c1=c2_c1/lcc
    cosphi=((r1+r2)/lcc)
    phi=-cosphi+1j*(1-cosphi**2)**0.5
    t1=c1-r1*ec2_c1*phi
    t2=c2+r2*ec2_c1*phi
    return [t1,t2]
    
        
def pipe(*f):#reverses the order of chained function calls: f3(f2(f1(x))) = pipe(f1,f2,f3)(x)
  if len(f)==0:
    def g(*x):
      return None
  else:
    def g(*x):
      x=f[0](*x)
      for fi in f[1:]:
        if fi==None:
            continue
        x=fi(x)
      return x
  return g
  
      
def iterize(f):
    if f==None:
      def wrapper(*args,**kwargs):
        if len(args)==0:
           return None
        else:
           return args[0]
    else:  
      def wrapper(*args,**kwargs):
        if len(args)==0:
            return f(**kwargs)
        else: 
            t=args[0]
            return (f(t_,*args[1:],**kwargs) for t_ in t) if hasattr(t,'__next__') else f(*args,**kwargs)
      wrapper.__name__='iterized_'+f.__name__
    return wrapper
```

<!-- #region jupyter={"outputs_hidden": false, "source_hidden": false} pythonista={"disabled": false} -->
### NamedList
<!-- #endregion -->

```python jupyter={"source_hidden": false} pythonista={"disabled": false}
class NamedList(list): 
    '''
    A mutable version of namedtuple.
    '''
    def __init__(self,*args,**kwargs):
        super().__init__(*args)
        super().__setattr__('_lookup',dict())
        self._lookup.update({ f'_{i}':i for i in range(len(self))})
        self.update(**kwargs)
    def __setattr__(self,name,value):
           self.update(**{name:value})
    def __getattr__(self,name):
        try: 
          return self[self._lookup[name]]
        except:
          raise AttributeError(f"Found no attribute named '{name}'.  ")
    def append(self,value):
        raise IndexError(f"Use the 'update' method to add elements to a {self.__class__.__name}!") 
    def extend(self,value):
        raise IndexError(f"Use the 'update' method to add elements to a {self.__class__.__name}!") 
    def update(self,*args,**kwargs): 
        if len(args)>0:
           if len(args)>len(self):
              self[:]=args[:len(self)]
              self._lookup.update({f'_{i}':i for i in range(len(self),len(args))})
              super().extend(args[len(self):])
           else:
              self[:len(args)]=args
        for name,value in kwargs.items():
            if  not name in self._lookup:
                if name[0]=='_':
                    new_index=int(name[1:])
                    self._lookup.update({f'_{i}':i for i in range(len(self),new_index+1)})
                    super().extend([None]*(new_index-len(self)+1))
                    self[new_index]=value
               #     KeyError(f"Error in 'update': Field '{name}' does not exist.")
                else:
                   self._lookup[f'_{len(self)}']=len(self)
                   self._lookup[name]=len(self)
                   super().append(value)
            else:
                self[self._lookup[name]]=value
        return self
    def alias(self,*args,**kwargs):
        if len(args)>0:
            if len(args)>len(self):
                self._lookup.update({f'_{i}':i for i in range(len(self),len(args))})
                super().extend([None]*(len(args)-len(self)))
            self._lookup.update({name:i for i,name in enumerate(args)})
        for name,alias in kwargs.items():
            if type(alias)!=str:
                raise KeyError(f"Error in 'alias': Field name '{alias}' must be a  valid variable name.")
            if alias[0]=='_':
                raise KeyError(f"Error in 'alias': Field name '{alias}' must not start with an underscore ('_').")
            if alias in self._lookup:
                if self._lookup[alias]==self._lookup[name]:
                    return
                else:
                    raise KeyError(f"The alias name '{alias}' is already used for a different field.")
            self._lookup[alias]=self._lookup[name]
    equivalence=alias
    def rename(self,**kwargs):
        for old_name,new_name in kwargs.items():
            if old_name[0]=='_':
                raise KeyError(f"The name '{old_name}' cannot be renamed.")
            
            if (new_name!=None) and (new_name[0]=='_'):
                raise KeyError(f"Error in 'rename': Field name '{new_name}' must not start with an underscore ('_').")
            old_index=self._lookup.pop(old_name,None)
            if old_index==None:
                raise KeyError(f"The name '{old_name}' does not exist.")
            if new_name in self._lookup:
                if self._lookup[new_name]==old_index:
                    return
                else:
                    raise KeyError(f"The name '{alias}' is already used for a different member of the list.")
            if new_name!=None:
                self._lookup[new_name]=old_index
    def __repr__(self):
        args=", ".join(f'{key}={value}' for key,value in self.as_dict().items())
        return f'{self.__class__.__name__}({args})' 
    def as_dict(self):
        key_for_index={index:key for key,index in self._lookup.items()}
        return {key_for_index.get(index,index):value for index,value in enumerate(self)}
    def copy(self):
        myCopy=self.__class__(*self)
        super(myCopy.__class__,myCopy).__setattr__('_lookup',self._lookup.copy())
        return myCopy

def update_listitem(list,index,value):
    list[index]=value
    return list
```

```python jupyter={"outputs_hidden": false, "source_hidden": false} pythonista={"disabled": false}
class make_iterable:
  def __init__(self,f,*args,**kwargs):
    self.fgen=f
    self.args=args
    self.kwargs=kwargs
  def __iter__(self):
    yield from self.fgen(*self.args,**self.kwargs)
  def __call__(self):
    yield from self.fgen(*self.args,**self.kwargs)
  def __repr__(self):
    return f'{self.__class__.__name__}({",".join((self.fgen.__repr__(),*map(str,self.args),*(str(k)+"="+str(v) for k,v in self.kwargs.items()))) })'
    
from itertools import takewhile

#from bisect import bisect_right
def bisect_right(a,x,lo=0,hi=None):
    if hi is None:
        hi = len(a)
    while lo < hi:
        mid = (lo + hi) // 2
        if x < a[mid]:
            hi = mid
        else:
            lo = mid + 1
    return lo

def index_frac(x,ax):
    idx=bisect_right(ax,x,1,len(ax)-1)
    return idx-1,(x-ax[idx-1])/(ax[idx]-ax[idx-1])
    
def interp(x,ax,ay):
    idx=bisect_right(ax,x,1,len(ax)-1)
    return ay[idx-1]+(x-ax[idx-1])/(ax[idx]-ax[idx-1])*(ay[idx]-ay[idx-1])

def cumsum(x,x_start=None):
    if x_start!=None:
        csum=[x_start]+x.copy()
    else:
        csum=x.copy()
    for i in range(1,len(csum)):
        csum[i]+=csum[i-1]
    return csum
        
```

```python jupyter={"source_hidden": false} pythonista={"disabled": false}
#interpSegments, Segments2Complex
import numpy as np
from matplotlib import pyplot as plt
from numpy import pi,exp,sign,inf

def polygonArea(p):
  def crossprod(v1,v2):
    return v1.real*v2.imag-v2.real*v1.imag
  return 0.5*np.sum(crossprod(p[range(-1,len(p)-1)],p))

def SegmentsLength(Segs):
    return sum(l for l,*_ in Segs)

def SegmentsArea(Segs):
  nSegs=len(Segs)
  dl,dang,*opts=np.array(Segs).transpose()
  ang=np.cumsum(dang)
  ang=exp(1j*np.insert( ang,0,0))
  dang_2=np.exp(1j*dang/2)
  viSeg=np.sinc(dang/(2*pi))*dl*dang_2*ang[:-1]
  pSeg=np.cumsum(viSeg)
  olderr=np.geterr()
  np.seterr(divide='ignore',invalid='ignore')#suppress the warnings from 0/0 = nan. nansum assumes nan=0, which is the correct value in this case
  area=polygonArea(pSeg) +  np.nansum((dl/dang)**2*(dang/2.0-dang_2.real*dang_2.imag))
  np.seterr(**olderr)
  return area

def InterpSegments(Segs,t,p0=0.+0.j,a0=0+1j,scale=1.0,eps=1e-6):
  """
  Segment points are calculated for values of 't', where 't' is the normalized
  length of the path. t is in the range of [0..1[
  """
  dl,dang=np.array([(l,a) for l,a,*_ in Segs]).transpose()
  L=np.cumsum(np.insert(dl,0,0.0))
  ang_=np.cumsum(np.insert(dang,0,0.0))
  ang=exp(1j*ang_)
  viSeg=np.sinc(dang/(2*pi))*dl*scale*np.exp(1j*dang/2)*ang[:-1]
  pSeg=np.cumsum(np.insert(viSeg,0,0+0j))
  if not hasattr(t,'__getitem__'): #not an array
    t=np.array([t]) #convert to array
  else:
    if t.shape==(): #no dimensions
      t=np.array([t])
    else:
      t=np.array(t)
  T=t.astype(int)
#  t=t-T
  if ((abs(pSeg[-1])<eps) and (abs(ang[-1]-(1+0j))<eps)):
    pr,ar=np.zeros((len(t),),dtype=complex), np.ones((len(t),),dtype=complex) # closed loop. No translation/rotation necessary for t>1
  else: #endpoint of path != startpoint => repeat path for t>1 by translating and rotating it
    def rotateSecant(v,beta,T):
      beta2=beta/2
      rot2=exp(1j*beta2)
      uniqueT,inverseIndex=np.unique(T,return_inverse=True) #don't re-calculate for identical values of T
      p=(v*rot2**(uniqueT-1)/nsinc(beta2/np.pi) * uniqueT * np.sinc(uniqueT*beta2/np.pi))[inverseIndex]
      a=(rot2**(2*uniqueT))[inverseIndex]
      return p,a
    pr,ar=rotateSecant(pSeg[-1]*a0,ang_[-1],T)
  pr+=p0
  ar*=a0
  l=L/L[-1]
  Xx=np.interp(t-T,l,range(len(l)))
  X=np.maximum(0,np.minimum(Xx.astype(int),len(dang)-1)) #segment index
  x=Xx-X#within seggment
  p=pSeg[X] + np.sinc( dang[X]*x /(2*pi))* dl[X]*x *scale*np.exp(1j* dang[X]*x /2)*ang[X]
  p=p*ar+pr
  a=ang[X]*np.exp(1j*dang[X]*x)*ar
  if len(p)==1:
      p=p[0] #convert array to single value if argument was a single value
      a=a[0]
  return p,a,L[-1]*t,X

def Segments2Complex(Segs,p0=0.+0.j,scale=1.0,a0=0+1j,tol=0.05,offs=0,loops=1,return_start=False):
  """
  The parameter "tol defines the resolution. It is the maximum allowable
  difference between circular arc segment, and the secant between the
  calculated points on the arc. Smaller values for tol will result in
  more points per segment.
  """
  a=a0
  p=p0
  p-=1j*a*offs
  L=0
  if return_start:
      yield p,a,L,-1 #assuming closed loop: start-point = end-point
  loopcount=0
  while (loops==None) or (loops==inf) or (loopcount<loops):
      loopcount+=1
      for X,(l,da,*_) in enumerate(Segs):
        l=l*scale
        if da!=0:
          r=l/da
          r+=offs
          if r!=0:
            l=r*da
            dl=2*abs(2*r*tol)**0.5
            n=max(int(abs(6*(da/(2*pi)))),int(l//dl)+1)
          else:
            n=1
          dda=exp(1j*da/n)
          dda2=dda**0.5
          v=(2*r*dda2.imag)*dda2*a
        else:
          n=1
          dda=1
          v=l*a
        for i in range(n):
          L+=l/n
          p+=v
          yield p,a,L,X
          v*=dda
          a*=dda
```

```python jupyter={"outputs_hidden": false, "source_hidden": false} pythonista={"disabled": false}
def arcChainInterpolator(*,arcChain,p0=0.+0.j,a0=0+1j,scale=1.0,eps=1e-6):
  """
  Segment points are calculated for values of 't', where 't' is the normalized
  length of the path. t is in the range of [0..1[
  """
  from cmath import exp as cmath_exp
  from math import sin
  def sinc(alpha):
    return 1.0 if alpha==0 else sin(alpha)/alpha
  dl=[arc[0] for arc in arcChain]
  dang=[arc[1] for arc in arcChain]  
  L=cumsum(dl,0.0)
  l=[L_/L[-1] for L_ in L]
  ang_=cumsum(dang,0.0)
  ang=[cmath_exp(1j*ang_) for ang_ in ang_]
  viSeg=[sinc(dang/2)*dl*scale*cmath_exp(1j*dang/2)*ang for dang,dl,ang in zip(dang,dl,ang[:-1])]
  pSeg=cumsum(viSeg,0+0j)
  l_idx=list(range(len(l)))
  is_closed_loop=((abs(pSeg[-1])<eps) and (abs(ang[-1]-(1+0j))<eps))
  def interpolateArcChain(t,/):
      T=int(t)
      if is_closed_loop:
          pr,ar=0.0+0.0j,1.0+0.0j
      else: #endpoint of path != startpoint => repeat path for t>1 by translating and rotating it
          v=pSeg[-1]
          beta=ang_[-1]
          beta2=beta/2
          rot2=cmath_exp(1j*beta2)
          pr=(v*rot2**(T-1)/sinc(beta2) * T * sinc(T*beta2))*a0
          ar=(rot2**(2*T))
      pr+=p0
      ar*=a0
      X,x=index_frac(t-T,l)
      p=pSeg[X] + sinc( dang[X]*x/2)* dl[X]*x *scale*cmath_exp(1j* dang[X]*x /2)*ang[X]
      p=p*ar+pr
      a=ang[X]*cmath_exp(1j*dang[X]*x)*ar
      return p,a,L[-1]*t,X
  return interpolateArcChain
```

```python jupyter={"source_hidden": true} pythonista={"disabled": false}
def ISO_thread(z=None,phi=0.0,*,Pitch,External=False):
    tan60=3**0.5
    cos60=0.5
    sin60=tan60*cos60
    Pitch2=Pitch/2
    H=(Pitch2)*tan60
    flank_start=Pitch2/8
    r_maj=flank_start/sin60
    r_maj2=r_maj**2
    c_maj=-flank_start/tan60
    r_min=2*r_maj    
    r_min2=r_min**2
    c_min=-5/8*H+Pitch2/4/tan60
    flank_end=(3/4)*Pitch2
    if External:
        def ISO_thread_(z,phi=phi,*_,**__):
            dz=abs((z-phi*Pitch+Pitch2)%Pitch-Pitch2)#use symmetries
            if dz<flank_start:
                return 0.0
            if dz<=flank_end:
                return -tan60*(dz-flank_start)
            return  (c_min-(r_min2-(Pitch2-dz)**2)**0.5)
    else:# internal thread
        def ISO_thread_(z,phi=phi,*_,**__):
            dz=abs((z-phi*Pitch+Pitch2)%Pitch-Pitch2)#use symmetries
            if dz<flank_start:
                return c_maj+(r_maj2-dz**2)**0.5
            if dz<=flank_end:
                return -tan60*(dz-flank_start)
            return -5/8*H
    return ISO_thread_(z) if z!= None else ISO_thread_

```

```python jupyter={"source_hidden": true} pythonista={"disabled": false}
def arcsFromSpec_(*,R_rim, R_hub, n_spokes, n_strands,phi_rim, r_fillet_rim, phi_hub, r_fillet_hub):
    import cmath
    from cmath import pi
  #geometry of the mesh:
    phi_tot=2*pi*n_strands
    phi_spoke2=phi_tot/(2*n_spokes)#angle be
    l1,r1,l2,r2=phi_rim*phi_spoke2*R_rim,r_fillet_rim*phi_spoke2*R_rim,phi_hub*phi_spoke2*R_hub,r_fillet_hub*phi_spoke2*R_hub
    c1=(0-1j*(R_rim-r1))*cmath.exp(1j*phi_rim*phi_spoke2)#center of the rim fillet
    c2=(0-1j*(R_hub+r2))*cmath.exp(1j*((1-phi_hub)*phi_spoke2))#center of the hub fillet
    t1,t2=calcTangent(c1,r1,c2,r2)#end points of tangent between fillet1 and fillet 2
    ltan,phitan=cmath.polar(t2-t1)
    phitan%=2*pi #counter-clockwise 0-360deg
    arcs=[(l1,phi_rim*phi_spoke2),(r1*(phitan-phi_rim*phi_spoke2),phitan-phi_rim*phi_spoke2),(ltan,0),(r2*(phitan-((1-phi_hub)*phi_spoke2)),-phitan+((1-phi_hub)*phi_spoke2)),(l2,phi_hub*phi_spoke2)]
    return arcs

```

<!-- #region jupyter={"outputs_hidden": false, "source_hidden": false} pythonista={"disabled": false} -->
### Import FullControl, colab.files
<!-- #endregion -->

```python jupyter={"source_hidden": true} pythonista={"disabled": false}
if 'google.colab' in str(get_ipython()):
  try:
    import fullcontrol as fc
  except Exception as e:
    print(e)
    print('Attempting to install missing packages. Please wait ...')
    !pip install git+https://github.com/FullControlXYZ/fullcontrol --quiet
    import fullcontrol as fc
  from google import colab
import os,sys
if os.path.exists('../site-packages') and sys.path[0]!='../site-packages':
    sys.path.insert(0,'../site-packages')
import fullcontrol as fc
from ipywidgets import widgets
preview_output=widgets.Output()#shared preview for all models because of problems with plotly custom widget
```

<!-- #region jupyter={"outputs_hidden": false, "source_hidden": false} pythonista={"disabled": false} -->
### StepGenerator()
<!-- #endregion -->

```python jupyter={"outputs_hidden": false, "source_hidden": false} pythonista={"disabled": false}
def StepGenerator(PointGenerator,*,nominal_ew,hl,nominal_print_speed,max_print_speed,fan_percent,fan_z_start=0.0,**__):
    old_eh=-1e9
    old_ew=-1e9       
    initialized=False
    fan_initialized=fan_z_start==0.0
    for x,y,z,eh,ew,*_ in PointGenerator:
      #update extrusion geometry if it has changed
      if (not fan_initialized) and (z>fan_z_start):
          yield fc.Fan(speed_percent=fan_percent)
          fan_initialized=True
      if (abs(eh-old_eh)/hl)>0.01 or abs(ew-old_ew)>0.005:
       # print(f'{zb=}{z=}{zt=}{eh=}{w=}')
        if (ew==0) or (eh==0): 
            yield fc.Extruder(on=False)
            initialized=False
        else:
            yield fc.ExtrusionGeometry(area_model='rectangle',height=eh,width=ew)
            yield fc.Printer(print_speed=min(max_print_speed,nominal_print_speed*nominal_ew/ew))#set print speed to keep extrusion rate constant
        old_eh=eh
        old_ew=ew
      if not initialized:
        yield from fc.travel_to(fc.Point(x=x,y=y))
        yield from fc.travel_to(fc.Point(z=z))
        yield fc.Extruder(on=True)
        initialized=True
      else:
        yield fc.Point(x=x,y=y,z=z)
```

<!-- #region jupyter={"outputs_hidden": false, "source_hidden": false} pythonista={"disabled": false} -->
## linear mesh
<!-- #endregion -->

```python jupyter={"source_hidden": true} pythonista={"disabled": false}
import matplotlib.pyplot as plt
import matplotlib
import numpy
import cmath  
    
%matplotlib notebook

linearData=dict(l1=0.15, r1=0.1, l2=0.15, r2=0.1,strands=5)
def update_linear_plot(l1,r1,l2,r2,strands,**kwargs):
  output1.clear_output(wait=True)
  with output1:
    %matplotlib inline
    fig=plt.figure(figsize=(3,3)) 
    ax=fig.add_subplot(1,1,1)  
    plotArcchain(ax,0+0j,1+0j,((l1,0),(r1*2*pi,2*pi)))
    c1=l1+1j*r1
    ax.plot(c1.real,c1.imag,('k+'))
    plotArcchain(ax,1+1j,-1+0j,((l2,0),(r2*2*pi,2*pi)))
    c2=(1-l2)+1j*(1-r2)
    ax.plot(c2.real,c2.imag,'k+') 
    t1,t2=calcTangent(c1,r1,c2,r2)
    ax.plot([t1.real,t2.real],[t1.imag,t2.imag])  
    ax.set_aspect('equal')
    ax.set_aspect(1.0)
    display(fig)
    plt.close()
  output2.clear_output(wait=True)
  with output2:
    fig=plt.figure(figsize=(12,3 )) 
    ax=fig.add_subplot(1,1,1)  
    ltan,phitan=cmath.polar(t2-t1)
    phitan%=2*pi
    arcs=[(l1,0),(r1*phitan,phitan),(ltan,0),(r2*phitan,-phitan),(l2,0)]
    arcs=(arcs+arcs[-1::-1])*2
    for i in range(strands):
      p0=2*i/strands+0j
      n0=1+0j
      plotArcchain(ax,p0,n0,arcs,c=['r','g','b','y','k'][i])
    ax.set_xlim(0,4)
    ax.set_aspect('equal')
    display(fig)
    plt.close()
  output3.clear_output(wait=True)
  with output3:
    print(f'{l1=:0.3f}, {r1=:0.3f}, {l2=:0.3f}, {r2=:0.3f}, {strands=:d}')
    print(f'mesh={arcs[:5]}')
  plt.close()

from ipywidgets import widgets,HBox,VBox
output1=widgets.Output() 
output2=widgets.Output()
output3=widgets.Output()
def handle_linearChange(msg):
  linearData[msg['owner'].description] = msg['new']
  update_linear_plot(**linearData)

widgetList1=dict()

for key,value in linearData.items():
    if type(value)==float:
      widgetList1[key]=widgets.FloatSlider(description=key,
            min=0.0,max=1.0,value=value,step=0.01,continuous_update=False, orientation='horizontal',layout={'width':'5in'},
            readout_format='0.3f')
    else:
      widgetList1[key]=widgets.BoundedIntText(description=key,
            min=1,max=5,value=value,step=1,continuous_update=False,layout={'width':'1.5in'},
            readout_format='d')
    widgetList1[key].observe(handle_linearChange,'value')

Layout1=VBox(
        [
        HBox([VBox([widgets.HTML(value="<h2>Parameters</h2>"),widgetList1['l1'], widgetList1['r1'],widgetList1['l2'],widgetList1['r2'],widgetList1['strands']],), output1,],),
        output2,
        output3
        ])
update_linear_plot(**linearData);
```

<!-- #region jupyter={"outputs_hidden": false, "source_hidden": false} pythonista={"disabled": false} -->
## circular mesh
<!-- #endregion -->

```python jupyter={"source_hidden": true} pythonista={"disabled": false}
from cmath import pi
circularData=dict(R_rim= 10.000, R_hub=3.500, n_spokes=13, n_strands=5,
                  phi_rim=0.160, r_fillet_rim=0.060, phi_hub=0.050, r_fillet_hub=0.200,)

deg=pi/180.0               

def update_circular_plot(*args,R_rim, R_hub, n_spokes, phi_rim, r_fillet_rim, phi_hub, r_fillet_hub, n_strands,**kwargs):#all arguments are required keyword arguments. additional arguments are allowed
  output4.clear_output(wait=True)
  phi_tot=2*pi*n_strands
  phi_spoke2=phi_tot/(2*n_spokes)
  l1,r1,l2,r2=phi_rim*phi_spoke2*R_rim,r_fillet_rim*phi_spoke2*R_rim,phi_hub*phi_spoke2*R_hub,r_fillet_hub*phi_spoke2*R_hub
  with output4:
    %matplotlib inline
    fig=plt.figure(figsize=(4,4)) 
    ax=fig.add_subplot(1,1,1)  
    if phi_spoke2>60*deg:
        ax.plot(0,0,'+')
    for phi in np.exp(1j*np.linspace(0,phi_spoke2,2)):
      p1=-1.0j*(R_hub-0.2*(R_rim-R_hub))*phi
      p2=-1.0j*(R_rim+0.2*(R_rim-R_hub))*phi
      ax.plot((p1.real,p2.real),(p1.imag,p2.imag),'k-.',lw=1)
    xlim=ax.get_xlim()
    ylim=ax.get_ylim()
    plotArcchain(ax,0-1j*R_rim,1+0j,((l1,phi_rim*phi_spoke2),(r1*2*pi,2*pi)))
    c1=(0-1j*(R_rim-r1))*exp(1j*phi_rim*phi_spoke2)
    ax.plot(c1.real,c1.imag,('k+'))
    plotArcchain(ax,(0-1j*R_hub)*exp(1j*phi_spoke2),(-1+0j)*exp(1j*phi_spoke2),((l2,-phi_hub*phi_spoke2),(r2*2*pi,2*pi)))
    c2=(0-1j*(R_hub+r2))*exp(1j*((1-phi_hub)*phi_spoke2))
    ax.plot(c2.real,c2.imag,'k+') 
    t1,t2=calcTangent(c1,r1,c2,r2)
    ax.plot([t1.real,t2.real],[t1.imag,t2.imag])  
    ltan,phitan=cmath.polar(t2-t1)
    phitan%=2*pi #counter-clockwise 0-360deg
    arcs=[(l1,phi_rim*phi_spoke2),(r1*(phitan-phi_rim*phi_spoke2),phitan-phi_rim*phi_spoke2),(ltan,0),(r2*(phitan-((1-phi_hub)*phi_spoke2)),-phitan+((1-phi_hub)*phi_spoke2)),(l2,phi_hub*phi_spoke2)]
    arcs=(arcs+arcs[-1::-1])# add mirrored arc sequence
    for i in range(n_strands):
        dphi=2*phi_spoke2/n_strands
        phi=exp(-1j*dphi*(i+1))
        plotArcchain(ax,(-1j*R_rim)*phi,(1+0j)*phi,arcs*2,c='lightgray',zorder=-1)
    if n_strands<2:
      for r in [R_hub,R_rim]:
        p=-1j*r*np.exp(1j*np.linspace(0,phi_spoke2,25))
        ax.plot(p.real,p.imag,c='lightgray',lw=1,zorder=-1)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal')
    ax.set_aspect(1.0)
    display(fig)
    plt.close()
  output5.clear_output(wait=True)
  with output5:
    fig=plt.figure(figsize=(6,6)) 
    ax=fig.add_subplot(1,1,1)  
    p0=0-1j*R_rim
    n0=1+0j
    plotArcchain(ax,p0,n0,arcs*n_spokes,c='b')
    ax.plot([xlim[0],xlim[1],xlim[1],xlim[0],xlim[0]],[ylim[0],ylim[0],ylim[1],ylim[1],ylim[0]],'--',c='gray',lw=1)
    ax.set_aspect('equal')
    display(fig)
    plt.close()
  output6.clear_output(wait=True)
  with output6:
    from math import gcd
    if  gcd(n_strands,n_spokes)!=1:
        print('\x1b[31m'+f'n_strands and n_spokes are not coprime: {gcd(n_strands,n_spokes)=} (should be 1)'+'\x1b[0m')
    print(f'{R_rim=: 0.3f}, {R_hub=:0.3f}, {n_spokes=:d}, {n_strands=:d},')
    print(f'{phi_rim=:0.3f}, {r_fillet_rim=:0.3f}, {phi_hub=:0.3f}, {r_fillet_hub=:0.3f},')
    print()
    print(f'mesh={arcs[:5]}')
    
  plt.close()

from ipywidgets import widgets,HBox,VBox
output4=widgets.Output() 
output5=widgets.Output()
output6=widgets.Output()

def handle_circularChange(msg):
  circularData[msg['owner'].description] = msg['new']
  update_circular_plot(**circularData)
    
widgetList2={}
for key,value in circularData.items():
    if type(value)==int:
        widgetList2[key]=widgets.IntText(description=key, value=value,layout={'width':'1.5in'}, readout_format='d')
    elif type(value)==float:
        widgetList2[key]=widgets.FloatText(description=key, value=value,layout={'width':'1.5in'}, readout_format='0.3f')
    else: print(f'need to add widget for {key=}, {type(value)=}');continue
    widgetList2[key].observe(handle_circularChange,'value')

Layout2=VBox(
        [HBox([
        VBox([HBox([VBox([widgetList2['R_rim'],widgetList2['R_hub'],widgetList2['n_spokes'],widgetList2['n_strands']],),VBox([widgetList2['phi_rim'], widgetList2['r_fillet_rim'],widgetList2['phi_hub'],widgetList2['r_fillet_hub'],]),],),
             output4]),
        output5,]),
        output6
        ])
update_circular_plot(**circularData);
plt.close()
```

```python jupyter={"source_hidden": true} pythonista={"disabled": false}
## transformors
```

```python jupyter={"source_hidden": true} pythonista={"disabled": false}
#transformer, rotor,...
def mesh_transformer(*,R_rim,R_hub,outline,p0o,n0o,outline_stretch=None,inline,p0i,n0i,inline_stretch=None,finline_offset=lambda *_:0.0,foutline_offset=lambda *_:0.0,**kwargs):
    from cmath import polar as cmath_polar
    unstretched_outline=cumsum([l for l,*_ in outline],0.0)
    unstretched_outline=[x/unstretched_outline[-1] for x in unstretched_outline]
    outlineInterpolator_=arcChainInterpolator(arcChain=outline,p0=p0o,a0=n0o)
    if outline_stretch!=None:
        stretched_outline=cumsum([l/s for (l,*_),s in zip(outline,outline_stretch)],0.0)
        stretched_outline=[x/stretched_outline[-1] for x in stretched_outline]
        outlineInterpolator=lambda T_mesh:outlineInterpolator_(interp(T_mesh,stretched_outline,unstretched_outline))
    else:
        outlineInterpolator=outlineInterpolator_
    unstretched_inline=cumsum([l for l,*_ in inline],0.0)
    unstretched_inline=[x/unstretched_inline[-1] for x in unstretched_inline]
    inlineInterpolator_=arcChainInterpolator(arcChain=inline,p0=p0i,a0=n0i)
    if inline_stretch!=None:
        stretched_inline=cumsum([l/s for (l,*_),s in zip(inline,inline_stretch)],0.0)
        stretched_inline=[x/stretched_inline[-1] for x in stretched_inline]
        inlineInterpolator=lambda T_mesh:inlineInterpolator_(interp(T_mesh,stretched_inline,unstretched_inline))
    else:
        inlineInterpolator=inlineInterpolator_
    def transform_mesh(pz,/):
      p=pz[0]
      #transform raw mesh to tripod shape: 
      T_mesh=(cmath_polar(p)[1]/(2*pi))%1.0 #phase angle of mesh points [0..1[
      r_mesh=abs(p)#amplitude of mesh points
      pout,aout,Lout,*_=outlineInterpolator(T_mesh)
      pout+=aout*1j*foutline_offset(pz[1],Lout)
      pin,ain,Lin,*_=inlineInterpolator(T_mesh)
      pin+=ain*1j*finline_offset(pz[1],Lin)
      p_transformed_point=pin+(pout-pin)*((r_mesh-R_hub)/(R_rim-R_hub)) #transform raw mesh to tripod shape
      pz[0]=p_transformed_point
      return pz
    return transform_mesh
        
def calibrator(*,p_center=0.0+0.0j,r_ref,dr_offset=0.0,f_offset=None,dr_rigid=0.0,mesh_compression=2.0,ew_factor=None,**kwargs):
    from cmath import polar as cmath_polar
    r_rigid_max=r_ref+dr_rigid
    r_rigid_min=r_ref-dr_rigid
    def calibrate(p_z_ew,/):
        p=p_z_ew[0]
        dr=dr_offset
        if ew_factor!=None:
            dr+=p_z_ew[2]*ew_factor
        dp=p-p_center
        r=abs(dp)
        if f_offset==None:
            r_max=r_rigid_max+dr/2+abs(dr)*(0.5+1/(mesh_compression-1))
            r_min=r_rigid_min+dr/2-abs(dr)*(0.5+1/(mesh_compression-1))
            if (r>r_max) or (r<r_min): 
                return p_z_ew#return early if point is outside the affected zone
        phi=dp/r
        if f_offset!=None:
            dr+=f_offset(p_z_ew[1],(cmath_polar(phi)[1]/(2*pi))%1,r=r_ref)
            r_max=r_rigid_max+dr/2+abs(dr)*(0.5+1/(mesh_compression-1))
            r_min=r_rigid_min+dr/2-abs(dr)*(0.5+1/(mesh_compression-1))
            if (r>r_max) or (r<r_min): 
                return p_z_ew#return early if point is outside the affected zone
        if r_rigid_min < r <r_rigid_max: #just shift the point if it is in the rigid zone
            p_z_ew[0]+=dr*phi
            return p_z_ew
        if r>r_ref:#scale the amout of shift down to zero towards r_max
            if (r_max-r_rigid_max)!=0.0: 
              x=(r_max-r)/(r_max-r_rigid_max)
              p_z_ew[0]=p_center+(r_max - x*(r_max - (dr+r_rigid_max)))*phi
            return p_z_ew
        else: #(r<=r_ref) scale the amount of shift down to zero towards r_min
            if (r_rigid_min-r_min)!=0.0: 
              x=(r-r_min)/(r_rigid_min-r_min)
              p_z_ew[0]=p_center+(r_min + x*((dr+r_rigid_min)-r_min))*phi
            return p_z_ew
    return calibrate

    
def rotor(*,center=0+0j,angle=None,phase=None,fphase=None,**kwargs):
    if fphase != None:
        def rotate(pz,/):
           pz[0]=center+(pz[0]-center)*fphase(pz[1],pz)
           return pz
    else: 
        if angle!=None:
            phase=1j**(angle*2/pi)#same as cmath.exp(1j*angle), but without cmath
        if phase == None:
            raise Exception("Argunent error in call to 'rotor': one of the arguments {phase|angle|fphase} is required") 
        def rotate(pz,/):
            pz[0]=center+(pz[0]-center)*phase
            return pz
    return rotate
    
def point_shiftor(*,p0,tol,dphi=1.0+0.0j,dang=None,dr=0.0,fblend=None,**kwargs):
    from math import pi
    if dang!=None:
        dphi=1j**(dang*2/pi)#same as cmath.exp(1j*angle), but without cmath
    if fblend != None:
        if dr!=0.0:
            def shift_point(pz,/):
               if abs(pz[0]-p0)>tol: return pz
               r0=abs(pz[0])
               x=fblend(pz[1])
               pz[0]=p0*dphi**x*(1+dr*x/r0)
               return pz
        else:
            def shift_point(pz,/):
               if abs(pz[0]-p0)>tol: return pz
               x=fblend(pz[1])
               pz[0]=p0*dphi**x
               return pz
    else:
        if dr!=0.0:
            def shift_point(pz,/):
               if abs(pz[0]-p0)>tol: return pz
               r0=abs(pz[0])
               pz[0]=p0*dphi*(1+dr/r0)
               return pz
        else:
            def shift_point(pz,/):
               if abs(pz[0]-p0)>tol: return pz
               pz[0]=p0*dphi
               return pz
                
    return shift_point
    
def terminator(stopcondition=lambda *_:False):
    def checkterm(p,/):
        for x in p:
            if stopcondition(x):
                return
            else:
                yield x
        return
    def terminate(p,/):
        if hasattr(p,'__next__'):
            return checkterm(p)
        else:
            return p
    return terminate

```

<!-- #region jupyter={"outputs_hidden": false, "source_hidden": false} pythonista={"disabled": false} -->
## Tripod 
<!-- #endregion -->

<!-- #region jupyter={"outputs_hidden": false, "source_hidden": false} pythonista={"disabled": false} -->
### Tripod()
<!-- #endregion -->

```python jupyter={"source_hidden": true} pythonista={"disabled": false}
#Tripod
tripod_parameters=dict(R_rim= 35.000, R_hub=31.000, n_spokes=63, n_strands=5,
                phi_rim=0.160, r_fillet_rim=0.060, phi_hub=0.160, r_fillet_hub=0.060,
                ew_rim=1.0,ew_fillet_rim=0.8,ew_spokes=0.7,ew_fillet_hub=0.5,ew_hub=0.5,
                L=35.000, R1=11.000, R2=4.000, R3=8.000, R4=8.000, R5=26.000,
                sL=1.000, sR1=1.500, sR2=0.450, sR3=1.000, sR4=0.700, sR5=0.800,
                z_=0.4,H=8,w_chamfer=1.0,w_chamfer1=6.5,w_chamfer1_bottom=0.3,R1_tolerance=0.2,ew_factor=0.5,
                n_skirt=3,skirt_offset=1.0,R2_thread_pitch=1.25, R2_tolerance=0.40,hl=0.2,hl_start=0.05,
                #print_parameters
                design_name = 'Tripod',
                nozzle_temp = 220.0, bed_temp = 120.0,
                nominal_print_speed = 20.0*60.0,#10*60 #print slow to give the layer time to cool
                max_print_speed = 30*60,#=speed for ew=0.5mm 
                nominal_ew = 0.75,   # extrusion width
                fan_percent = 0.0,
                fan_z_start = 0.0,
              #  nominal_eh = 0.2,    # extrusion/layer heigth
                printer_name='generic', # generic / ultimaker2plus / prusa_i3 / ender_3 / cr_10 / bambulab_x1 / toolchanger_T0
                )

tripod_parameter_descriptions={'nominal_print_speed':'print speed',
                                'nominal_ew': 'extrusion width',
                               'hl':'layer height',
                              }
   
def tripod(     z_=None,*,#keywords only from here on
                R_rim= 35.000, R_hub=31.000, n_spokes=63, n_strands=5,
                phi_rim=0.160, r_fillet_rim=0.060, phi_hub=0.160, r_fillet_hub=0.060,
                ew_rim=1.0,ew_fillet_rim=0.8,ew_spokes=0.7,ew_fillet_hub=0.5,ew_hub=0.5,
           
                L=35.000, R1=11.000, R2=4.000, R3=8.000, R4=8.000, R5=26.000,
                sL=1.000, sR1=1.500, sR2=0.450, sR3=1.000, sR4=0.700, sR5=0.800,
                H=8,w_chamfer=1.0,w_chamfer1_bottom=0.3,w_chamfer1=6.5,R1_tolerance=0.2,ew_factor=0.5,
                n_skirt=0,skirt_offset=1.0,R2_thread_pitch=1.25, R2_tolerance=0.25,hl=0.2,hl_start=0.05,
                env=None,**kwargs):
    import math
    from math import pi
    deg=pi/180
  #geometry of the mesh:
    phi_tot=2*pi*n_strands
    phi_spoke2=phi_tot/(2*n_spokes)#angle be
    l1,r1,l2,r2=phi_rim*phi_spoke2*R_rim,r_fillet_rim*phi_spoke2*R_rim,phi_hub*phi_spoke2*R_hub,r_fillet_hub*phi_spoke2*R_hub
    c1=(0-1j*(R_rim-r1))*cmath.exp(1j*phi_rim*phi_spoke2)#center of the rim fillet
    c2=(0-1j*(R_hub+r2))*cmath.exp(1j*((1-phi_hub)*phi_spoke2))#center of the hub fillet
    t1,t2=calcTangent(c1,r1,c2,r2)#end points of tangent between fillet1 and fillet 2
    ltan,phitan=cmath.polar(t2-t1)
    phitan%=2*pi #counter-clockwise 0-360deg
    arcs=[(l1,phi_rim*phi_spoke2),(r1*(phitan-phi_rim*phi_spoke2),phitan-phi_rim*phi_spoke2),(ltan,0),(r2*(phitan-((1-phi_hub)*phi_spoke2)),-phitan+((1-phi_hub)*phi_spoke2)),(l2,phi_hub*phi_spoke2)]
    l_layer=sum(l for l,*_ in arcs)*2*n_spokes #extrusion path length for one complete layer (used to calculate z-coordinate)
    arcs=(arcs+arcs[-1::-1])# add mirrored arc sequence

  #lookup table for the extrusion widths:
    arc_ew=[ew_rim,ew_fillet_rim,ew_spokes,ew_fillet_hub,ew_hub]#extrusion widths for rim...hub arc segments
    arc_ew=arc_ew+arc_ew[-1::-1]#mirror the 5 segments
    
  #outer surface of the deformed mesh:
    p0o=L+R3+0j
    n0o=0+1j
    def sss(c,a,b):          #triangle with 3 sides: return angle opposite first side 'c'
      cosgamma=(a**2+b**2-c**2)/(2*a*b)
      return math.acos(cosgamma)
    phi3=180*deg-sss(R5+R4,L,R3+R4) #angle for R3
    phi4=sss(L,R5+R4,R3+R4)         #angle for R4
    phi5=60*deg-sss(R3+R4,L,R5+R4)  #angle for R5
    outline=[(R3*phi3,phi3),(R4*phi4,-phi4),(R5*phi5,phi5)]#turtle path for 1/6 of the outline
    outline+=outline[-1::-1]#mirror to get 1/3 of the outline
    outline*=3 #repeat 3 times to get the complete closed outline turtle path
    
 #inner surface of the deformed mesh:
    p0i=L+R2+0j
    n0i=0+1j
    inline=[(R2*180*deg,180*deg),(0,-90*deg),(L-R2-R1,0),(0,-90*deg),(R1*60*deg,60*deg)]# 1/6 of inside path
    inline+=inline[-1::-1]# add mirrored path
    inline*=3 #repeat 3 times
    
 #stretch factors for inline/outline:
    outline_stretch=[sR3,sR4,sR5]
    outline_stretch+=outline_stretch[-1::-1]
    outline_stretch*=3
    inline_stretch=[sR2,1,sL,1,sR1]#there are 2 90° turns in the path which cannot be stretched
    inline_stretch+=inline_stretch[-1::-1]
    inline_stretch*=3
    
#1. generate the points of the raw annular mesh:
    blank_points=lambda:Segments2Complex(arcs,p0=R_rim+0.j,a0=0+1j,tol=0.005,return_start=True,loops=n_spokes if z_!=None else math.inf)
    def add_z_ew(blank_points,point_data=None):
        if point_data==None:
          point_data=[None]*4#NamedList(p=None,z=None,ew=None,a=None)
        for p,a,l,X in blank_points:
           # calculate the z-coordinate based on the total extrusion length, re-arrange the variables:
            if z_==None:
              z=hl*l/l_layer+hl_start
            else:
               z=z_ 
            if z>(H+hl):
                return
            point_data[:4]=p,z,arc_ew[X],a
            yield point_data
    def pz2xyz(point_data):
        p,z,ew,a,*_=point_data
        x=p.real
        y=p.imag
        z=min(z,H)
        eh=min(z if z<(hl+hl_start) else hl, H-(z-hl))
#        return(x, y, z, eh, ew)
        point_data[:5]= x, y, z, eh, ew
        return point_data
    transformations=[      
#2. rotate the mesh (helical spokes):
        rotor(fphase=lambda z,*_:1j**(z/H/n_spokes*4)),
#3. deform the annular mesh to fit between 'inline' and 'outline', and top/bottom chamfer the outside outline:
           mesh_transformer(R_rim=R_rim,R_hub=R_hub, 
                                  outline=outline,p0o=p0o,n0o=n0o,outline_stretch=outline_stretch,
                                  foutline_offset=lambda z,*_,**__:max(0,w_chamfer-z,w_chamfer-(H-z)),
                                  inline=inline,p0i=p0i,n0i=n0i,inline_stretch=inline_stretch,),
#4. counter-sink the 3 'R2' holes:
           *(calibrator(p_center=L*1j**(i/3*4),r_ref=R2,f_offset=lambda z,*_,**__:max(0,(w_chamfer-z),w_chamfer-(H-z)))for i in range(3)),
#5. tap the 3 'R2' holes (M8 ISO thread), apply tolerance and correction for extrusion with:
           *(calibrator(p_center=L*1j**(i/3*4),r_ref=R2,f_offset= ISO_thread(Pitch=R2_thread_pitch) if R2_thread_pitch else None, dr_offset=R2_tolerance, ew_factor=ew_factor)for i in range(3)),
#6. increase wall thickness around center hole: 
           lambda p_z_ew:p_z_ew if abs(p_z_ew[0])>(R1+0.001) else update_listitem(p_z_ew,2,2.0*ew_hub),#p_z_ew.update(ew=2.0*ew_hub),
#7. chamfer the center hole:
           calibrator(r_ref=R1,dr_offset=1.0*ew_hub,f_offset=lambda z,*_,**__:max(R1_tolerance,w_chamfer1_bottom-z,w_chamfer1-1.0*(H-z))),
         pz2xyz,
         ]
    transformation_generators=[(lambda tr:lambda p:map(tr,p))(transformation) for transformation in transformations] 
    def meshpoints():
        point_data=[None]*5#NamedList(p=None,z=None,ew=None,a=None)
        point_generator=add_z_ew(blank_points(),point_data)
        for transformation_generator in transformation_generators:
            point_generator=transformation_generator(point_generator)
        yield from point_generator
        return
   
               
    def skirt_and_meshpoints():
        skirt=([p[0].real,p[0].imag,hl,hl,0.6] for offs in range(n_skirt) for p in Segments2Complex(outline,p0=L+R3+0.0j,a0=0+1j,tol=0.005,offs=0.5*offs+1,return_start=True) )
        for x,y,z,eh,ew in skirt:#skip the first few points so that the start point of the skirt is not near the start point of the print.
            if y>R3:
                break
        yield from skirt
        yield [L+R3,0,hl,0.0,0.0] #move to start of print
        yield from meshpoints()
        if z_==None:
            yield [0.0,0.0,max(H+10,30),0.0,0.0] #move print head go to parking position if not layer preview
    if env!=None:
        env.update({key:value for key,value in locals().items() if not key in ['env','args','kwargs']})
    return skirt_and_meshpoints if  ((z_==None) and(n_skirt>0)) or ((z_<=hl) and (n_skirt>0)) else meshpoints
    
```

<!-- #region jupyter={"outputs_hidden": false, "source_hidden": false} pythonista={"disabled": false} -->
### update_tripod_plot()
<!-- #endregion -->

```python jupyter={"source_hidden": true} pythonista={"disabled": false}
def update_tripod_plot(*,
                       z_=None, R_rim, R_hub, n_spokes, n_strands, phi_rim, r_fillet_rim, phi_hub, r_fillet_hub, 
                       ew_rim, ew_fillet_rim, ew_spokes, ew_fillet_hub, ew_hub, L, R1, R2, R3, R4, R5, 
                       sL, sR1, sR2, sR3, sR4, sR5, H, w_chamfer, w_chamfer1_bottom, w_chamfer1, R1_tolerance, 
                       ew_factor, n_skirt, skirt_offset, R2_thread_pitch, R2_tolerance, hl, hl_start, pi, deg, p0o, n0o, sss, 
                       phi3, phi4, phi5, p0i, n0i, inline, outline_stretch, inline_stretch, skirt_and_meshpoints, 
                       arc_ew, arcs, blank_points, l_layer, math, meshpoints, outline,
                    
                       phi_spoke2,l1,r1,l2,r2,c1,c2,t1,t2,
                       **kwargs ):

  output7.clear_output(wait=True)
  phi_tot=2*pi*n_strands
#  phi_spoke2=phi_tot/(2*n_spokes)
#  l1,r1,l2,r2=phi_rim*phi_spoke2*R_rim,r_fillet_rim*phi_spoke2*R_rim,phi_hub*phi_spoke2*R_hub,r_fillet_hub*phi_spoke2*R_hub
  with output7:
    %matplotlib inline
    fig=plt.figure(figsize=(3.5,3.5)) 
    ax=fig.add_subplot(1,1,1)  
    if phi_spoke2>60*deg:
        ax.plot(0,0,'+')
    for phi in np.exp(1j*np.linspace(0,phi_spoke2,2)):
      p1=-1.0j*(R_hub-0.2*(R_rim-R_hub))*phi
      p2=-1.0j*(R_rim+0.2*(R_rim-R_hub))*phi
      ax.plot((p1.real,p2.real),(p1.imag,p2.imag),'k-.',lw=1)
    xlim=ax.get_xlim()
    ylim=ax.get_ylim()
    plotArcchain(ax,0-1j*R_rim,1+0j,((l1,phi_rim*phi_spoke2),(r1*2*pi,2*pi)))
#    c1=(0-1j*(R_rim-r1))*exp(1j*phi_rim*phi_spoke2)
    ax.plot(c1.real,c1.imag,('k+'))
    plotArcchain(ax,(0-1j*R_hub)*exp(1j*phi_spoke2),(-1+0j)*exp(1j*phi_spoke2),((l2,-phi_hub*phi_spoke2),(r2*2*pi,2*pi)))
#    c2=(0-1j*(R_hub+r2))*exp(1j*((1-phi_hub)*phi_spoke2))
    ax.plot(c2.real,c2.imag,'k+') 
#    t1,t2=calcTangent(c1,r1,c2,r2)
    ax.plot([t1.real,t2.real],[t1.imag,t2.imag])  
    ltan,phitan=cmath.polar(t2-t1)
    phitan%=2*pi #counter-clockwise 0-360deg
    arcs=[(l1,phi_rim*phi_spoke2),(r1*(phitan-phi_rim*phi_spoke2),phitan-phi_rim*phi_spoke2),(ltan,0),(r2*(phitan-((1-phi_hub)*phi_spoke2)),-phitan+((1-phi_hub)*phi_spoke2)),(l2,phi_hub*phi_spoke2)]
    arcs=(arcs+arcs[-1::-1])# add mirrored arc sequence
    for i in range(n_strands):
        dphi=2*phi_spoke2/n_strands
        phi=exp(-1j*dphi*(i+1))
        plotArcchain(ax,(-1j*R_rim)*phi,(1+0j)*phi,arcs*2,c='lightgray',zorder=-1)
    if n_strands<2:
      for r in [R_hub,R_rim]:
        p=-1j*r*np.exp(1j*np.linspace(0,phi_spoke2,25))
        ax.plot(p.real,p.imag,c='lightgray',lw=1,zorder=-1)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal')
    ax.set_aspect(1.0)
    display(fig)
    plt.close()
  output8.clear_output(wait=True)
  with output8:
    fig=plt.figure(figsize=(6,6)) 
    ax=fig.add_subplot(1,1,1) 
    plotArcchain(ax,p0o,n0o,outline,'b',lw=2,zorder=2)
    plotArcchain(ax,p0i,n0i,inline,'b',lw=2,zorder=2)
    ax.plot(0,0,'b',lw=2,label='mesh outline')
    p_arrow=R1*cmath.exp(1j*45*deg)
    for text,c,r,offs,ang in [(r'$R_1$',0+0j,R1,-7.5,150*deg),(r'$R_2$',L*cmath.exp(1j*0*deg),R2,-9.5,45*deg),(r'$R_3$',L+0j,R3,8.0,60*deg),
                              (r'$R_4$',L+(R3+R4)*cmath.exp(1j*phi3),-R4,8.0,60*deg),(r'$R_5$',0+0j,R5,8.0,50*deg),
                              (r'$L$',L/2-10j,L/2,-L/2,0*deg),(r'$L$',L/2-10j,L/2,-L/2,180*deg)]:
        phi=cmath.exp(1j*ang)
        ptip=c+r*phi 
        ptext=c+(r+offs)*phi
        ax.annotate(text,xy=(ptip.real,ptip.imag),xycoords='data',
            xytext=(ptext.real, ptext.imag), textcoords='data',arrowprops=dict(facecolor='black', headwidth=6, headlength=10, width=0.25, shrink=0.0),
            horizontalalignment='center', verticalalignment='center',fontsize=14,backgroundcolor='w')
    for p1,p2 in [(0+5j,0-15j),(-5,5),(L+5j,L-15j),(L-5,L+5)]:
        ax.plot((p1.real,p2.real),(p1.imag,p2.imag),'k-.',lw=1) 
    p_mesh=list(zip(*((p.real,p.imag) for p,*_ in Segments2Complex(arcs,p0=0.0-1j*R_rim,a0=1.0+0.0j,tol=0.01,return_start=True,loops=n_spokes))))
    ax.plot(p_mesh[0],p_mesh[1],'-',c='lightgray',label='raw mesh',zorder=-1) 
    ax.plot([xlim[0],xlim[1],xlim[1],xlim[0],xlim[0]],[ylim[0],ylim[0],ylim[1],ylim[1],ylim[0]],'--',c='gray',lw=1,label='mesh detail')
    transformed_mesh_points=list(zip(*[p.copy() for p in meshpoints()]))
    ax.plot(transformed_mesh_points[0],transformed_mesh_points[1],'-',c='g',label='transformed mesh',zorder=1)
    if (n_skirt>0) and (z_<=hl):
      skirt=list(zip(*((p[0].real,p[0].imag) for offs in range(n_skirt) for p in Segments2Complex(outline,p0=L+R3+0.0j,a0=0+1j,tol=0.005,offs=0.5*offs+1,return_start=True) )))
      ax.plot(skirt[0],skirt[1],'c-',label='skirt')

# NEMA 17 outline
    A,B,C,D,E,F=42.2,22,31,4,3,5
    p0=A/2+0j
    n0=0+1j
    NEMA=[(A/2-D,0),(D*45*deg,45*deg)]
    plotArcchain(ax,p0,n0,(NEMA+NEMA[-1::-1])*4,'--',c='gray',zorder=-1)
    ax.plot(0,0,'--',c='gray',label='NEMA 17 motor')
    n0=0+1j
    for p0,r in [(F/2+0j,F/2),(B/2+0j,B/2)]+[(C/2*(1+1j)*1j**i+E/2,E/2) for i in range(4)]:
      plotArc(ax,p0,n0,r*360*deg,360*deg,'--',c='gray',zorder=-1)
    ax.legend(loc='lower right')
    ax.set_aspect('equal')
    display(fig)
    plt.close()
  output9.clear_output(wait=True)
  with output9:
#    print(f'{R3+R4=}, {L=}, {R5+R4=}, {sss(R3+R3,L,R5+R4)/deg=}, {phi3/deg=},{phi4/deg=}, {phi5/deg=}, {R1=}')
#    print(f'{len(skirt)=}, {min(skirt[0])=:.2f}, {max(skirt[0])=:.2f}')
#    print(f'{transformed_mesh_points[0][0]=},{transformed_mesh_points[1][0]=}')
    from math import gcd
    if  gcd(n_strands,n_spokes)!=1:
        print('\x1b[31m'+f'n_strands and n_spokes are not coprime: {gcd(n_strands,n_spokes)=} (should be 1)'+'\x1b[0m')
    print(f'{R_rim=: 0.3f}, {R_hub=:0.3f}, {n_spokes=:d}, {n_strands=:d},')
    print(f'{phi_rim=:0.3f}, {r_fillet_rim=:0.3f}, {phi_hub=:0.3f}, {r_fillet_hub=:0.3f},')
    print(f'{L=:0.3f}, {R1=:0.3f}, {R2=:0.3f}, {R3=:0.3f}, {R4=:0.3f}, {R5=:0.3f},')
    print(f'{sL=:0.3f}, {sR1=:0.3f}, {sR2=:0.3f}, {sR3=:0.3f}, {sR4=:0.3f}, {sR5=:0.3f},')
    print()
    print(f'mesh={arcs[:5]}')
    print()
    
  plt.close()

from ipywidgets import widgets,HBox,VBox
output7=widgets.Output() 
output8=widgets.Output()
output9=widgets.Output()


def handle_change_tripod(msg):
  key,value=msg['owner'].key,msg['new']
  tripod_parameters[key] =  value
  if key=='H':
      widgetList3['z_'].max=value
  env={}
  tripod(**tripod_parameters,env=env)
  preview_output.clear_output(wait=False) #preview may not be valid, since parameters have changed
  update_tripod_plot(**env)
    
def save_tripod_gcode(*args,**kwargs):
    from time import perf_counter
    from datetime import datetime
    filename=tripod_parameters['design_name']+datetime.now().strftime("__%d-%m-%Y__%H-%M-%S")
    tripod_status_widget.value=" calculating extrusion path ..."
    t0=perf_counter()
    myTripod=tripod(**(tripod_parameters|dict(z_=None))) 
    myTripodPoints=list(p.copy() for p in myTripod())
    t1=perf_counter()
    tripod_status_widget.value+=f"({t1-t0:0.3f}s), calculating steps ..."
    steps=list(StepGenerator(myTripodPoints,**tripod_parameters))
    t2=perf_counter()
    xmax,ymax=xmin,ymin=0,0
    for x,y in ((point.x,point.y) for point in steps if type(point)==fc.Point):
        if x!= None: xmin=min(xmin,x)
        if x!= None: xmax=max(xmax,x)
        if y!= None: ymin=min(ymin,y)
        if y!= None: ymax=max(ymax,y)
    
    model_offset = fc.Vector(x=100-0.5*(xmax+xmin), y=100-0.5*(ymax+ymin), z=0.0)
    steps = fc.move(steps, model_offset)
    gcode_controls = fc.GcodeControls(
                    printer_name=tripod_parameters['printer_name'],
                    save_as=filename,
                    include_date=False,
                    initialization_data={
                        'primer': 'no_primer',
                         }|tripod_parameters|({'fan_percent':0.0} if tripod_parameters.get('fan_z_start',0.0)!=0.0 else {})
                    )
    tripod_status_widget.value=' saving gcode to "'+filename+'.gcode" ...'
    fc.transform(steps, 'gcode', gcode_controls)
    if "colab" in globals():
        tripod_status_widget.value=' preparing file for download ...'
        colab.files.download(filename+".gcode")
        tripod_status_widget.value=''
    else:
        tripod_status_widget.value+=' File saved to disk! Download manually.'
 
        
def update_tripod_preview(*args,**kwargs):
    from time import perf_counter
    import os
    preview_output.clear_output(wait=True)
    with preview_output:
        tripod_status_widget.value=" calculating extrusion path ..."
        t0=perf_counter()
        myTripod=tripod(**(tripod_parameters|dict(z_=None))) 
        myTripodPoints=list(p.copy() for p in myTripod())
        t1=perf_counter()
        tripod_status_widget.value+=f"({t1-t0:0.3f}s), generating steps ..."
        steps=list(StepGenerator(myTripodPoints,**tripod_parameters))
        t2=perf_counter()
        tripod_status_widget.value+=f"({t2-t1:0.3f}s), running fc.transform ..."
        xmax,ymax=xmin,ymin=0,0
        for x,y in ((point.x,point.y) for point in steps if type(point)==fc.Point):
            if x!= None: xmin=min(xmin,x)
            if x!= None: xmax=max(xmax,x)
            if y!= None: ymin=min(ymin,y)
            if y!= None: ymax=max(ymax,y)
        model_offset = fc.Vector(x=100-0.5*(xmax+xmin), y=100-0.5*(ymax+ymin), z=0.0)
        steps = fc.move(steps, model_offset)
        style='tube'
        if ('iPad' in os.uname().machine):
            plt.close()
            fc.transform(steps, 'plot', fc.PlotControls(style='line',color_type='print_sequence'))
            _=plt.show()
        else:
            _=fc.transform(steps, 'plot', fc.PlotControls(style=style ,color_type='print_sequence'))
        tripod_status_widget.value=''
            
style={'description_width':'1.0in'}
layout={'width':'1.7in'}
widgetList3={}
for key,value in tripod_parameters.items():
    if type(value)==int:
        widgetList3[key]=widgets.IntText(description=key, value=value,style=style,layout=layout, readout_format='d')
    elif type(value)==float:
        widgetList3[key]=widgets.FloatText(description=tripod_parameter_descriptions.get(key,key), value=value,style=style,layout=layout, readout_format='0.3f')
    elif type(value)==str:
        widgetList3[key]=widgets.Text(description=key, value=value,style=style,layout=layout,)
    elif type(value)==bool:
        widgetList3[key]=widgets.Dropdown(description=key,options=[True,False], value=value,style=style,layout=layout,indent=True)
    else: print(f'need to add widget for {key=}, {type(value)=}');continue
        
key='z_'
widgetList3[key]=widgets.FloatSlider(description=key,
            min=0.0,max=tripod_parameters['H'],value=tripod_parameters[key],step=0.2,continuous_update=True, orientation='vertical',   
           readout_format='0.1f',layout={'height':'4.5in'})
for key in tripod_parameters: 
    widgetList3[key].observe(handle_change_tripod,'value')      
    widgetList3[key].key=key

SaveTripodGcodeButton=widgets.Button(description='download G-Code',on_click=save_tripod_gcode,button_style='primary')
SaveTripodGcodeButton.on_click(save_tripod_gcode)
UpdateTripodPreviewButton=widgets.Button(description='update preview',on_click=update_tripod_preview)
UpdateTripodPreviewButton.on_click(update_tripod_preview)
tripod_status_widget=widgets.Label()
widgetList3['design_name'].layout.width='3.0in'
widgetList3['printer_name'].layout.width='3.0in'
Layout3=widgets.Tab([VBox(
        [
        HBox([
            VBox([
                widgets.HTML(value='<h2 align=left>Mesh Parameters: </h2>'),
                HBox([widgetList3['n_spokes'],widgetList3['n_strands']]),
                HBox([widgetList3['R_hub'],widgetList3['phi_hub'],widgetList3['r_fillet_hub'],],),
                HBox([widgetList3['R_rim'],widgetList3['phi_rim'],widgetList3['r_fillet_rim'],]),
                ]),
            output7,
            ],layout={'border': '1px solid black'}),              
        HBox([ 
          VBox([
              widgets.HTML(value='<h2 align=left>Geometry: </h2>'),
              HBox([widgets.HTML(value='<h4 align=center>Dimensions: </h4>',layout=dict(width='50%')),widgets.HTML(value='<h4 align=center>Mesh-Stretch: </h3>',layout=dict(width='50%')), ],layout=dict(width='100%')),
              HBox([widgetList3['L'], widgetList3['sL'], ]),
              HBox([widgetList3['R1'],widgetList3['sR1'],]),
              HBox([widgetList3['R2'],widgetList3['sR2'],]),
              HBox([widgetList3['R3'],widgetList3['sR3'],]),
              HBox([widgetList3['R4'],widgetList3['sR4'],]),
              HBox([widgetList3['R5'],widgetList3['sR5'],]),
              HBox([widgetList3['H'],]),
              HBox([widgetList3['R2_thread_pitch'], widgetList3['R2_tolerance'],]),
              HBox([widgetList3['w_chamfer'],widgetList3['w_chamfer1'],]),
              HBox([widgetList3['n_skirt'],widgetList3 ['skirt_offset'],]),
              ]),
          HBox([output8,widgetList3['z_']]), 
          ],layout={'border': '1px solid black'}),
        VBox([
 #          widgets.HTML(value='<h2 align=left>Output: </h2>'), #Title for output box
           output9,
            ],layout={'border': '1px solid black'})
        ]),
        HBox([             
             VBox([
               widgets.HTML(value='<h2 align=left>Print Parameters: </h2>'),
               HBox([widgetList3['hl'],UpdateTripodPreviewButton,tripod_status_widget],)
               ]),
           #  tripod_preview_output, #plotly output not showing inside HBox
             ],layout={'border': '1px solid black'}),
        VBox([
          widgets.HTML(value='<h2 align=left>Print Parameters: </h2>'),
          HBox([widgetList3['hl'],widgetList3['nominal_ew'],]),
          HBox([widgetList3['nominal_print_speed'],]),
          HBox([widgetList3['nozzle_temp'],widgetList3['bed_temp'],]),
          HBox([widgetList3['fan_percent'],widgetList3['fan_z_start'],]),
          HBox([widgetList3['printer_name'],]),
          HBox([widgetList3['design_name'],],),
          HBox([SaveTripodGcodeButton,tripod_status_widget],)
          ],layout={'border': '1px solid black'}),
      ],titles=['Design','Preview','G-Code'])

env={}
tripod(**tripod_parameters,env=env)
update_tripod_plot(**env)
output9.clear_output(wait=False)#make form smaller 
plt.close()
```

<!-- #region jupyter={"outputs_hidden": false, "source_hidden": false} pythonista={"disabled": false} -->
## Capstan
<!-- #endregion -->

<!-- #region jupyter={"outputs_hidden": false, "source_hidden": false} pythonista={"disabled": false} -->
### alpha_blend(), groove_depth_pattern()
<!-- #endregion -->

```python editable=true jupyter={"source_hidden": true} pythonista={"disabled": false} slideshow={"slide_type": ""}
#groove toolpath
from math import cos,pi
def alpha_blend(x,xstart, xend,f0,f1):
    alpha=0.5+0.5*cos(max(0.0,min(1,(x-xstart)/(xend-xstart)))*pi)
    return [alpha*y0+(1.0-alpha)*y1 for y0,y1 in zip(f0(x),f1(x))]
    
def groove_depth_pattern_factory(*,delta_phi_ramp_start, delta_phi_chamfer_transition, phi_center, delta_phi_ramp,delta_phi_helix,
                                r_chamfer, z_bottom, dz_chamfer, r_circ, r_cable_ramp, z_top, r_cable, 
                                z_helix_start, groove_pitch, phi_helix_start, h_rim, groove_flank_angle,
                                R_rim,env=None,**_):
    from math import sin,cos,tan,pi
    deg=pi/180
    phi_0=delta_phi_ramp_start-delta_phi_chamfer_transition-phi_center
    phi_1=phi_0+delta_phi_chamfer_transition
    phi_2=phi_1+delta_phi_ramp
    phi_3=phi_2+delta_phi_helix
    phi_4=phi_3+delta_phi_ramp
    phi_5=phi_4+delta_phi_chamfer_transition
    #tool paths:
    #variable input parameters
    def groove_toolpath(phi):
        def bottom_chamfer(phi):  return [ r_chamfer,         z_bottom-dz_chamfer ]
        def bottom_cable(phi):    return [ r_cable_ramp,  z_bottom-r_cable ]
        def helix(phi):           return [ r_circ,            (phi-phi_2)/(2*pi)*groove_pitch+h_rim+r_cable ]
        def top_cable(phi):       return [ r_cable_ramp,  z_top+r_cable ]
        def top_chamfer(phi):     return [ r_chamfer,         z_top+dz_chamfer ]
    
        if    phi<phi_0              : return  bottom_chamfer(phi)
        elif (phi>=phi_0)&(phi<phi_1): return  alpha_blend(phi,phi_0,phi_1,bottom_chamfer,bottom_cable)
        elif (phi>=phi_1)&(phi<phi_2): return  alpha_blend(phi,phi_1,phi_2,bottom_cable,helix)
        elif (phi>=phi_2)&(phi<phi_3): return  helix(phi)
        elif (phi>=phi_3)&(phi<phi_4): return  alpha_blend(phi,phi_3,phi_4,helix,top_cable)
        elif (phi>=phi_4)&(phi<phi_5): return  alpha_blend(phi,phi_4,phi_5,top_cable,top_chamfer)
        elif  phi>=phi_5             : return  top_chamfer(phi)
        else: raise(Exception('This line should never be reached.'))
            
    def groove_tool(dz,r=0.45,a=30*deg):
        flank=abs(dz)>r*cos(a)
        if flank:
          result=abs(dz)/tan(a)-r/sin(a)
        else:
          result=-(r**2 - dz**2)**0.5
        return result
    
    def phi_groove(z):
        return (z-z_helix_start)/groove_pitch*(2*pi)+phi_helix_start
    
    def thread_depth_pattern(z,phi,z0=0.0):
        i_groove=(phi_groove(z)-pi)//(2*pi)
        phi_tool_1=(phi)%(2*pi)+i_groove*(2*pi)
        r_tool_1,z_tool_1=groove_toolpath(phi_tool_1)
        dz_1=z_tool_1-z
        if abs(dz_1)<=(groove_pitch/2):
            return r_tool_1+groove_tool(dz=dz_1,r=r_cable,a=groove_flank_angle)
        phi_tool_2=phi_tool_1+2*pi
        r_tool_2,z_tool_2=groove_toolpath(phi_tool_2)
        dz_2=z_tool_2-z
        if abs(dz_2)<=(groove_pitch/2):
            return r_tool_2+groove_tool(dz=dz_2,r=r_cable,a=groove_flank_angle)
        while dz_1>0:
            phi_tool_2,r_tool_2,z_tool_2,dz_2=phi_tool_1,r_tool_1,z_tool_1,dz_1
            phi_tool_1=phi_tool_1-2*pi
            r_tool_1,z_tool_1=groove_toolpath(phi_tool_1)
            dz_1=z_tool_1-z
        while dz_2<0:
            phi_tool_1,r_tool_1,z_tool_1,dz_1=phi_tool_2,r_tool_2,z_tool_2,dz_2
            phi_tool_2=phi_tool_2+2*pi
            r_tool_2,z_tool_2=groove_toolpath(phi_tool_2)
            dz_2=z_tool_2-z
        r_2=r_tool_2+groove_tool(dz=dz_2,r=r_cable,a=groove_flank_angle)
        r_1=r_tool_1+groove_tool(dz=dz_1,r=r_cable,a=groove_flank_angle)
        return min(r_1,r_2,R_rim)
    if env!=None:
        env.update({key:value for key,value in locals().items() if not key in ['env','args','kwargs']})
    return thread_depth_pattern
```

<!-- #region jupyter={"outputs_hidden": false, "source_hidden": false} pythonista={"disabled": false} -->
### cpastan()
<!-- #endregion -->

```python jupyter={"outputs_hidden": false, "source_hidden": false} pythonista={"disabled": false}
capstan_parameters=dict(n_spokes=13, n_strands=5,
                phi_rim=0.160, r_fillet_rim=0.060, phi_hub=0.10, r_fillet_hub=0.160,
                ew_rim=1.0,ew_fillet_rim=0.8,ew_spokes=0.7,ew_fillet_hub=0.5,ew_hub=0.5,
                spoke_midpoint=0.38,mesh_twist_pitch = -60,
                l_turn= 60.0, l_tot= 420.0, d_cable=0.9, groove_pitch=1.25, left_handed=False, n_cable_tunnels=2,
                tunnel_pos=0.5,
                hub_squeezeout_factor=2.,shaft_type='D', d_shaft=5.0,D_key=0.5, shaft_tolerance=0.15, countersink_chamfer=0.75,
                z_=3.2,
                n_skirt=3,skirt_offset=1.0,hl=0.2,hl_start=0.05,
                #print_parameters
                design_name = 'Capstan',
                nozzle_temp = 220.0, bed_temp = 120.0,
                nominal_print_speed = 10.0*60.0,#10*60 #print slow to give the layer time to cool
                max_print_speed = 15*60,#=speed for ew=0.5mm 
                nominal_ew = 0.75,   # extrusion width
                fan_percent = 100.0,
                fan_z_start = 1.0,
                #  nominal_eh = 0.2,    # extrusion/layer heigth
                printer_name='generic', # generic / ultimaker2plus / prusa_i3 / ender_3 / cr_10 / bambulab_x1 / toolchanger_T0
                )

capstan_parameter_descriptions={'l_turn':'circumference',
                                'l_tot': 'cable groove length',
                                'd_shaft':'shaft diameter',
                                'n_cable_tunnels':'cable tunnels',
                                'hl':'layer height',
                                'nominal_print_speed':'print speed',
                                'nominal_ew':'extrusion width'}

         

def capstan(n_spokes=13, n_strands=5,
                  phi_rim=0.160, r_fillet_rim=0.060, phi_hub=0.160, r_fillet_hub=0.060,
                  ew_rim=1.0,ew_fillet_rim=0.8,ew_spokes=0.7,ew_fillet_hub=0.5,ew_hub=0.5,
                  spoke_midpoint=0.38,mesh_twist_pitch = -60.0,
                  l_turn= 60.0, l_tot= 600.0, d_cable=0.9, groove_pitch=1.25, left_handed=False, n_cable_tunnels=2,
                  tunnel_pos=0.5,
                  hub_squeezeout_factor=2,shaft_type='D', d_shaft=5.0,D_key=0.5, shaft_tolerance=0.0, countersink_chamfer=0.75,
                  z_=5.0,
                  n_skirt=3,skirt_offset=1.0,hl=0.2,hl_start=0.05,
           env=None,**kwargs):
  import math
  import cmath
  from cmath import polar as cmath_polar
  from math import pi
  deg=pi/180.0       
      #calculate dependent parameters
  r_cable=d_cable/2
  r_circ=(l_turn**2-groove_pitch**2)**0.5/(2*pi)
  r_shaft=d_shaft/2
  h_rim=2.0*d_cable
  R_rim=r_circ+r_cable
  r_cable_ramp=r_circ-d_cable-ew_rim
  phi_offset_channel2=0 if n_cable_tunnels!=2 else (n_spokes//2)/n_spokes*2*pi
  dz_chamfer=1.75*d_cable#vertical tool position above/below top/bottom
  r_chamfer=r_cable_ramp#horizontal tool position
  groove_flank_angle=30*deg
  delta_phi_ramp_start=1.5*(2*pi)/n_spokes #offset relative to return channel center
  delta_phi_ramp=pi/2 #
  delta_phi_chamfer_transition=pi/16
  dz_ends=2*(h_rim+r_cable)
  phi_ends=2*(delta_phi_ramp_start+delta_phi_ramp)+phi_offset_channel2
  delta_phi_helix_=(l_tot/l_turn)*2*pi#preliminary
  n_rot=(dz_ends*(2*pi)+ delta_phi_helix_*groove_pitch - mesh_twist_pitch*(phi_ends + delta_phi_helix_))/(-mesh_twist_pitch*(2*pi))
  n_rot=int(n_rot+1)
  delta_phi_helix=(-dz_ends*(2*pi) - n_rot*mesh_twist_pitch*2*pi + phi_ends*mesh_twist_pitch)/(groove_pitch -mesh_twist_pitch)
  z_bottom=0.0
  z_top=z_bottom+delta_phi_helix/(2*pi)*groove_pitch+dz_ends
  z_helix_start=z_bottom+h_rim+r_cable
  phi_helix_start=delta_phi_ramp_start+delta_phi_ramp
  phi_center=0.5*2*pi/n_spokes
#  phi_width=2*pi/n_spokes
#  max_phase_advance=0.7*2*pi/n_spokes
    
  phi_tot=2*pi*n_strands
  phi_spoke2=phi_tot/(2*n_spokes)
    
  R_hub=r_shaft
  l1,r1,l2,r2=phi_rim*phi_spoke2*R_rim,r_fillet_rim*phi_spoke2*R_rim,phi_hub*phi_spoke2*R_hub,r_fillet_hub*phi_spoke2*R_hub
  c1=(0-1j*(R_rim-r1))*cmath.exp(1j*phi_rim*phi_spoke2)
  c2=(0-1j*(R_hub+r2))*cmath.exp(1j*((1-phi_hub)*phi_spoke2))
  t1,t2=calcTangent(c1,r1,c2,r2)
  ltan,phitan=cmath.polar(t2-t1)
  phitan%=2*pi #counter-clockwise 0-360deg
  arcs=[(l1,phi_rim*phi_spoke2),(r1*(phitan-phi_rim*phi_spoke2),phitan-phi_rim*phi_spoke2),(ltan*(1-spoke_midpoint),0),(ltan*spoke_midpoint,0),(r2*(phitan-((1-phi_hub)*phi_spoke2)),-phitan+((1-phi_hub)*phi_spoke2)),(l2,phi_hub*phi_spoke2)]
  l_layer=sum(l for l,*_ in arcs)*2*n_spokes #extrusion path length for one complete layer (used to calculate z-coordinate)
  arcs=(arcs+arcs[-1::-1])# add mirrored arc sequence
  arc_ew=[ew_rim,ew_fillet_rim,ew_spokes,0.5*ew_spokes+0.5*ew_fillet_hub,ew_fillet_hub,ew_hub]#extrusion widths for rim...hub arc segments
  arc_ew=arc_ew+arc_ew[-1::-1]
  p_spokes_mid=list(Segments2Complex(arcs[:3],p0=R_rim+0.j,a0=0+1j,tol=0.01))[-1][0]
  r_spokes_mid=abs(p_spokes_mid)
  R_tunnel=R_hub+tunnel_pos*(R_rim-R_hub)
  r_lead_in=h_rim+d_cable
  outline=[(R_rim*2*pi,2*pi),]
  p0o=R_rim+0.0j
  n0o=0.0+1.0j
  if shaft_type.upper()=='O':
    DshaftOutline=[(r_shaft*2*pi,2*pi)]#'plain' shaft outline is a circle
  else:
    Dkeycp = (r_shaft**2-(r_shaft-D_key)**2)**0.5 + 1j*(r_shaft-D_key)#corner point of D-shaft (x+ iy)
    Dkeyang=cmath_polar(Dkeycp)[1] # angle up to the corner point
    DshaftOutline=[(r_shaft*Dkeyang,Dkeyang),(0,pi/2-Dkeyang),(Dkeycp.real,0)]#1/4 of D-shaft outline
    DshaftOutline=DshaftOutline+DshaftOutline[-1::-1]#add mirror image -> upper half of D-shaft outlone
    DshaftOutline+=DshaftOutline if shaft_type.upper()=='DD' else [(r_shaft*pi,pi)] #add lower half of D-shaft outline
  inline_offset=0.5*ew_hub*hub_squeezeout_factor+shaft_tolerance/2
  inline=[(l+a*inline_offset,a) for l,a in DshaftOutline]
  p0i=R_hub+inline_offset+0.0j
  n0i=0.0+1.0j
#1. generate the points of the raw annular mesh:
  blank_points=lambda:Segments2Complex(arcs,p0=R_rim+0.j,a0=0+1j,tol=0.003,return_start=True,loops=n_spokes if z_!=None else math.inf)
  thread_depth_pattern=groove_depth_pattern_factory(**locals())
  transformations=[
      #2a. shift spokes midpoint radially to make room for cable tunnel
            calibrator(r_ref=r_spokes_mid,dr_offset=R_tunnel-r_spokes_mid),
      #2b. shift 1 spoke mid point to round over the cable tunnel in/outlet  
           point_shiftor(p0=R_tunnel,tol=(R_rim-R_hub)/10,dang=0.6*2*pi/n_spokes,dr=0.4*(R_rim-R_tunnel),
                fblend=lambda z:(1.0-(r_lead_in**2-min(max(r_lead_in-(z-z_bottom),0.0),r_lead_in)**2)**0.5/r_lead_in),),
      #2c. shift 1 spoke mid point to round over the cable tunnel in/outlet  
           point_shiftor(p0=R_tunnel*cmath.exp(1j*(-2*phi_center-phi_offset_channel2)),tol=(R_rim-R_hub)/10,dang=-0.6*2*pi/n_spokes,dr=0.4*(R_rim-R_tunnel),
                fblend=lambda z:(1.0-(r_lead_in**2-min(max(r_lead_in-(z_top-z),0.0),r_lead_in)**2)**0.5/r_lead_in),),
      #3. counterssink
           calibrator(r_ref=R_hub,f_offset=lambda z,*_,**__:max(0,countersink_chamfer-(z-z_bottom),countersink_chamfer-1.0*(z_top-z)),ew_factor=0.0),
      #4. rotate the mesh (helical spokes):
           rotor(fphase=lambda z,*_:1j**(z/mesh_twist_pitch*4)),
      #5,6. deform the annular mesh to fit between 'inline' (shaft) and 'outline' (cable groove, and top/bottom chamfer):
           mesh_transformer(R_rim=R_rim,R_hub=R_hub, 
                                  outline=outline,p0o=p0o,n0o=n0o,
                                  foutline_offset=(lambda z,L:R_rim-thread_depth_pattern(z,L/R_rim)+ew_rim/2),#lambda z,*_,**__:0.0,#max(0,w_chamfer-z,w_chamfer-(h-z)),
                                  inline=inline,p0i=p0i,n0i=n0i,),
          
                 ]   
  
  transformation_generators=[(lambda tr:lambda p:map(tr,p))(transformation) for transformation in transformations] 
  def meshpoints():
        def lX2zew(blank_points,point_data=None):
            if point_data==None:
              point_data=[None]*4#NamedList(p=None,z=None,ew=None,a=None)
            for p,a,l,X,*_ in blank_points:
               # calculate the z-coordinate based on the total extrusion length, re-arrange the variables:
                if z_==None:
                  z=hl*l/l_layer+hl_start
                else:
                   z=z_ 
                point_data[:4]=p,z,arc_ew[X],a
                yield point_data

        def pz2xyz(point_generator,xyz_data=None):
            if xyz_data==None:
              xyz_data=[None]*5 # reused buffer
            for p,z,ew,a,*_ in point_generator:
                x=p.real
                y=p.imag
                if left_handed: y*=-1
                z=min(z,z_top)
                eh=min(z if z<(hl+hl_start) else hl, z_top-(z-hl))
                xyz_data[:5]= x, y, z, eh, ew
                yield xyz_data

        point_generator=blank_points()
        point_generator=lX2zew(point_generator)#calculate z-coordinate based on total extrusion length and layer height
        point_generator=takewhile(lambda pzew:pzew[1]<=(z_top+hl),point_generator)#stop generating points once top is reached
        for transformation_generator in transformation_generators:
            point_generator=transformation_generator(point_generator)
        point_generator=pz2xyz(point_generator)
        yield from point_generator
        return
            
  def skirt_and_meshpoints():
        skirt=([p[0].real,p[0].imag,hl,hl,0.6] for offs in range(n_skirt) for p in Segments2Complex(outline,p0=R_rim+0.0j,a0=0+1j,tol=0.005,offs=0.5*offs+skirt_offset,return_start=True) )
        for x,y,z,eh,ew in skirt:#skip the first few points so that the start point of the skirt is not near the start point of the print.
            if y>R_rim/4:
                break
        yield from skirt
        yield [R_rim,0,hl,0.0,0.0] #move to start of print
        yield from meshpoints()
        if z_==None:
            yield [0.0,0.0,max(z_top+10,30),0.0,0.0] #move print head go to parking position if not layer preview
  capstan_point_factory=skirt_and_meshpoints if  ((z_==None) and(n_skirt>0)) or ((z_<=hl) and (n_skirt>0)) else meshpoints
  if env!=None:
        env.update({key:value for key,value in locals().items() if not key in ['env','args','kwargs']})
  return capstan_point_factory


```

<!-- #region jupyter={"outputs_hidden": false, "source_hidden": false} pythonista={"disabled": false} -->
### update_capstan_plot()
<!-- #endregion -->

```python jupyter={"outputs_hidden": false, "source_hidden": false} pythonista={"disabled": false}
def update_capstan_plot(*,n_spokes, n_strands, phi_rim, r_fillet_rim, phi_hub, r_fillet_hub, ew_rim,
                         ew_fillet_rim, ew_spokes, ew_fillet_hub, ew_hub, spoke_midpoint, mesh_twist_pitch,
                         l_turn, l_tot, d_cable, groove_pitch, left_handed, n_cable_tunnels, tunnel_pos, 
                         hub_squeezeout_factor, shaft_type, d_shaft, D_key, shaft_tolerance, countersink_chamfer, z_, n_skirt, 
                         skirt_offset, hl, hl_start, r_cable, r_circ, r_shaft, h_rim, r_cable_ramp, 
                         phi_offset_channel2, dz_chamfer, r_chamfer, groove_flank_angle, delta_phi_ramp_start,
                         delta_phi_ramp, delta_phi_chamfer_transition, dz_ends, phi_ends, delta_phi_helix_, n_rot, 
                         delta_phi_helix, z_helix_start, phi_helix_start, phi_center, phi_tot, phi_spoke2, R_hub,
                         l1, r1, l2, r2, c1, c2, t1, t2, ltan, phitan, r_spokes_mid, p_spokes_mid, 
                         capstan_point_factory, arc_ew, arcs, R_rim, z_bottom, z_top, DshaftOutline,R_tunnel,transformations,
                **__):
  import cmath
  output10.clear_output(wait=True)
  with output10:
    %matplotlib inline
    fig=plt.figure(figsize=(3.5,3.5)) 
    ax=fig.add_subplot(1,1,1)  
    if phi_spoke2>60*deg:
        ax.plot(0,0,'+')
    for phi in np.exp(1j*np.linspace(0,phi_spoke2,2)):
      p1=-1.0j*(R_hub-0.2*(R_rim-R_hub))*phi
      p2=-1.0j*(R_rim+0.2*(R_rim-R_hub))*phi
      ax.plot((-p1.imag,-p2.imag),(p1.real,p2.real),'k-.',lw=1)
    xlim=ax.get_xlim()
    ylim=ax.get_ylim()
    plotArcchain(ax,R_rim+0j,0.0+1j,((l1,phi_rim*phi_spoke2),(r1*2*pi,2*pi)))
    ax.plot(-c1.imag,c1.real,('k+'))
    plotArcchain(ax,(R_hub)*exp(1j*phi_spoke2),(-1j)*exp(1j*phi_spoke2),((l2,-phi_hub*phi_spoke2),(r2*2*pi,2*pi)))
    ax.plot(-c2.imag,c2.real,'k+') 
    ax.plot([-t1.imag,-t2.imag],[t1.real,t2.real])  
    for i in range(n_strands):
        dphi=2*phi_spoke2/n_strands
        phi=exp(-1j*dphi*(i+1))
        plotArcchain(ax,(R_rim)*phi,(1j)*phi,arcs*2,c='lightgray',zorder=-1)
    if n_strands<2:
      for r in [R_hub,R_rim]:
        p=-1j*r*np.exp(1j*np.linspace(0,phi_spoke2,25))
        ax.plot(p.real,p.imag,c='lightgray',lw=1,zorder=-1)
    ax.plot(p_spokes_mid.real,p_spokes_mid.imag,'k.')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal')
    ax.set_aspect(1.0)
    display(fig)
    plt.close()
  output11.clear_output(wait=True)
  with output11:
    fig=plt.figure(figsize=(6,6)) 
    ax=fig.add_subplot(1,1,1) 
    p_mesh=list(zip(*((p.real,p.imag) for p,*_ in Segments2Complex(arcs,p0=R_rim+0.j,a0=0+1j,tol=0.01,return_start=True,loops=n_spokes))))
    ax.plot(p_mesh[0],p_mesh[1],'.-',c='lightgray',label='raw mesh',zorder=-1) 
    ax.plot([xlim[0],xlim[1],xlim[1],xlim[0],xlim[0]],[ylim[0],ylim[0],ylim[1],ylim[1],ylim[0]],'--',c='gray',lw=1,label='mesh detail')          

    transformed_mesh_points=list(zip(*[p.copy() for p in capstan_point_factory()]))
    ax.plot(transformed_mesh_points[0],transformed_mesh_points[1],'-',c='g',label='transformed mesh',zorder=1)
    plotArcchain(ax,(-1 if left_handed else 1)*r_shaft,(-1 if left_handed else 1)*1j,DshaftOutline,'b-')
    ax.plot(0,0,'b-',label='shaft')
    groove_depth_env={}
    groove_depth_pattern_factory(**(locals()|dict(env=groove_depth_env)))#get the toolpath function 
    groove_toolpath=groove_depth_env['groove_toolpath'] 
    phi_max=groove_depth_env['phi_5']
    phi_cable=np.linspace(0.0,phi_max,int((z_top-z_bottom)/groove_pitch*60)+1)
    groove_r_z=np.array([groove_toolpath(phi) for phi in phi_cable]).T
    phi_z=np.interp(z_,groove_r_z[1],phi_cable)
    r_z=groove_toolpath(phi_z)
    p_cable=r_z[0]*exp(1j*phi_z)
    def transform(pz,transformations):
        for f in transformations:
            pz=f(pz)
        return pz
    p_cable_tunnel1=transform([R_tunnel,z_,0.0],transformations[1:])[0]
    p_cable_tunnel1*=cmath.exp(-1j*phi_center)*(R_tunnel-r_cable)/R_tunnel
    p_cable_tunnel2=transform([R_tunnel*cmath.exp(1j*(-2*phi_center-phi_offset_channel2)),z_,0.0],transformations[1:])[0]
    p_cable_tunnel2*=cmath.exp(1j*phi_center)*(R_tunnel-r_cable)/R_tunnel
    if n_cable_tunnels==1:
        if z_<(z_bottom+z_top)/2:
            p_cable_tunnel2=p_cable_tunnel1
        else:
            p_cable_tunnel1=p_cable_tunnel2
    if left_handed:
        p_cable=p_cable.conjugate()
        p_cable_tunnel1=p_cable_tunnel1.conjugate()
        p_cable_tunnel2=p_cable_tunnel2.conjugate()
    ax.plot(p_cable.real,p_cable.imag,'ro',label='cable')
    ax.plot(p_cable_tunnel1.real,p_cable_tunnel1.imag,'ro')
    ax.plot(p_cable_tunnel2.real,p_cable_tunnel2.imag,'ro')
    
#    if (n_skirt>0) and (z_<=hl):
#      skirt=list(zip(*((p[0].real,p[0].imag) for offs in range(n_skirt) for p in Segments2Complex(outline,p0=L+R3+0.0j,a0=0+1j,tol=0.005,offs=0.5*offs+1,return_start=True) )))
#      ax.plot(skirt[0],skirt[1],'c-',label='skirt')
    ax.legend(loc='lower right')
#    ax.set_xlim(xlim)
#    ax.set_ylim(ylim)
    ax.set_aspect('equal')
    display(fig)
    plt.close()
  output12.clear_output(wait=True)
  with output12:
#    print(f'{R3+R4=}, {L=}, {R5+R4=}, {sss(R3+R3,L,R5+R4)/deg=}, {phi3/deg=},{phi4/deg=}, {phi5/deg=}, {R1=}')
#    print(f'{len(skirt)=}, {min(skirt[0])=:.2f}, {max(skirt[0])=:.2f}')
#    print(f'{transformed_mesh_points[0][0]=},{transformed_mesh_points[1][0]=}')
    from math import gcd
    if  gcd(n_strands,n_spokes)!=1:
        print('\x1b[31m'+f'n_strands and n_spokes are not coprime: {gcd(n_strands,n_spokes)=} (should be 1)'+'\x1b[0m')
    print(f'{r_spokes_mid=: 0.3f}')
    print(f'{R_rim=: 0.3f}, {R_hub=:0.3f}, {n_spokes=:d}, {n_strands=:d},')
    print(f'{phi_rim=:0.3f}, {r_fillet_rim=:0.3f}, {phi_hub=:0.3f}, {r_fillet_hub=:0.3f},')
    print()
    print(f'mesh={arcs[:5]}')
  plt.close()

from ipywidgets import widgets,HBox,VBox,Box
output10=widgets.Output() 
output11=widgets.Output()
output12=widgets.Output()

def handle_change_capstan(change):
  key,value=change['owner'].key,change['new']
  capstan_parameters[key] = value
  env={}
  capstan(**capstan_parameters,env=env)#calculate intermediate parameters
  update_capstan_plot(**env)
  widgetList4['z_'].max=env['z_top']
    
def save_capstan_gcode(*args,**kwargs):
    from time import perf_counter
    from datetime import datetime
    filename=capstan_parameters['design_name']+datetime.now().strftime("__%d-%m-%Y__%H-%M-%S")
    my_capstan=capstan(**(capstan_parameters|dict(z_=None))) 
    t0=perf_counter()
    capstan_status_widget.value='calculating extrusion path ...'
    my_capstan_points=[p.copy() for p in my_capstan()]
    t1=perf_counter()
    capstan_status_widget.value+=f'({t1-t0:0.3f}s), generating steps ...'
    steps=list(StepGenerator(my_capstan_points,**capstan_parameters))
    t2=perf_counter()
    capstan_status_widget.value+=f'({t2-t1:0.3f}s), running fc.transform ...'
    xmax,ymax=xmin,ymin=0,0
    for x,y in ((point.x,point.y) for point in steps if type(point)==fc.Point):
        if x!= None: xmin=min(xmin,x)
        if x!= None: xmax=max(xmax,x)
        if y!= None: ymin=min(ymin,y)
        if y!= None: ymax=max(ymax,y)
    
    model_offset = fc.Vector(x=100-0.5*(xmax+xmin), y=100-0.5*(ymax+ymin), z=0.0)
    steps = fc.move(steps, model_offset)
    gcode_controls = fc.GcodeControls(
         printer_name=capstan_parameters['printer_name'],
         save_as=filename,
         include_date=False,
         initialization_data={
         'primer': 'no_primer',
         }|capstan_parameters|({'fan_percent':0.0} if capstan_parameters.get('fan_z_start',0.0)!=0.0 else {})
         )
    capstan_status_widget.value=' saving gcode to "'+filename+'.gcode" ...'
    fc.transform(steps, 'gcode', gcode_controls)
    if "colab" in globals():
        capstan_status_widget.value=' preparing file for download...'
        colab.files.download(filename+".gcode")
        capstan_status_widget.value=''
    else:
        capstan_status_widget.value+=' File saved to disk on server! Download manually from there.'
        
def update_capstan_preview(*args,**kwargs):
    from time import perf_counter
    import os
    preview_output.clear_output(wait=True)
    with preview_output:
        my_capstan=capstan(**(capstan_parameters|dict(z_=None))) 
        t0=perf_counter()
        capstan_status_widget.value='calculating extrusion path ...'
        my_capstan_points=[p.copy() for p in my_capstan()]
        t1=perf_counter()
        capstan_status_widget.value+=f'({t1-t0:0.3f}s), generating steps ...'
        steps=list(StepGenerator(my_capstan_points,**capstan_parameters))
        t2=perf_counter()
        capstan_status_widget.value+=f'({t2-t1:0.3f}s), running fc.transform ...'
        xmax,ymax=xmin,ymin=0,0
        for x,y in ((point.x,point.y) for point in steps if type(point)==fc.Point):
            if x!= None: xmin=min(xmin,x)
            if x!= None: xmax=max(xmax,x)
            if y!= None: ymin=min(ymin,y)
            if y!= None: ymax=max(ymax,y)
        model_offset = fc.Vector(x=100-0.5*(xmax+xmin), y=100-0.5*(ymax+ymin), z=0.0)
        steps = fc.move(steps, model_offset)
        style='tube'
        if ('iPad' in os.uname().machine):
            plt.close()
            capstan_status_widget.value='running fc.transform...'
            fc.transform(steps, 'plot', fc.PlotControls(style='line',color_type='print_sequence'))
            capstan_status_widget.value=''
            _=plt.show()
        else:
            capstan_status_widget.value='running fc.transform...'
            _=fc.transform(steps, 'plot', fc.PlotControls(style=style ,color_type='print_sequence'))
            capstan_status_widget.value=''
   
style={'description_width':'1.5in'}
layout={'width':'2.2in'}

def handle_change_capstan(change):
  key,value=change['owner'].key,change['new']
  capstan_parameters[key] = value
  env={}
  capstan(**capstan_parameters,env=env)#calculate intermediate parameters
  update_capstan_plot(**env)
  widgetList4['z_'].max=env['z_top']

widgetList4={}
for key,value in capstan_parameters.items():
    if type(value)==int:
        widgetList4[key]=widgets.IntText(description=key, value=value,style=style,layout=layout, readout_format='d')
    elif type(value)==float:
        widgetList4[key]=widgets.FloatText(description=capstan_parameter_descriptions.get(key,key), value=value,style=style,layout=layout, readout_format='0.3f')
    elif type(value)==str:
        widgetList4[key]=widgets.Text(description=key, value=value,style=style,layout=layout,)
    elif type(value)==bool:
        widgetList4[key]=widgets.Dropdown(description=key,options=[True,False], value=value,style=style,layout=layout,indent=True)
    else: print(f'need to add widget for {key=}, {type(value)=}');continue
        
env={}
capstan(**capstan_parameters,env=env)        

key='z_'
widgetList4[key]=widgets.FloatSlider(description=key,
            min=env['z_bottom'],max=env['z_top'],value=capstan_parameters['z_'],step=0.2,continuous_update=True, orientation='vertical',   
           readout_format='0.1f',layout={'height':'4.5in'})
key='shaft_type'
widgetList4[key]=widgets.Dropdown(description=key,
            options=['O','D','DD'],value=capstan_parameters['shaft_type'],layout=layout,style=style)
key='n_cable_tunnels'
widgetList4[key]=widgets.Dropdown(description=capstan_parameter_descriptions.get(key,key),
            options=[1,2],value=capstan_parameters['n_cable_tunnels'],layout=layout,style=style)

for key in capstan_parameters: 
    widgetList4[key].observe(handle_change_capstan,'value')      
    widgetList4[key].key=key
SaveCapstanGcodeButton=widgets.Button(description='download G-Code',on_click=save_capstan_gcode,button_style='primary')
SaveCapstanGcodeButton.on_click(save_capstan_gcode)
UpdateCapstanPreviewButton=widgets.Button(description='update preview',on_click=update_capstan_preview)
UpdateCapstanPreviewButton.on_click(update_capstan_preview)
capstan_status_widget=widgets.Label()
widgetList4['design_name'].layout.width='3.0in'
widgetList4['printer_name'].layout.width='3.0in'
Layout4=widgets.Tab([
          VBox([              
              HBox([VBox([
                       widgets.HTML(value='<h2 align=left>Mesh</h2>'),
                       HBox([widgetList4['n_spokes'],widgetList4['n_strands'],]),
                       HBox([widgetList4['l_turn'],widgetList4['d_shaft'],]),
                       HBox([widgetList4['phi_rim'],widgetList4['ew_rim'],]),
                       HBox([ widgetList4['r_fillet_rim'],widgetList4['ew_fillet_rim'],]),
                       HBox([widgetList4['spoke_midpoint'],widgetList4['ew_spokes'],]),
                       HBox([widgetList4['r_fillet_hub'],widgetList4['ew_fillet_hub'],]),
                       HBox([widgetList4['phi_hub'],widgetList4['ew_hub'],],),
                       ]),
                    output10,
                   ],layout={'border': '1px solid black'}),
              HBox([VBox([
                       widgets.HTML(value='<h2 align=left>Capstan Geometry: </h2>'),
                       widgetList4['l_tot'],widgetList4['l_turn'],widgetList4['groove_pitch'],widgetList4['d_shaft'],widgetList4['shaft_tolerance'],widgetList4['shaft_type'],widgetList4['D_key'],
                       widgetList4['mesh_twist_pitch'],widgetList4['n_cable_tunnels'], widgetList4['tunnel_pos'],widgetList4['countersink_chamfer'],
                       widgetList4['left_handed'],
                       widgetList4['n_skirt'],widgetList4['skirt_offset'],
                        ],),
                   HBox([output11,widgetList4['z_']]),
                   ],layout={'border': '1px solid black'}),
             HBox([VBox([
#                      widgets.HTML(value='<h2 align=left>Output: </h2>'),#title of output cell 
                      output12,
                       ]),
                  ],layout={'border': '1px solid black'} ),
            ]),
        HBox([             
             VBox([
               widgets.HTML(value='<h2 align=left>Print Parameters: </h2>'),
               HBox([widgetList4['hl'],UpdateCapstanPreviewButton,capstan_status_widget],)
               ]),
           #  tripod_preview_output, #plotly output not showing inside HBox
             ],layout={'border': '1px solid black'}),
        VBox([
          widgets.HTML(value='<h2 align=left>Print Parameters: </h2>'),
          HBox([widgetList4['hl'],widgetList4['nominal_ew'],]),
          HBox([widgetList4['nominal_print_speed'],]),
          HBox([widgetList4['nozzle_temp'],widgetList4['bed_temp'],]),
          HBox([widgetList4['fan_percent'],widgetList4['fan_z_start'],]),
          HBox([widgetList4['printer_name'],]),
          HBox([widgetList4['design_name'],],),
          HBox([SaveCapstanGcodeButton,capstan_status_widget],)
          ],layout={'border': '1px solid black'}),
      ],titles=['Design','Preview','G-Code'])

update_capstan_plot(**env);
output12.clear_output(wait=False)#make form smaller 
plt.close()
```

<!-- #region jupyter={"outputs_hidden": false, "source_hidden": false} pythonista={"disabled": false} -->
# Interactive Section
<!-- #endregion -->

```python jupyter={"source_hidden": true} pythonista={"disabled": false}
Layout=widgets.Tab([Layout1,Layout2,Layout3,Layout4],selected_index=3)
Layout.set_title(0,'Linear Mesh')
Layout.set_title(1,'Circular Mesh')
Layout.set_title(2,'Tripod')
Layout.children[2].set_title(0,'Design')#colab, for some reason, shows empty Tab titles if not explicitly set like this
Layout.children[2].set_title(1,'Preview')
Layout.children[2].set_title(2,'G-Code')
Layout.set_title(3,'Capstan')
Layout.children[3].set_title(0,'Design')#colab, for some reason, shows empty Tab titles if not explicitly set like this
Layout.children[3].set_title(1,'Preview')
Layout.children[3].set_title(2,'G-Code')

```

<!-- #region jupyter={"outputs_hidden": false, "source_hidden": false} pythonista={"disabled": false} -->
## Design Form
<!-- #endregion -->

```python jupyter={"outputs_hidden": false, "source_hidden": false} pythonista={"disabled": false}
display(Layout);
```

```python jupyter={"source_hidden": false} pythonista={"disabled": false}
#%matplotlib ipympl
if "colab" in globals(): colab.output.enable_custom_widget_manager()
display(preview_output)
if "colab" in globals(): colab.output.disable_custom_widget_manager()
```

<!-- #region jupyter={"outputs_hidden": false, "source_hidden": false} pythonista={"disabled": false} -->
# Appendix
<!-- #endregion -->

```python jupyter={"source_hidden": true} pythonista={"disabled": false}

```

```python jupyter={"source_hidden": true} pythonista={"disabled": false}
raise Exception('Exception raised to stop execution of the remaining cells in this Notebook')
```

```python jupyter={"source_hidden": true} pythonista={"disabled": false}
print_parameters=dict(
design_name = 'Tripod',
nozzle_temp = 220,
bed_temp = 120,
nominal_print_speed = 20*60,#10*60 #print slow to give the layer time to cool
nominal_ew = 0.75,   # extrusion width
fan_percent = 0,
nominal_eh = 0.2,    # extrusion/layer heigth
printer_name='generic', # generic / ultimaker2plus / prusa_i3 / ender_3 / cr_10 / bambulab_x1 / toolchanger_T0
)
```

<!-- #region jupyter={"outputs_hidden": false, "source_hidden": false} pythonista={"disabled": false} -->
### calculate Tripod extrusion path
<!-- #endregion -->

```python jupyter={"source_hidden": true} pythonista={"disabled": false}
import time
t0=time.perf_counter()
my_model=capstan
my_model_parameters=capstan_parameters 
my_model_point_generator=my_model(**(my_model_parameters|dict(z_=None))) 
my_model_points=list(my_model_point_generator( ))
t1=time.perf_counter()
steps=list(StepGenerator(my_model_points,**my_model_parameters))
t2=time.perf_counter()
print(f'time to generate extrusion path coordinates: {t1-t0:.3f}s, time to generate fc.Points from coordinates: {t2-t1:.3f}s ')
len(steps)
xmax,ymax=xmin,ymin=0,0
for x,y in ((point.x,point.y) for point in steps if type(point)==fc.Point):
    if x!= None: xmin=min(xmin,x)
    if x!= None: xmax=max(xmax,x)
    if y!= None: ymin=min(ymin,y)
    if y!= None: ymax=max(ymax,y)

model_offset = fc.Vector(x=100-0.5*(xmax+xmin), y=100-0.5*(ymax+ymin), z=0.0)
steps = fc.move(steps, model_offset)
steps.append(fc.Point(x=100,y=100))
```

```python jupyter={"source_hidden": true} pythonista={"disabled": false}
my_model_points[20000]
```

<!-- #region jupyter={"outputs_hidden": false, "source_hidden": false} pythonista={"disabled": false} -->
### show Tripod preview
<!-- #endregion -->

```python jupyter={"source_hidden": false} pythonista={"disabled": false}
# add annotations and plot
import os
style='tube'
import time 
t0=time.perf_counter()
fc.transform(steps, 'plot', fc.PlotControls(style=style if not ('iPad' in os.uname().machine) else 'line',color_type='print_sequence'));
t1=time.perf_counter()
print(f'Time to cerate the FullControl preview: {(t1-t0):0.3f}s') 
```

<!-- #region jupyter={"outputs_hidden": false, "source_hidden": false} pythonista={"disabled": false} -->
### Generate Tripod gcode
<!-- #endregion -->

```python jupyter={"source_hidden": true} pythonista={"disabled": false}
gcode_controls = fc.GcodeControls(
    printer_name=print_parameters['printer_name'],
    save_as=print_parameters['design_name'],
    initialization_data={
        'primer': 'no_primer',
         }|print_parameters)
import time 
t0=time.perf_counter()
#gcode = fc.transform(steps, 'gcode', gcode_controls)
t1=time.perf_counter()
print(f'Time to generate G-Code file from FullControl steps: {(t1-t0):0.3f}s') 
```

<!-- #region jupyter={"outputs_hidden": false, "source_hidden": false} pythonista={"disabled": false} -->
## cProfile tripod
<!-- #endregion -->

```python jupyter={"source_hidden": true} pythonista={"disabled": false}
from time import perf_counter
t1=perf_counter()
myTripod=tripod(**tripod_parameters)
p=list(myTripod())
t2=perf_counter()
print(f'Time to generate the points for one layer: {t2-t1}s')
import cProfile
cProfile.run('p=list(myTripod())',sort=1)
```

```python jupyter={"source_hidden": true} pythonista={"disabled": false}

```

```python jupyter={"source_hidden": true} pythonista={"disabled": false}
from matplotlib import pyplot as plt
xy=list(zip(*p))
plt.plot(*xy[:2])
plt.gca().set_aspect('equal')
```

```python jupyter={"source_hidden": true} pythonista={"disabled": false}
mesh=[(1.9332877868244882, 0.19332877868244883), (0.9388633012258815, 1.2950143724065095), (7.473407339055301, 0), (0.28796015806691877, -0.3404535276619185), (0.2114533516839284, 0.060415243338265257)]
mesh=mesh[:2]+[(mesh[2][0]/2,0)]*2+mesh[3:]#split the spoke in half
plotArcchain(plt.gca(),10,1j,(mesh+mesh[-1::-1])*13,'b.-')
plt.gca().set_aspect('equal') 
```

```python jupyter={"source_hidden": true} pythonista={"disabled": false}
mesh=[(0.15, 0), (0.10412385590811431, 1.041238559081143), (1.0440306508910548, 0), (0.10412385590811431, -1.041238559081143), (0.15, 0)]
plotArcchain(plt.gca(),0,1,(mesh+mesh[-1::-1])*3,'b.-')
plt.gca().set_aspect('equal') 
```

```python jupyter={"source_hidden": true} pythonista={"disabled": false}
from matplotlib import pyplot as plt
import numpy as np
z=np.linspace(0,6,200)
M8_int=ISO_thread(Pitch=1.25)
M8_ext=ISO_thread(Pitch=1.25,External=True)
plt.plot([M8_int(z_,0.5)+4 for z_ in z],z,'r-',label='internal thread (nut)')
plt.plot([M8_ext(z_,0.5)-0.1+4 for z_ in z],z,'g-',label='external thread (bolt)')
plt.plot([-(M8_int(z_)+4) for z_ in z],z,'r-')
plt.plot([-(M8_ext(z_)-0.1+4) for z_ in z],z,'g-')
plt.title('ISO-thread(Pitch=1.25)')
plt.legend()
plt.gca().set_aspect('equal')
```

```python jupyter={"source_hidden": true} pythonista={"disabled": false}
from timeit import timeit
cos_sin=pipe(iterize(lambda x:1j**(x*4/(2*pi))), iterize(lambda x:(x.real,x.imag)))
print(*('/'.join(f'{x:.5f}' for x in sc) for sc in cos_sin(iter(range(7)))),sep=', ')
print(timeit(lambda:list(cos_sin(iter(range(1000)))),number=1000))
def pipetest(x=None,*,description,function):
    print(f'initializing "{description}"')
    def wrapper(x,/):
        print(f'in wrapper "{description}"')
        for xi in x:
            print(f'yielding {description}({xi})={function(xi)}')
            yield function(xi)
            print(f'post yield {description}({xi})')
        print(f'post loop "{description}"')    
        return
    return wrapper(x) if x!=None else wrapper
import cmath 
print('p=pipe(...):')
p=pipe((pipetest(description='inc',function=lambda x:x+1)),None,iterize(iterize(None)),iterize(lambda x:2*x),pipetest(description='sin',function=cmath.sin),)
print('piter=p(iter(...)):')
piter=p(iter([0,10,20.0,30+0.1j]))
print('\ncalling next(piter):\n')
x=next(piter)        
print(f'x={x}')
print(f'\nrunning the loop:\n')
for i,x in enumerate(piter):
    print(f'x{i}={x}\n')
```

```python jupyter={"source_hidden": true} pythonista={"disabled": false}
import numpy as np
from matplotlib import pyplot as plt
def plot_depthpattern(ax5,*,z_bottom,z_top,hl,thread_depth_pattern,r_circ,r_cable,**_):
    z=np.arange(z_bottom,z_top+hl,hl)
    phi=np.linspace(0.0,2*pi,300)
    Phi,Z=np.meshgrid(phi,z)
    R=np.vectorize(thread_depth_pattern)(Z,Phi)
    ax5.imshow(R, interpolation='bilinear', cmap='gray',
                   origin='lower', extent=[0,r_circ*2*pi, z_bottom, z_top],
                   vmax=r_circ+r_cable, vmin=r_circ-1.5*r_cable)
    ax5.set_title('Depth Map')
    ax5.set_ylabel('axial position [mm]')
    ax5.set_xlabel('circumferential position [mm]')
    ax5.set_aspect('equal')
fig=plt.figure(figsize=(12,18))
ax5=fig.add_subplot(1,1,1)
env={}
capstan(**capstan_parameters,env=env)
thread_depth_pattern=groove_depth_pattern_factory(**env)
plot_depthpattern(ax5,**env)#,thread_depth_pattern= thread_depth_pattern)
```

```python jupyter={"source_hidden": true} pythonista={"disabled": false}
widgetList4['l_tot'].value
```

```python jupyter={"source_hidden": true} pythonista={"disabled": false}
debug_view = widgets.Output(layout={'border': '1px solid black'})

@debug_view.capture(clear_output=True)
def bad_callback(event):
    print('This is about to explode')
    return 1.0 / 0.0

button = widgets.Button(
    description='click me to raise an exception',
    layout={'width': '300px'}
)
button.on_click(bad_callback)
VBox([button,debug_view])
```

```python jupyter={"source_hidden": true} pythonista={"disabled": false}
env={}
tripod(**tripod_parameters,env=env)
', '.join(env.keys())
```

```python jupyter={"source_hidden": true} pythonista={"disabled": false}
from IPython.display import display, HTML
display(HTML("<style>div.output_scroll { height: 44em; }</style>"))
```

```python jupyter={"source_hidden": true} pythonista={"disabled": false}
display(preview_output)
update_tripod_preview()
```

```python jupyter={"source_hidden": true} pythonista={"disabled": false}
import plotly.graph_objs as go
from ipywidgets import interact
fig = go.FigureWidget()
scatt = fig.add_scatter()
scatt=scatt.data[0]
fig
```

```python jupyter={"source_hidden": true} pythonista={"disabled": false}
"colab" in globals()
```

```python jupyter={"source_hidden": true} pythonista={"disabled": false}
import inspect
```

```python jupyter={"source_hidden": true} pythonista={"disabled": false}
inspect.getsourcelines(NamedList)
```

```python jupyter={"source_hidden": true} pythonista={"disabled": false}
inspect.getsourcelines(NamedList.__class_getitem__)
```

```python jupyter={"source_hidden": true} pythonista={"disabled": false}
def calct(x,y,X=None,Y=None):
  t=1.0
  if X!=None:
      x0,x1,x2,x3=X
      if(x<x0 or x>x3):
          return 0.0
      elif x<x1:
          t=(x-x0)/(x1-x0)
      elif x>=x2:
          t=(x3-x)/(x3-x2)
  if Y!=None:
      y0,y1,y2,y3=Y
      if (y<y0 or y>y3):
        return 0.0
      elif y<y1:
        t=t*((y-y0)/(y1-y0))
      elif y>=y2:
        t=t*((y3-y)/(y3-y2))
  return t
```

```python jupyter={"source_hidden": true} pythonista={"disabled": false}
import numpy as np
from matplotlib import pyplot as plt
X=[-1.5,-0.5,0.5,2.5]
Y=[-1.5,-0.5,0.5,1.5]
x=np.linspace(-1.5,2.5,36)
y=np.linspace(-1.5,1.5,31)
xi,yi=np.meshgrid(x,y)
t=np.vectorize(lambda x,y:calct(x,y,X,Y=Y))(xi,yi)
plt.imshow(t, interpolation='bilinear', cmap='jet',origin='lower', extent=[-1.5,2.5, -1.5, 1.5],)
```

```python jupyter={"source_hidden": true} pythonista={"disabled": false}
px=np.array([x for row in xi for x in row])
py=np.array([y for row in yi for y in row])
vx,vy=-0.5,0.0
t=t=np.vectorize(lambda x,y:calct(x/vx,y/vx,X,Y=Y))(px,py)
plt.plot(px+vx*t,py+vy*t, '+')
plt.show()
```

```python jupyter={"source_hidden": false} pythonista={"disabled": false}
#dodecahedron pattern 
from math import pi,tan,sin
deg=pi/180.0

pentagon=([(1.0,0),(0,72*deg)]*2+[(1,0),(0,-36*deg)]*2+[(1.0,0),(0,72*deg)])
scoreline=[(0.0,-108*deg)]+((pentagon*5)[2:-1]+[(0.0,-36*deg)] +(pentagon*5)[2:])+[(1.0,0),(0.0,180*deg),(1.5,0)]

fs=0.5#flap start
fsa=80*deg#start angle
fsb=20*deg#flap start edge curvature 80deg+20deg=100deg at root -> 'snap'-fit
fe=0.95#flap end
fea=30*deg#end angle
fw=0.16#flap width
flap=[(fs,0),(0.,-fsa-fsb),(fw/sin(fsa)*(fsb/sin(fsb) if fsb!=0.0 else 1.0),2*fsb),(0.0,fsa-fsb),
       (fe-fs-fw*(1/tan(fsa)+1/tan(fea)),0),(0,fea),(fw/sin(fea),0),(0,-fea),(1.0-fe,0)]
flapped_pentagon=[*flap,(0,72*deg)]*2+[*flap,(0,-144*deg)]+[*flap,(0.0,72*deg)]
cutline=[(0.0,-108*deg)]+(flapped_pentagon*5)[len(flap)+1:-1]+[(0.0,-36*deg)]+(flapped_pentagon*5)[len(flap)+1:-1]+[(0,-108*deg),(0.25,0)]


from matplotlib import pyplot as plt 
plt.close()
#DIN_A=4
#plt.figure(figsize=(1000*2**(-DIN_A/2+1/4)/25.4,1000*2**(-DIN_A/2-1/4)/25.4))
plt.figure(figsize=(11,8.5))
plotArcchain(plt.gca(),0.0,0.0 +1.0j,scoreline,'k:',)
plotArcchain(plt.gca(),0.0,0.0 +1.0j,cutline,'k-')
plt.plot(0,0,'k-',label='cut')
plt.plot(0,0,'k:',label='fold up')
plt.gca().set_aspect('equal')
xlim=plt.gca().get_xlim()
ylim=plt.gca().get_ylim()
plt.legend()
plt.gca().set_axis_off()
plt.savefig('dodecahedron.pdf')
plt.show()
(xlim[1]-xlim[0])/(ylim[1]-ylim[0])
```

```python jupyter={"source_hidden": false} pythonista={"disabled": false}
from IPython.display import display, HTML
points=[(100,10), (40,198), (190,78), (10,78), (160,198)]
scale=70
points=[(p.imag*scale,p.real*scale) for p,*_ in Segments2Complex(scoreline,3.7+4.1j,.0+1j,tol=0.001)]
HTML(f"""<!DOCTYPE html>
<html>
<body>
​
<svg width="600" height="450">
  <polygon points="{' '.join('%.1f'%x+','+'%.1f'%y for x,y in points)}"
  style="fill:lime;stroke:purple;stroke-width:3;fill-rule:even;" />
Sorry, your browser does not support inline SVG.
</svg>
 
</body>
</html>
​""")
```

```python jupyter={"outputs_hidden": false, "source_hidden": false} pythonista={"disabled": false}
from IPython.display import display, HTML
HTML(f"""<!DOCTYPE html>
<html> 
<body>

<canvas id="myCanvas" width="200" height="100" style="border:1px solid #d3d3d3;">
Your browser does not support the HTML canvas tag.</canvas>

<script>
var c = document.getElementById("myCanvas");
var ctx = c.getContext("2d");
ctx.beginPath();
ctx.arc(95,50,40,0,2*Math.PI);
ctx.arc(40,50,40,Math.PI,1.5*Math.PI);
ctx.stroke();
</script> 

</body>
</html>""")
```

```python jupyter={"source_hidden": false} pythonista={"disabled": false}
from IPython.display import display, HTML
canvas=HTML(f"""<!DOCTYPE html>
<html> 
<header>
<script>
function cmul(a,b):
  return [a[0]*b[0]-a[1]*b[1],a[0]*b[1]+a[1]*b[0]]
function cabs(a):
  return [Math.sqrt(a[0]**2+a[1]**2),0.0]
</script>
</header>
<body>
​<p id='output'>Results:<br/></p>
<p id='demo'>Java Script Demo</p>
<canvas id="myCanvas1"  width="300" height="200" style="border:1px solid #d3d3d3;">
Your browser does not support the HTML canvas tag.</canvas>
​<script>
var output = document.getElementById("output");
const cars = new Array(["Audi", "Volvo", "BMW"]);
//const p = new Array([100, 100]);
//let r=[0,1]
output.innerHTML=cars[0]+'<br/>'
document.getElementById("output").innerHTML="test"+<br/>
document.getElementById("demo").innerHTML = "Hello JavaScript!"
</script>
<script>
var c = document.getElementById("myCanvas1");
var ctx = c.getContext("2d");
ctx.beginPath();
ctx.moveTo(0,0);
ctx.lineTo(100,100);
ctx.arcTo(150,150,100,200,50);
ctx.lineTo(100,200)
ctx.stroke();
</script> 
</body>
</html>""")
display(canvas)
```

```python jupyter={"outputs_hidden": false, "source_hidden": false} pythonista={"disabled": false}
from IPython.display import display, HTML
ht=HTML("""<!DOCTYPE html>
<html>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<body>
​
<div id="myPlot" style="width:100%;max-width:700px"></div>
​
<script>
const xArray = [50,60,70,80,90,100,110,120,130,140,150];
const yArray = [7,8,8,9,9,9,10,11,14,14,15];
​
// Define Data
const data = [{
  x: xArray,
  y: yArray,
  mode:"lines"
}];
​
// Define Layout
const layout = {
  xaxis: {range: [40, 160], title: "Square Meters"},
  yaxis: {range: [5, 16], title: "Price in Millions"},  
  title: "House Prices vs. Size"
};
​
// Display using Plotly
Plotly.newPlot("myPlot", data, layout);
</script>
​
</body>
</html>""")
display(ht)
```

```python jupyter={"source_hidden": true} pythonista={"disabled": false}
x=[1,2,3,4,5,6]
cumsum(x,0)
```

```python jupyter={"source_hidden": true} pythonista={"disabled": false}
np.cumsum(np.insert(x,0,0.0))
```

```python jupyter={"source_hidden": true} pythonista={"disabled": false}
ps=point_shiftor(p0=9.6,tol=0.5,dang=1)
ps([10,1])
```

```python jupyter={"source_hidden": true} pythonista={"disabled": false}
x=NamedList([4,5,6],a=5,b=6,_1=2)
x.alias(a='x')
x.rename(b='v')
x.v
```

```python jupyter={"source_hidden": true} pythonista={"disabled": false}
from IPython.display import Javascript
x=1
```

```javascript jupyter={"source_hidden": true} pythonista={"disabled": false}
var x=2
```

```python jupyter={"source_hidden": true} pythonista={"disabled": false}
%matplotlib ipympl
```

```python jupyter={"outputs_hidden": false, "source_hidden": false} pythonista={"disabled": false}
with open('test.txt','w') as f:
    f.write('hello test')
```

```python jupyter={"outputs_hidden": false, "source_hidden": false} pythonista={"disabled": false}
import os
os.getcwd()
```

```python jupyter={"outputs_hidden": false, "source_hidden": false} pythonista={"disabled": false}

```
