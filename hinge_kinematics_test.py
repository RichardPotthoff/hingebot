import io
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
from cmath import pi,exp,polar
from math import atan2,copysign,sqrt
deg=pi/180

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
  def interpolateArcChain(t):
      T=int(t)
      if is_closed_loop:
          pr,ar=0.0+0.0j,1.0+0.0j
      else: #endpoint of path != startpoint => repeat path for t>1 by translating and rotating it
          v=pSeg[-1]
          beta=ang_[-1]
          beta2=beta/2
          rot2=cmath_exp(1j*beta2)
          pr=(v*rot2**(T-1)/sinc(beta2) * T * sinc(T*beta2))*a0 #???
          ar=(rot2**(2*T))
      pr+=p0
      ar*=a0
      X,x=index_frac(t-T,l)
      p=pSeg[X] + sinc( dang[X]*x/2)* dl[X]*x *scale*cmath_exp(1j* dang[X]*x /2)*ang[X]
      p=p*ar+pr
      a=ang[X]*cmath_exp(1j*dang[X]*x)*ar
      return p,a,L[-1]*t,X
  return interpolateArcChain


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
        
def evolvent(t,t0=0.0,r=1):
  from numpy import exp,sin,cos
  tt0=t+t0
  return r*(cos(tt0)+t*sin(tt0)+1j*(sin(tt0)-t*cos(tt0)))
  
def sss(c,a,b):#triangle with 3 sides: return angle opposite first side 'c'
    cosgamma=(a*a+b*b-c*c)/(2*a*b)
    return cosgamma 
    
def intersect_circles(c1x,c1y,r1,c2x,c2y,r2):# intersection point of 2 circles
    dx,dy=c1x-c2x,c1y-c2y
    dl=sqrt(dx*dx+dy*dy)
    ex,ey=dx/dl,dy/dl 
    cos1=sss(r1,dl,r2)
    sin1=sqrt(1.0-cos1*cos1)
    p1x,p1y=c2x+r2*(ex*cos1+ey*sin1),c2y+r2*(-ex*sin1+ey*cos1)
    p2x,p2y=c2x+r2*(ex*cos1-ey*sin1),c2y+r2*(ex*sin1+ey*cos1)
    if (p1x*p1x+p1y*p1y)<(p2x*p2x+p2y*p2y):#point closest to the origin
        return p1x,p1y
    else:
        return p2x,p2y

class stepper:
  def __init__(self,cx,cy,cz,r,L):
    self.C=cx+1j*cy
    self.cx=cx
    self.cy=cy
    self.cz=cz
    self.ang0=atan2(-cy,-cx)
    self.r=r
    self.L=L
    
  def stepper_angle(self,p):
    dl=p-self.Center 
    r=self.l_turn/(2*pi) # radius of the capstan
    l=(abs(dl)**2-r**2)**0.5 # length of unwound cable
    abs_dl,ang_dl=polar(dl) 
    return ang_dl+atan2(r,abs_dl)+l/r+(pi/2 if self.l_turn<0 else -pi/2)
    
  def stepper_pos(self,p):
    cx,cy=self.cx,self.cy
    px,py=p.real,p.imag
    dx,dy=cx-px,cy-py
#    r=self.r # radius of the capstan
    ucl=sqrt(dx*dx+dy*dy-self.r*self.r) # length of unwound cable
    dl_ang=atan2(-dy,-dx)-self.ang0#angle difference between capstan->origin and capstan->point
    if dl_ang<-pi:dl_ang+=2*pi
    elif dl_ang>pi:dl_ang-=2*pi
    rl_ang=atan2(self.r,ucl) 
    return self.r*(dl_ang+rl_ang) +ucl
  
   
  def plot(self,ax,p=0,*args,stepper_pos=None,adjust_angle=0,plot_evolvent=False,plot_circle=False,plot_rod=True,
          r_hinge=6,w_hinge=20,r_spring=5,n_spring=3):
    C=self.cx+self.cy*1j
    L=self.L
    r=self.r
    sign_r=copysign(1.0,r)
    abs_r=abs(r)
    dl=p-C
    l=(abs(dl)**2-r**2)**0.5
    if stepper_pos!=None:
      l=stepper_pos-adjust_angle*r
      rotation_angle=stepper_pos/r+self.ang0
    else:
      rotation_angle=self.stepper_pos(p)/r+self.ang0
    
    hook=[(0,pi/2),(r_spring*pi/2,-pi/2),(0,pi),(r_spring*3/2*pi,3/2*pi),(0,-pi/2),(0*r_spring,0)]
    helix=[(0,pi/2-pi/20),(r_spring,0),(r_spring/2,-pi+2*pi/20),(2*r_spring,0),
          (r_spring/2,pi-2*pi/20),(r_spring,0), (0,-pi/2+pi/20) ]
    mirror=lambda path:[(l,-a) for l,a in path]
    spring=[(r_spring,0)]+hook+helix*n_spring+mirror(hook[-1::-1])+[(r_spring,0)]
    fspring=arcChainInterpolator(arcChain=spring)
    l_spring=abs(fspring(1)[0])#distance between the first and last point of the spring
    
    capstan_cable_turns=(-l/abs_r)%(2*pi)+2*pi # at least one turn 
    capstan=[(0,-pi/2),(abs_r,0),(0,pi/2),(abs_r*capstan_cable_turns,capstan_cable_turns)]#starting from the center
    cable=[(l-r_hinge,0)]
    hinge=[(0,-90*deg),(r_hinge*450*deg,450*deg),(0,-90*deg),(w_hinge,0),(0,180*deg),(w_hinge/2,0),(0,90*deg)]
    rod=[(L+l_spring,0),(0,90*deg),(w_hinge/8,0),(0,90*deg),(L+l_spring,0),(0,-90*deg), (w_hinge/8,0), (0,-90*deg),
        (L+l_spring,0), (0,-90*deg), (r_hinge+w_hinge*(1/2+1/8+1/8),0), (0,-90*deg)]
    cable2=[(L-l,0)]
    arcChain= capstan+cable+hinge+rod+spring+cable2
    if sign_r<0:
      arcChain=mirror(arcChain)
    plotArcchain(ax,C,exp(1j*rotation_angle),arcChain,*args,zorder=-1,lw=0.7)
    if plot_evolvent:
      t=np.linspace(0,2*pi*9,1000)
      p_ev=evolvent(-t*sign_r,t0=rotation_angle-pi/2*sign_r,r=abs_r)+C
      plt.plot(p_ev.real,p_ev.imag,'k-',lw=0.1)
    if plot_circle:
      tan_ang=rotation_angle+((-l/abs_r)%(2*pi)-pi/2)*sign_r
      cc=C+abs_r*exp(1j*tan_ang)
      plt.plot(cc.real,cc.imag,'kx')
      plotArcchain(plt.gca(),cc,exp(1j*(tan_ang+pi/2*sign_r )),[(l,0),(0,pi/2),(l*2*pi,2*pi)],'r-',lw=0.15)
    
t=np.linspace(0,2*pi*9,1000)
plt.figure(figsize=(5,5))
plt.gca().set_aspect('equal')
l_turn=120
r=l_turn/(2*pi)
Lx=500
Ly=550
Cx=-350+r*1j
Cy=-r+360j
n=61
R=100
fcircle=arcChainInterpolator(arcChain=[(R*2*pi,2*pi)],p0=R)
circle=lambda t:fcircle(t)[0]
frectangle=arcChainInterpolator(arcChain=[(1.5*R,0),(0,90*deg),(2*R,0),(0,90*deg),(1.5*R,0)]*2,p0=R)
rectangle=lambda t:frectangle(t)[0]
l=np.array([rectangle(j/(n-1)) for j in range(n+1)])
i_p=59
plt.plot(0,0,'b+')
stepper_x=stepper(Cx.real,Cx.imag,0,r,L=Lx)
stepper_y=stepper(Cy.real,Cy.imag,0.0,-r,L=Ly)
p=l[i_p]
sp_x=stepper_x.stepper_pos(p)
sp_y=stepper_y.stepper_pos(p)
steppers=[stepper_x,stepper_y]
stepper_positions=[stepper.stepper_pos(p) for stepper in steppers]
#sp_x0=stepper_x.stepper_pos(-0+0j)
#p_guess=0+0j
#plt.plot(p_guess.real,p_guess.imag,'kx')
def calc_pos(steppers,stepper_positions,p_guess=0.0+0.0j):
  from math import sqrt, pi, sin, cos, atan2, copysign
  def evolvent(t,t0=0.0,r=1.0):
    tt0=t+t0
    return r*(cos(tt0)+t*sin(tt0)+1j*(sin(tt0)-t*cos(tt0)))
  def intersect_circles(c1x,c1y,r1,c2x,c2y,r2):# intersection point of 2 circles
      def sss(c,a,b):#triangle with 3 sides: return cosine of angle opposite first side 'c'
        cosgamma=(a*a+b*b-c*c)/(2*a*b)
        return cosgamma 
      dx,dy=c1x-c2x,c1y-c2y
      dl=sqrt(dx*dx+dy*dy)
      r2ex=r2*dx/dl
      r2ey=r2*dy/dl 
      cos1=sss(r1,dl,r2)
      sin1=sqrt(1.0-cos1*cos1)
      p1x=c2x+( r2ex*cos1 + r2ey*sin1)
      p1y=c2y+(-r2ex*sin1 + r2ey*cos1)
      p2x=c2x+( r2ex*cos1 - r2ey*sin1)
      p2y=c2y+( r2ex*sin1 + r2ey*cos1)
      if (p1x*p1x+p1y*p1y)<(p2x*p2x+p2y*p2y):#choose point closest to the origin
          return p1x,p1y
      else:
          return p2x,p2y
  pe=[None]*2#cable end point
  pt=[None]*2#cable tangent point (at capstan)
  ucl=[None]*2#unwound cable length
  for j in range(10):
    for i,(stepper,sp) in enumerate(zip(steppers,stepper_positions)):
      drot1=(p_guess-stepper.C)*((abs(p_guess-stepper.C)**2-stepper.r**2)**0.5+1j*stepper.r)
      pt[i] = -1j*stepper.r * drot1/abs(drot1) + stepper.C #tangent point = center of approx. circle
      drot=drot1*(0+0j-stepper.C).conjugate()#rotation relative to stepper
      ucl[i]=sp-stepper.r*atan2(drot.imag,drot.real)# unwound cable length = radius of approx. circle
      pe[i]=evolvent(t=-ucl[i]/stepper.r, 
                    t0=sp/stepper.r+stepper.ang0-copysign(pi/2,stepper.r), 
                    r=abs(stepper.r)
                    ) + stepper.C
      plt.plot(pe[i].real,pe[i].imag,'k+')
      plt.plot(pt[i].real,pt[i].imag,'r+')
    newx,newy=intersect_circles(pt[0].real,pt[0].imag,ucl[0],pt[1].real,pt[1].imag,ucl[1])
    p_guess=newx+1j*newy
    plt.plot(p_guess.real,p_guess.imag,'kx')
    error=abs(pe[0]-pe[1])
    print(f'{error = :6.2e}, new {p_guess = }')
    if error<1e-6: break
  return p_guess
  
pos=calc_pos(steppers,stepper_positions)
    
#sp_y0=stepper_y.stepper_pos(0+0j)
#pey1=evolvent(-sp_y/-r,t0=sp_y/-r+stepper_y.ang0+pi/2,r=r)+stepper_y.cx+1j*stepper_y.cy
#plt.plot(pey1.real,pey1.imag,'k+')
stepper_x.plot(plt.gca(),pos,'g-',plot_evolvent=True,plot_circle=True)
stepper_y.plot(plt.gca(),pos,'b-',plot_evolvent=True,plot_circle=True)


plt.plot(l[:i_p+1].real,l[:i_p+1].imag,'r-',lw=1,zorder=-3)

plt.xlim(-600,200)
plt.ylim(-200,600)
plt.show() 
plt.close()

def save_gif_animation(image_generator,output_file,*args,**kwargs):
    durations=list(image_generator('duration',*args,**kwargs))
    imgs=image_generator('img',*args,**kwargs)
    img=next(imgs)
    img.save(fp=output_file, format='GIF', append_images=imgs,
                 save_all=True, duration=durations, loop=0)
    
def show_animation_frames(image_generator,frames=None):
    for i,frame in enumerate(image_generator('fig')):
        if (frame==None) or  (i in frames):
            display(frame)
            
def gen_ani(returntype='fig',evolvent=True):  #fig,duration,img
  frame=0
  def snapshot(fig=None,duration=200,params=None):
      nonlocal frame #frame counter
      if fig==None: fig=plt.gca()
      if params==None: params=dict() #create a new local dict, because dict is mutable
  #    ax.set_title(f'frame #{frame}, duration:{duration}')
      framelabel=fig.text(0.,0., f'{frame}')
      if returntype=='fig':
        yield fig
      elif returntype=='duration':
        yield duration
      elif returntype=='img':
        with io.BytesIO() as img_buf:  
          fig.savefig(img_buf, format='png')
          yield Image.open(img_buf)
      else:
        raise Exception(f'Wrong value for "returntype": gen_any(returntype={returntype})\n'
                        "(must be [ 'fig' | 'duration' | 'img' ] )")
        return
      frame+=1
      framelabel.remove()
      return
  for i in range(n):
    plt.close()
    fig=plt.figure(figsize=(5,5))
    plt.gca().set_aspect('equal')
    stepper_x=stepper(Cx.real,Cx.imag,0,r,L=Lx)
    stepper_x.plot(plt.gca(),l[i],'g-',plot_evolvent=evolvent)
    stepper_y=stepper(Cy.real,Cy.imag,0,-r,L=Ly)
    stepper_y.plot(plt.gca(),l[i],'b-',plot_evolvent=evolvent)
    plt.plot(l[:i+1].real,l[:i+1].imag,'r-',lw=1,zorder=-3)
    plt.xlim(-600,200)
    plt.ylim(-200,600)
    duration=200
    if i in (0,n-1): duration=400
    yield from snapshot(fig,duration=duration)
  plt.close()
  return
#show_animation_frames(gen_ani,[1,2,3,4,5,10,15,21,30,40,50])
save_gif_animation(gen_ani,'hingebot.gif',evolvent=False)
print(f'saved')
