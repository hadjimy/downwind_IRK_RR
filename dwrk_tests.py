"""
Tests of various simple flux-differencing FV schemes for advection and Burgers.
In particular, includes downwind implicit WENO-RK.

The main purpose is to test the downwind RK schemes with large SSP coefficient.
The biggest challenge is getting the nonlinear solver to converge for large CFL number
and with shocks.  Really I should use something more sophisticated than SciPy; 
in the future I plan to use PETSc for the nonlinear solve.  The right approach would
be to implement the downwind RK in PetClaw and abandon this script.
"""

import numpy as np
import matplotlib.pyplot as pl
from recon import weno
#from pyclaw.limiters.reconstruct import weno
from scipy.optimize import fsolve
from scipy.optimize.nonlin import newton_krylov

def run(method,eqn='advection',limiter='weno',cflnum=0.9,r=8.,N=128,IC='sin',BC='periodic',solver='krylov',guesstype='BE',ls='-k',plotall=False,squarenum=30):
    """
    Run advection or Burgers equation test.  Options:

    method = 
        be
        itrap
        dwrk             - optimal 2nd order downwind implicit SSP RK
        fe               - Forward Euler

    eqn = 'advection' or 'burgers'
    limiter = 'weno' or 'vanleer'
    r is a parameter used to define the downwind RK method; if another method is used
        then r is irrelevant (see the paper for details)

    IC = sin, sinp, gaussian, or square (initial condition)

    solver = fsolve or krylov (always tries fsolve as a last resort anyway)
    
    guesstype = BE, BE_fine, itrap  (different approaches to finding a good initial guess for the
                nonlinear solve)
    """

    #Set up grid
    dx=1./N;
    nghost = 3; N2=N+2*nghost;
    x=np.linspace(-(nghost-0.5)*dx,1.+(nghost-0.5)*dx,N2)
    t=0.

    if IC=='sin':
        q0 = np.sin(2*np.pi*x)
        myaxis=(0.,1.,-1.1,1.1)
    elif IC=='sinp':
        q0 = np.sin(2*np.pi*x)+1.5
        myaxis=(0.,1.,0.4,2.6)
    elif IC=='gaussian':
        q0 = np.exp(-200*(x-0.3)**2)    # Smooth
    elif IC=='square':
        q0 = 1.0*(x>0.0)*(x<0.5)        # Discontinuous
        myaxis=(0.,1.,-0.1,1.1)
    elif IC=='BL-Riemann':               # Buckley-Leverett Riemann problem
        q0 = 1.*(x>0.5) + 0.5*(x<0.)
        print q0[0]
        myaxis=(0.,1.,-0.1,1.1)
    else:
        raise Exception('Unrecognized value of IC.')

    q=q0.copy();
    #pl.plot(x,q); pl.draw(); pl.hold(False)

    if eqn == 'advection': 
        flux = lambda q : q
        dt=cflnum*dx;
        T=1.
    elif eqn == 'burgers': 
        flux = lambda q : 0.5*q**2
        dt=cflnum*dx/np.max(q);
        T=0.16
    elif eqn == 'buckley-leverett': 
        a=1./3.
        flux = lambda q : q**2/(q**2 + a*(1.-q)**2)
        maxvel = 2*a*0.25
        dt=cflnum*dx/maxvel
        T=0.25
    else: raise Exception('Unrecognized eqn')


    while t<T:

        # If we're nearly at the end, choose the step size to reach T exactly
        if dt>T-t: dt=T-t

        # Take one step using the chosen method

        if method=='be':  # Backward Euler
            nonlinearf = lambda(y) : be(q,dt,dx,y,flux,limiter=limiter,BC=BC)
            qnew=fsolve(nonlinearf,q)
            q[:]=qnew[:]

        elif method=='itrap':  # Implicit Trapezoidal rule
            nonlinearf = lambda(y) : itrap_solve(q,dt,dx,y,flux,limiter=limiter,BC=BC)
            y1=fsolve(nonlinearf,q)
            y1hat=np.empty([1,len(y1)])
            y1hat[0,:]=y1
            ql,qr=limit(y1hat,limiter)
            q[1:] = q[1:] - dt/dx *  (flux(qr[0,1:])-flux(qr[0,:-1]))

        elif method=='dwrk':  # 2nd-order Downwind Runge-Kutta
            guess=setguess(q,dt,dx,flux,guesstype,r)
            nonlinearf = lambda(y) : dwrk_solve(q,dt,dx,y,flux,r,limiter=limiter,BC=BC)
            if solver=='fsolve':
                Y,info,ier,mesg=fsolve(nonlinearf,guess,full_output=1)
                if ier>1: return ier,mesg
            elif solver=='krylov':
                try: Y=newton_krylov(nonlinearf,guess,method='lgmres',maxiter=100)
                except: 
                    print 'Trying fsolve as last resort'
                    Y,info,ier,mesg=fsolve(nonlinearf,guess,full_output=1)
                    if ier>1: return ier,mesg
            Y=np.reshape(Y,(2,-1),'C')
            y2=np.empty([1,len(q)])
            y2[0,:]=Y[1,:]

            ql,qr=limit(y2,limiter)

            q[1:] = y2[0,1:] - dt/dx *  (flux(qr[0,1:])-flux(qr[0,:-1]))/r

        elif method=='fe':
            qq=np.empty([1,len(q)])
            qq[:,0]=q
            ql,qr=limit(qq,limiter)
            q[1:] = q[1:] - dt/dx * (flux(qr[1:,0])-flux(qr[:-1,0]))

        else:
            print 'error: unrecognized method'

        if BC=='periodic':
            q[0:nghost]  = q[-2*nghost:-nghost] # Periodic boundary
            q[-nghost:]  = q[nghost:2*nghost] # Periodic boundary
        #Else fixed Dirichlet; do nothing

        t=t+dt

        if plotall:
            pl.plot(x,q,ls,linewidth=3,markersize=7,markevery=int(N/N)); pl.hold(True); 
            pl.axis(myaxis)
            pl.title(str(t)); pl.draw(); pl.hold(False)
            print t

    pl.plot(x,q,ls,linewidth=3,markersize=7,markevery=int(N/squarenum)); pl.hold(True);
    pl.axis(myaxis); pl.title(str(t)); pl.draw()

def be(q,dt,dx,y,flux,limiter='weno',BC='periodic'):
    """
    This is the function to pass to fsolve to take a step on the advection
    equation using backward Euler.

    For a correctly computed BE step:
    y = q + dt*f(y)
    """
    nghost=3
    qq=np.empty([1,len(y)])
    qq[0,:]=y
    f=np.zeros(len(y))
    ql,qr=limit(qq,limiter)
    if BC=='periodic':
        qr[0,0:nghost]  = qr[0,-2*nghost:-nghost] # Periodic boundary
        qr[0,-nghost:]  = qr[0,nghost:2*nghost] # Periodic boundary
    else:
        qr[0,0:nghost]  = y[0:nghost] # Periodic boundary
        qr[0,-nghost:]  = y[-nghost:] # Periodic boundary

    f[1:] = - dt/dx * (flux(qr[0,1:])-flux(qr[0,:-1]))
    if BC=='periodic':
        f[0] = -dt/dx * (flux(qr[0,0])-flux(qr[0,-1]))
    else:
        pass
        #f[0] = f[1]
    return q - y + f

def itrap_solve(q,dt,dx,y,flux,limiter='weno',BC='periodic'):
    """
    This is the function to pass to fsolve to compute the first stage for the advection
    equation using the implicit trapezoidal method.
    """
    nghost=3
    qq=np.empty([1,len(y)])
    qq[0,:]=y
    f=np.zeros(len(y))
    ql,qr=limit(qq,limiter)

    if BC=='periodic':
        qr[0,0:nghost]  = qr[0,-2*nghost:-nghost] # Periodic boundary
        qr[0,-nghost:]  = qr[0,nghost:2*nghost] # Periodic boundary
    else:
        qr[0,0:nghost]  = y[0:nghost] # Periodic boundary
        qr[0,-nghost:]  = y[-nghost:] # Periodic boundary

    f[1:] = - dt/dx * (flux(qr[0,1:])-flux(qr[0,:-1]))
    f[0] = -dt/dx * (flux(qr[0,0])-flux(qr[0,-1]))
    return q - y + f/2

def dwrk_solve(q,dt,dx,y,flux,r=4.,limiter='weno',BC='periodic'):
    """
    This is the function to pass to fsolve to compute the first stage for the advection
    equation using the implicit downwind Runge-Kutta methods
    """
    y=np.reshape(y,(2,-1),'C')
    
    nghost=3
    ql,qr=limit(y,limiter)

    if BC=='periodic':
        ql[:,0:nghost]  = ql[:,-2*nghost:-nghost] # Periodic boundary
        ql[:,-nghost:]  = ql[:,nghost:2*nghost] # Periodic boundary 
        qr[:,0:nghost]  = qr[:,-2*nghost:-nghost] # Periodic boundary
        qr[:,-nghost:]  = qr[:,nghost:2*nghost] # Periodic boundary
    else:
        ql[:,0:nghost]  = y[:,0:nghost] # Periodic boundary
        ql[:,-nghost:]  = y[:,-nghost:] # Periodic boundary 
        qr[:,0:nghost]  = y[:,0:nghost] # Periodic boundary
        qr[:,-nghost:]  = y[:,-nghost:] # Periodic boundary

    f=np.empty(y.shape)
    rhs=np.empty(y.shape)
    f[0,1:] = - dt/dx * (flux(qr[0,1:])-flux(qr[0,:-1]))
    f[1,:-1] = - dt/dx * (flux(ql[1,:-1])-flux(ql[1,1:]))
    if BC=='periodic':
        f[0,0]  = - dt/dx * (flux(qr[0,0]) -flux(qr[0,-1 ]))
        f[1,-1 ] = - dt/dx * (flux(ql[1,1])-flux(ql[1,0]))
    else:
        f[0,0]  =  0.
        f[1,-1 ] = 0.

    rhs[0,:] = 2./(r*(r-2))*q + (2./r)*(y[0,:]+f[0,:]/r) \
                  + (r**2-4*r+2)/(r*(r-2))*(y[1,:]+f[1,:]/r)
    rhs[1,:] = y[0,:]+f[0,:]/r

    rhs=np.reshape(rhs,-1,'C')
    y=np.reshape(y,-1,'C')

    return rhs-y

def vanleer(q,eps=1.e-14):
    dum=np.empty(q.shape)
    dup=np.empty(q.shape)

    dup[:,:-1]=np.diff(q,1,1)
    dum[:,1:]=np.diff(q,1,1)

    delta = (np.sign(dup)+np.sign(dum))*(dup*dum)/(abs(dup+dum)+eps)

    ql = q-0.5*delta
    qr = q+0.5*delta
    return ql,qr


def setguess(q,dt,dx,flux,guesstype,r):
    c1 = 1. - 2./r
    c2 = 1. - 1./r
    if guesstype=='FE':
        #This initial guess is only good for r=8; need to generalize
        #But it doesn't seem to help anyway.
        y1guess=q.copy()
        y1guess[1:] = y1guess[1:] - c1*dt/dx * (flux(y1guess[1:])-flux(y1guess[:-1]))
        y2guess=q.copy()
        y2guess[1:] = y2guess[1:] - c2*dt/dx * (flux(y2guess[1:])-flux(y2guess[:-1]))
        guess=np.hstack([y1guess,y2guess])
    elif guesstype=='BE':
        myweno = lambda(y) : be(q,c1*dt,dx,y,flux)
        y1guess=fsolve(myweno,q)
        myweno = lambda(y) : be(y1guess,dt/r,dx,y,flux)
        y2guess=fsolve(myweno,y1guess)
        guess=np.hstack([y1guess,y2guess])
    elif guesstype=='BE_fine':
        yy=q.copy()
        for i in range(int(2*r)-2):
            myweno = lambda(y) : be(yy,dt/r/2.,dx,y,flux)
            yy=fsolve(myweno,yy)
        y1guess=yy
        myweno = lambda(y) : be(y1guess,dt/r,dx,y,flux)
        y2guess=fsolve(myweno,y1guess)
        guess=np.hstack([y1guess,y2guess])
    elif guesstype=='itrap':
        myweno = lambda(y) : itrap_solve(q,c1*dt,dx,y,flux)
        y1=fsolve(myweno,q)
        y1hat=np.empty([1,len(y1)])
        y1hat[0,:]=y1
        ql,qr=weno(5,y1hat)
        y1guess=q.copy()
        y1guess[1:] = y1guess[1:] - dt/dx *  (flux(qr[0,1:])-flux(qr[0,:-1]))
        y1guess[0] = y1guess[0]   - dt/dx *  (flux(qr[0,0])-flux(qr[0,-1]))
        myweno = lambda(y) : be(y1guess,dt/r,dx,y,flux)
        y2guess=fsolve(myweno,y1guess)
        guess=np.hstack([y1guess,y2guess])
    else:
        guess=np.hstack([q,q])

    return guess

def limit(q,limiter):
    if   limiter=='weno':    ql,qr=weno(5,q)
    elif limiter=='vanleer': ql,qr=vanleer(q)
    elif limiter=='donor':   ql=q.copy(); qr=q.copy()
    return ql,qr


#=========================================================
# Functions to reproduce results from the paper
#=========================================================

def fig1a():
    x=np.linspace(0.,1.,1000)
    q0 = 1.0*(x>0.0)*(x<0.5)        # Discontinuous
    pl.plot(x,q0,'-k',linewidth=3)
    pl.hold(True)

    for method,ls in zip(('be','itrap','dwrk'),('--k',':k','-sk')):
        print method
        run(method,ls=ls,solver='fsolve',IC='square',limiter='donor')

    pl.title('')
    pl.legend(['Exact','Backward Euler','Trapezoidal RK','Downwind RK'])

def fig1b():
    x=np.linspace(0.,1.,1000)
    q0 = 1.0*(x>0.0)*(x<0.5)        # Discontinuous
    pl.plot(x,q0,'-k',linewidth=3)

    for method,ls in zip(('be','itrap','dwrk'),('--k',':k','-sk')):
        print method
        run(method,ls=ls,solver='fsolve',IC='square',limiter='donor',cflnum=8.)

    pl.title('')
    pl.legend(['Exact','Backward Euler','Trapezoidal RK','Downwind RK'])

def fig2():
    from burgers_char import plot_soln
    plot_soln()
    pl.hold(True)

    for method,ls in zip(('be','itrap','dwrk'),('--k',':k','-sk')):
        print method
        run(method,'burgers',cflnum=6.5,N=512,ls=ls,solver='krylov',IC='sinp',squarenum=512)

    pl.title('')
    pl.legend(['Exact','Backward Euler','Trapezoidal RK','Downwind RK'])
    pl.axis((0.69,0.796,0.5,2.5))


def fig3():
    for cflnum,ls in zip((0.9,6.5),('--k',':k')):
        print cflnum
        run('dwrk','burgers',cflnum=cflnum,N=512,ls=ls,solver='krylov',IC='sinp')

    pl.title('')
    pl.legend(['CFL=0.9','CFL=6.5'])
    pl.axis((0.69,0.796,0.5,2.5))


def convtest():
    err=np.zeros([3,5])
    cflnum=8.
    for i,method in enumerate(['be','itrap','dwrk']):
        #for j,cflnum in enumerate([64,32,16,8,4]):
        for j,N in enumerate([16,32,64,128,256]):
            print method, N
            err[i,j] = run(method,cflnum=cflnum,r=cflnum*1.1,N=N,solver='fsolve')
            print err
    return err



