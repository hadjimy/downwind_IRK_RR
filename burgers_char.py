def plot_soln(N=4096):
    #Solution of Burger's equation by characteristics
    import numpy as np
    import pylab as pl

    T=0.16

    #Grid:
    L=1.
    dx=L/N; x=np.arange(N)*dx;

    #Initial condition:
    u0 = lambda x: 1.5+np.sin(2.*np.pi*x)

    tol = 1.e-10; #Accuracy of Newton iteration

    U=np.zeros(np.size(x))

    for i,X in enumerate(x):
      xi=X                  # Initial guess
      delta=1.
      while delta>tol:
        xinew = xi - (xi+u0(xi)*T-X)/(1+T*np.cos(xi))
        delta=abs(xinew-xi)
        xi=xinew
      U[i]=u0(xi)

    #pl.clf()
    pl.plot(x,U,'-k',linewidth=3)
    pl.draw()
