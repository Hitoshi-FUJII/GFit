import numpy as np  
from scipy.special import digamma
    
    
class KM:
    def __init__(self,x,y,pi,mu,v):
        print("=== KM (k-means method)===")
        self.x  = x
        self.y  = y
        self.pi = pi   # mixture ratio
        self.mu = mu   # peak position
        self.v  = v    # variance
    
    
    def update(self):
        # === E step ===
        c = np.array([np.argmin(np.abs(self.x-m)) for m in self.mu])
        b = (c[:-1] + c[1:])//2
        b0 = 0
        r = np.zeros([len(self.x),len(self.mu)])
        for j,b1 in enumerate(b):
            b1 = int(b1)
            r[b0:b1,j] = 1
            b0 = b1
        r[b0:,-1] = 1

        # === M step ===
        nk = np.sum(r*self.y[:,None],axis=0)
        self.pi = nk/np.sum(self.y)
        self.mu = np.sum(r*self.x[:,None]*self.y[:,None],axis=0)/nk
        self.v  = np.sum(r*(self.x[:,None]-self.mu)**2*self.y[:,None],axis=0)/nk
        

        
class EM:
    def __init__(self,x,y,pi,mu,v):
        print("=== EM (EM algorithm)===")
        self.x  = x
        self.y  = y
        self.pi = pi   # mixture ratio
        self.mu = mu   # peak position
        self.v  = v    # variance
        

    def update(self):
        # === E step ===    
        u = np.log(self.pi) - 0.5*np.log(self.v) - 0.5/self.v*(self.x[:,None]-self.mu)**2
        u = self.overflow_handling(u)
        sumexp = np.sum(np.exp(u),axis=1)
        r = np.exp(u)/sumexp[:,None]
        
        # === M step ===
        nk = np.sum(r*self.y[:,None],axis=0)
        self.pi = nk/np.sum(self.y)
        self.mu = np.sum(r*self.x[:,None]*self.y[:,None],axis=0)/nk
        self.v  = np.sum(r*(self.x[:,None]-self.mu)**2*self.y[:,None],axis=0)/nk       
   

    def overflow_handling(self,u):    
        umax = np.max(u,axis=1)
        udif = u - umax[:,None]
        udifmax = np.max(udif)
        if udifmax > 709:
            print(f"!!! Risk of overflow !!! {udifmax} > 709" )
            udif[udif>709] = 709
        return udif
    
    

    
class VB:
    def __init__(self,x,y,pi,mu,v,nd):
        print("=== VB (Variational Bayes)===")
        self.x  = x
        self.y  = y
        self.pi = pi   # mixture ratio
        self.mu = mu   # peak position
        self.v  = v    # variance
        
        self.c   = nd/len(x)  # coefficient to convert from spectrum to frequency (pseudo histgram).
                
        nk = nd*pi
        self.alpha = nk    
        self.beta  = nk
        self.m     = mu                 
        self.a     = 0.5*nk         
        self.b     = 0.5*nk*v

        
    def update(self):
        # === E step ===
        alpha_sum = np.sum(self.alpha)
        u = -0.5*( np.log(self.b) - digamma(self.a) \
          + 2*digamma(alpha_sum) - 2*digamma(self.alpha) + 1./self.beta ) \
          - 0.5*self.a/self.b*(self.x[:,None]-self.m)**2
        u = self.overflow_handling(u)
        sumexp = np.sum(np.exp(u),axis=1)
        r = np.exp(u)/sumexp[:,None]
       
        # === M step ===
        nk = np.sum(r*self.y[:,None],axis=0)*self.c
        xm = np.sum(r*self.y[:,None]*self.x[:,None],axis=0)*self.c/nk
        xv = np.sum(r*self.y[:,None]*(self.x[:,None]-xm)**2,axis=0)*self.c/nk

        self.alpha = nk + 1
        self.beta  = nk + 1
        self.m     = nk * xm / self.beta
        self.a     = 0.5 * nk + 1
        self.b     = 0.5 * nk * xv \
                   + 0.5 * nk / self.beta * xm**2 + 1 
        
        self.pi = self.alpha/np.sum(self.alpha)
        self.mu = self.m
        self.v  = self.b/self.a
              
            
    def overflow_handling(self,u):    
        umax = np.max(u,axis=1)
        udif = u - umax[:,None]
        udifmax = np.max(udif)
        if udifmax > 709:
            print(f"!!! Risk of overflow !!! {udifmax} > 709" )
            udif[udif>709] = 709
        return udif    
    
    