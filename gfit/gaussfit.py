import matplotlib.pyplot as plt
import numpy as np

from gfit.optimizer import KM,EM,VB
from gfit.util import Table


class GMM:
    def __init__(self,k=2,itr=10,algo="em",seed=None,fig=False,nd=1e6):
        self.k    = k
        self.itr  = itr
        self.algo = algo
        self.fig  = fig
        self.nd   = nd
        self.seed = seed
        np.random.seed(seed=self.seed)
        self.opt  = None
    
        # parameters for scaled data.
        self.pi = None   # mixture ratio
        self.mu = None   # peak position
        self.v  = None   # variance

        # parameters for original scale.
        self.pi_ori = None   # mixture ratio
        self.mu_ori = None   # peak position
        self.v_ori  = None   # variance
    
    
    def __repr__(self):
        rep = f"GMM(k={self.k},itr={self.itr},algo='{self.algo}',"
        rep += f"seed={self.seed},fig={self.fig},nd={self.nd})\n"
        if self.pi is not None:
            height  = self.pi / np.sqrt(2*np.pi*self.v)
            values  = [np.arange(self.k)+1,self.mu,height,self.pi,self.v]
            columns = ["Peak ID","Position(mu)","Height","Ratio(pi)","Variance(v)"]
            rep += Table(values,columns).get_table()
        return rep
        
        
    def set_options(self,k=None,itr=None,algo=None,seed=None,fig=None,nd=None):
        if itr  is not None: self.itr  = itr
        if algo is not None: self.algo = algo
        if fig  is not None: self.fig  = fig
        if nd   is not None: self.nd   = nd
        if k is not None:
            if k is not self.k:
                self.k  = k
                self.pi = None
                self.mu = None
                self.v  = None
        if seed is not None:
            self.seed = seed
            np.random.seed(seed=self.seed)
        return self

    
    def initialize(self,y):   
        n = len(y)
        xs = np.arange(n)/(n-1)
        ys = self.normalize(y)
        if self.pi is None: self.pi = np.array([1/self.k]*self.k)
        if self.mu is None: self.mu = np.sort(np.random.choice(xs,size=self.k,replace=False))
        if self.v  is None: self.v  = 0.01 * np.ones(self.k)
        
        if   self.algo is "km": self.opt = KM(xs,ys,self.pi,self.mu,self.v)
        elif self.algo is "em": self.opt = EM(xs,ys,self.pi,self.mu,self.v)
        elif self.algo is "vb": self.opt = VB(xs,ys,self.pi,self.mu,self.v,self.nd)
        else                  : raise ValueError('"algo" must be "km", "em" or "vb".')


    def area(self,y):
        return np.sum(y-np.min(y))/(len(y)-1)
    
        
    def normalize(self,y):
        return (y-np.min(y))/self.area(y)
    
        
    def fit(self,y):
        self.initialize(y)
        for _ in range(self.itr):
            self.opt.update()
            if self.fig:
                mg = self.make_gmm() 
                title = f"Itr = {_+1},  R2 score = {self.score(self.opt.y,mg):.5f}"
                self.plot(self.opt.x,self.opt.y,yp=None,title=title,size=(8,1))
                
        self.pi,self.mu,self.v = self.opt.pi,self.opt.mu,self.opt.v    
        return self
        
    
    def gauss(self,k):
        # parameters to a normalized Gaussian spectrum
        x  = self.opt.x
        pi = self.opt.pi[k]
        mu = self.opt.mu[k]
        v  = self.opt.v[k]
        return pi/np.sqrt(2*np.pi*v)*np.exp(-0.5/v*(x-mu)**2)
        
        
    def make_gmm(self):
        mg = 0
        for k in range(self.k):
            mg += self.gauss(k)
        return mg    
            
        
    def score(self,y,yp):
        e2  = np.sum((y - yp)**2)
        sy2 = np.sum((y - np.mean(y))**2)
        r2 = 1. - e2/sy2
        return r2 


    def plot(self,x,y,yp=None,title=None,size=None):
        xmin,xmax = x[0],x[-1]
        mu = self.opt.mu * (xmax-xmin) + xmin
        
        if yp is not None: self.show_params(x,y)
        plt.figure(figsize=size)
        plt.plot(x,y,color="k",lw=0.5)
        for k in range(self.k):
            g = self.gauss(k)
            plt.plot(x,g*self.area(y)+np.min(y),lw=2)
            plt.vlines(mu[k],np.min(y),np.max(y),color="k",lw=2,ls="dotted")
        plt.title(title)
        plt.show()     
        
        if yp is not None:
            print(f"R2 score = {self.score(y,yp):.5f}")
            self.plot2(x,y,yp)
            
        
    def plot2(self,x,y,yp,title=None,size=None):
        plt.figure(figsize=size)
        plt.plot(x,y,color="k",lw=0.5)
        plt.plot(x,yp,color="r",lw=2)
        plt.hlines(np.min(y),np.min(x),np.max(x),color="k",lw=1.0)
        plt.title(title)
        plt.show()
        

    def params(self,x,y):
        xmin,xmax = x[0],x[-1]
        pi = self.pi
        mu = self.mu * (xmax-xmin) + xmin
        v  = self.v * (xmax-xmin)**2
        s  = self.area(y)*(xmax-xmin)
        h  = pi * s / np.sqrt(2*np.pi*v)
        return pi,mu,v,h
    
    
    def show_params(self,x,y):
        pi,mu,v,h = self.params(x,y)
        values  = [np.arange(self.k)+1,mu,h,pi,v]
        columns = ["Peak ID","Position(mu)","Height","Ratio(pi)","Variance(v)"]
        print(Table(values,columns))
        
    
    def curve(self,y):
        mg = self.make_gmm()
        return mg * self.area(y) + np.min(y)
    
