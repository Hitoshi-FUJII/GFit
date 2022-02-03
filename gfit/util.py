import numpy as np

class Table:
    def __init__(self,values,columns,rounds=None,space=2,indent=0,right=True):
        self.values  = values
        self.columns = columns
        if rounds is None: rounds = [5]*len(columns)
        self.rounds  = rounds
        self.right   = right
        self.space   = space
        self.indent  = indent
        
    
    def __repr__(self):
        return self.get_table()
    
    
    def v2s(self,v,r):
        if   type(v) is np.int64: vstr = f"{v}"
        elif type(v) is int     : vstr = f"{v}"
        else                    : vstr = f"{v:.{r}f}"
        return " "*self.indent + vstr
    
    
    def maxrow(self,val,c,r):
        lis = [len(c)]
        for v in val:
            lis.append(len(self.v2s(v,r)))
        return max(lis)
        
        
    def maxrow_list(self):
        lis = []
        for i in range(len(self.columns)):
            lis.append(self.maxrow(self.values[i],self.columns[i],self.rounds[i]))
        return lis
    
        
    def get_table(self):
        maxrow_lis = self.maxrow_list()
        table = ""
        for i,cname in enumerate(self.columns):
            if self.right: table += " "*(maxrow_lis[i]-len(cname)) + cname
            else         : table += cname + " "*(maxrow_lis[i]-len(cname)) 
            table += " "*self.space
        table += "\n"
            
        for i in range(len(self.values[0])):
            for j in range(len(self.columns)):
                vji = self.values[j][i]
                table += " "*(maxrow_lis[j]-len(self.v2s(vji,self.rounds[j])))
                table += self.v2s(vji,self.rounds[j])                       
                table += " "*self.space
            table += "\n"
        return table
        