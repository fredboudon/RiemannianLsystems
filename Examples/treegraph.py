
class TreeNode:
  def __init__(self, values,n, meancoef):
    import numpy as np
    minv, maxv = np.min(values),np.max(values)
    if n == 0:
      _meancoef = 0.5
    elif n == 1:
      _meancoef = 1-meancoef
    else:
      _meancoef = meancoef        
    self.value = minv + (maxv-minv)*_meancoef
    if len(values) >= 2:
       self.left = TreeNode(values[:len(values)//2], -1,meancoef )
       self.right = TreeNode(values[len(values)//2:], 1, meancoef)
    else:
       self.left = None
       self.right = None
  def __repr__(self):
    return "TreeNode("+repr(self.value)+','+repr(self.left)+','+repr(self.right)+')'
  def depth(self):
    if isinstance(self.left,TreeNode):
      return max(self.left.depth(),self.right.depth())+1
    else : return 1

def Tree(values, meancoef = 0.5):
    return TreeNode(list(sorted(values)),0,meancoef)
