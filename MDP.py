class MDP(object):
  def __init__(self, state_w=10, state_h=10, omega=0.1, gamma=0.8, rf=None):
    self.V = np.array([[i+j*state_w for i in range(state_w)] for j in range(state_h)])
    self.actions = [[-1,0], [0,1], [1,0], [0,-1]]
    self.omega = omega
    self.gamma = gamma
    self.rf = rf
    self.policy = [[None for i in range(state_w)] for j in range(state_h)]

  def nextPos(self, s, a):
      return [s[0]+a[0], s[1]+a[1]]

  def P(self, s, sprime, a):
    """
    s: [x,y], initial state position, a list of two elements, x-axis and y-axis
    sprime: [x,y], next state position
    a: [x', y'], an action from the action list
    """
    next_pos = self.nextPos(s, a)
    # non-boundary state:
    if s[0] > 0 and s[0] < len(self.V)-1 and s[1] > 0 and s[1] < len(self.V[0])-1:
      if s == sprime:
        return 0
      if sprime == next_pos:
        return (1-self.omega+self.omega/4)
      return self.omega/4

    # upleft corner:
    elif s == [0,0]:
      # illegal state
      if sprime == [-1,0] or sprime == [0,-1]:
        return 0
      # action to move off the grid
      if a == [-1,0] or a == [0,-1]:
        if sprime == s:
          return 1-self.omega+self.omega/4+self.omega/4
        return self.omega/4
      else:
        if sprime == next_pos:
          return 1-self.omega+self.omega/4
        elif sprime == s:
          return self.omega/4+self.omega/4
        return self.omega/4

    # upright corner
    elif s == [0,len(self.V[0])-1]:
      # illegal state
      if sprime == [-1,len(self.V[0])-1] or sprime == [0,len(self.V[0])]:
        return 0
      # action to move off the grid
      if a == [-1,0] or a == [0,1]:
        if sprime == s:
          return 1-self.omega+self.omega/4+self.omega/4
        return self.omega/4
      else:
        if sprime == next_pos:
          return 1-self.omega+self.omega/4
        elif sprime == s:
          return self.omega/4+self.omega/4
        return self.omega/4

    # lowleft corner
    elif s == [len(self.V)-1,0]:
      # illegal state
      if sprime == [len(self.V)-1,-1] or sprime == [len(self.V),0]:
        return 0
      # action to move off the grid
      if a == [1,0] or a == [0,-1]:
        if sprime == s:
          return 1-self.omega+self.omega/4+self.omega/4
        return self.omega/4
      else:
        if sprime == next_pos:
          return 1-self.omega+self.omega/4
        elif sprime == s:
          return self.omega/4+self.omega/4
        return self.omega/4

    # lowright corner
    elif s == [len(self.V)-1, len(self.V[0])-1]:
      # illegal state
      if sprime == [len(self.V)-1, len(self.V[0])] or sprime == [len(self.V), len(self.V[0])-1]:
        return 0
      # action to move off the grid
      if a == [1,0] or a == [0,1]:
        if sprime == s:
          return 1-self.omega+self.omega/4+self.omega/4
        return self.omega/4
      else:
        if sprime == next_pos:
          return 1-self.omega+self.omega/4
        elif sprime == s:
          return self.omega/4+self.omega/4
        return self.omega/4

    # leftedge
    elif s[1] == 0:
      # illegal state
      if sprime[1] == -1:
        return 0
      # action to move off the grid
      if a == [0,-1]:
        if sprime == s:
          return 1-self.omega+self.omega/4
        else:
          return self.omega/4
      else:
        if sprime == next_pos:
          return 1-self.omega+self.omega/4
        return self.omega/4

    # upedge
    elif s[0] == 0:
      # illegal state
      if sprime[0] == -1:
        return 0
      # action to move off the grid
      if a == [-1,0]:
        if sprime == s:
          return 1-self.omega+self.omega/4
        else:
          return self.omega/4
      else:
        if sprime == next_pos:
          return 1-self.omega+self.omega/4
        return self.omega/4

    # rightedge
    elif s[1] == len(self.V[0])-1:
      # illegal state
      if sprime[1] == len(self.V[0]):
        return 0
      # action to move off the grid
      if a == [0,1]:
        if sprime == s:
          return 1-self.omega+self.omega/4
        else:
          return self.omega/4
      else:
        if sprime == next_pos:
          return 1-self.omega+self.omega/4
        return self.omega/4

    # downedge
    elif s[0] == len(self.V)-1:
      # illegal state
      if sprime[0] == len(self.V):
        return 0
      # action to move off the grid
      if a == [1,0]:
        if sprime == s:
          return 1-self.omega+self.omega/4
        else:
          return self.omega/4
      else:
        if sprime == next_pos:
          return 1-self.omega+self.omega/4
        return self.omega/4

  def valueIter(self, epsilon):
    # initialization
    for i in range(len(self.V)):
      for j in range(len(self.V[0])):
        V[i][j] = 0

    # estimation
    delta = float('inf')
    while delta > epsilon:
      delta = 0
      for i in range(len(self.V)):
        for j in range(len(self.V)):
          v = V[i][j]
          self.V[i][j], self.policy = self.optimalStateVal([i, j])
          delta = max(delta, abs(v - self.V[i][j]))

    return self.V, self.policy

  def optimalStateVal(self, s):
    bestPolicy = [0,0]
    bestValue = 0
    for a in self.actions:
      value = 0
      for move in [[-1,0], [0,1], [1,0], [0,-1], [0,0]]:
        sprime = self.nextPos(s, move)
        P_s_sprime_a = self.P(s, sprime, a)
        if P_s_sprime_a == 0:
          continue
        value += self.P(s, sprime, a)*(self.rf[sprime[0]][sprime[1]] + self.gamma*self.V[sprime[0]][sprime[1]])
      if value > bestValue:
        bestPolicy = a
        bestValue = value
    return bestValue, bestPolicy

  def heatMap(self):
    pass