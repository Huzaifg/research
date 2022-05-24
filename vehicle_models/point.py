class Point:
    def __init__(self,initx,inity):
        self.x = initx
        self.y = inity

    def getx(self):
        return self.x

    def gety(self):
        return self.y

    def negx(self):
        return -(self.x)

    def negy(self):
        return -(self.y)

    def __str__(self):
        return 'x=' + str(self.x) + ', y=' + str(self.y)

    def halfway(self,target):
        midx = (self.x + target.x) / 2
        midy = (self.y + target.y) / 2
        return Point(midx, midy)

    def distance(self,target):
        xdiff = target.x - self.x
        ydiff = target.y - self.y
        dist = math.sqrt(xdiff**2 + ydiff**2)
        return dist

    def reflect_x(self):
        return Point(self.negx(),self.y)

    def reflect_y(self):
        return Point(self.x,self.negy())

    def reflect_x_y(self):
        return Point(self.negx(),self.negy())

    def slope_from_origin(self):
        if self.x == 0:
            return None
        else:
            return self.y / self.x

    def slope(self,target):
        if target.x == self.x:
            return None
        else:
            m = (target.y - self.y) / (target.x - self.x)
            return m


    def get_eq(self,target):
        c = -(self.slope(target)*self.x - self.y)
        def fun(t):
            return self.slope(target)*t + c
        return fun