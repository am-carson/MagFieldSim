import numpy as np
import matplotlib.pyplot as plt
class Field(object):
    def __init__(self,Dim=(10,10,10),corners=[[0,5],[10,5]],curr=1):
        self.coords= np.indices(Dim)
        self.create_wire(corners,curr)
        self.views = [('XZ',   (90, -90, 0)),
         ('YZ',    (0, -90, 0)),
         ('XY',    (0,   0, 0)),
         ('PRO',   (45, 45, 0))]
    def create_wire(self,corners,curr):
        self.corners=corners
        self.curr = curr 
        self.wire = np.zeros(self.coords[0,:,:,:].shape, dtype=bool)
        
        for i in range(len(corners)-1):
            x = (((self.coords[0] >=corners[i][0] )&(self.coords[0] <= corners[i+1][0])) & ((self.coords[1] >= corners[i][1])&(self.coords[1] <= corners[i+1][1])) & (self.coords[2]  == self.wire.shape[2]/2))
            self.wire = self.wire | x
        
    def gen_field(self):
        self.field = ((self.coords[0] < 5) & (self.coords[1] == 2) & (self.coords[2]  == 2)) | ((self.coords[0] == 5) & (self.coords[1] >= 2)& (self.coords[1] < 8) & (self.coords[2]  == 2))
        return self.field
    def get_voxels(self):
        return self.coords 

    def setup_plot(self):
        layout = [['XY'],  ['YZ'],   ['XZ'],['PRO']]
        fig, self.ax = plt.subplot_mosaic(layout, subplot_kw={'projection': '3d'},figsize=(100,100))

    
    def plot_wire(self):
        self.gen_field()
       
        colors = np.empty(self.wire.shape, dtype=object)
        colors[self.wire] = 'blue'
        
        for plane, angles in self.views:
            self.ax[plane].voxels(self.wire, facecolors=colors, edgecolor='k')

        
    def plot_field(self):
        u_0 = 4*3.14*0.00000001
        x, y, z = np.meshgrid(np.arange(0, 10, 1),np.arange(0, 10, 1),np.arange(0, 10, 1))
        xz, yz, zz = np.meshgrid(np.arange(0, 10, 1),np.arange(0, 10, 1),0)
        xy, yy, zy = np.meshgrid(np.arange(0, 10, 1),0,np.arange(0, 10, 1))
        xx, yx, zx = np.meshgrid(0,np.arange(0, 10, 1),np.arange(0, 10, 1))
        self.sections= np.where(self.wire == True)
       
        bx = np.zeros((10, 10, 10))
        by = np.zeros((10, 10, 10))
        bz = np.zeros((10, 10, 10))
        spare = np.zeros((10, 10, 10))
        for i in range(len(self.sections[0])-1):
            dl = np.asarray([self.sections[0][i+1]-self.sections[0][i], self.sections[1][i+1]-self.sections[1][i], self.sections[2][i+1]-self.sections[2][i]])
            
            rx =  (x-self.sections[2][i])*(1-dl[2])
            ry =  (y-self.sections[1][i])*(1-dl[1])
            rz =  (z-self.sections[0][i])*(1-dl[0])
            r = (rx**2+ry**2+rz**2)**0.5
            
            
            bx = bx +((u_0*self.curr* ry)/r**3) # i have to use numpy arrays
            by = by +((u_0*self.curr* rx)/r**3)
            bz = bz +((u_0*self.curr* rz)/r**3)
            
        # Make the direction data for the arrows
        print(bz.shape)
        
        self.ax["XY"].quiver(spare, xz,yz,spare, bx, -1*by, length=0.5, normalize=True)
        self.ax["YZ"].quiver(zx, spare,yx,bz, spare, -1*by, length=0.5, normalize=True)
        self.ax["XZ"].quiver(zy, xy,spare,bz, bx, spare, length=0.5, normalize=True)
        self.ax["PRO"].quiver(z, x,y,bz, bx, -1*by, length=0.5, normalize=True)
    def draw_plot(self):
        
        for plane, angles in self.views:
            self.ax[plane].set_xlabel('x')
            self.ax[plane].set_ylabel('y')
            self.ax[plane].set_zlabel('z')
            self.ax[plane].set_proj_type('ortho')
            self.ax[plane].view_init(elev=angles[0], azim=angles[1], roll=angles[2])
            self.ax[plane].set_box_aspect(None, zoom=1.25)

            label = f'{plane}\n{angles}'
                
        plt.show()