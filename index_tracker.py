import numpy as np

class IndexTracker(object):
    def __init__(self, ax, X, Y, Z, extent, vmin = -300, vmax = 300):
        self.ax = ax
        self.X = X
        self.Y = np.flip(Y, 1)
        self.Z = np.flip(Z, 1)
        self.patients, rows, cols, self.slices = X.shape
        self.ind = self.slices//2
        self.patient = self.patients//2     
        self.extent = extent

        self.vmin = vmin
        self.vmax = vmax 

        self.im = self.ax.imshow(self.X[self.patient, :, :, self.ind], cmap='gray', extent=self.extent, vmin=self.vmin, vmax=self.vmax)
        self.im2 = self.ax.contour(self.Z[self.patient, :, :, self.ind], 0, linewidths=0.1, colors='r', extent=self.extent)
        self.im3 = self.ax.contour(self.Y[self.patient, :, :, self.ind], 0, linewidths=0.1, colors='b', extent=self.extent)

        self.update() 

    def onscroll(self, event):
        print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

 

    def onpress(self, event):
        if event.key == 'right':
            self.patient = (self.patient + 1) % self.patients
        elif event.key == 'left':
            self.patient = (self.patient - 1) % self.patients
        self.update()

       

    def onpress2(self, event):
        if event.key == 'up':
            self.patient = (self.patient + 10) % self.patients
        elif event.key == 'down':
            self.patient = (self.patient - 10) % self.patients
        self.update()

       

    def onpress3(self, event):
        if event.key == '8':
            self.vmin += 10
            self.vmax += 10
        elif event.key == '2':
            self.vmin -= 10
            self.vmax -= 10           

        self.update()

       

    def onpress4(self, event):
        if event.key == '4':
            self.vmin += 10
            self.vmax -= 10
        elif event.key == '6':
            self.vmin -= 10
            self.vmax += 10           

        self.update()

       

    def update(self):
        self.im.set_data(self.X[self.patient, :, :, self.ind])
        self.im.set_clim(vmin=self.vmin, vmax=self.vmax)
        self.im2.collections[0].remove()
        self.im2 = self.ax.contour(self.Z[self.patient, :, :, self.ind], 0, colors='r', extent=self.extent)
        self.im3.collections[0].remove()
        self.im3 = self.ax.contour(self.Y[self.patient, :, :, self.ind], 0, colors='b', extent=self.extent)
        self.ax.set_title('patient %s, slice %s, vmin %s, vmax %s' % (self.patient, self.ind, self.vmin, self.vmax))
        self.im.axes.figure.canvas.draw()

       
