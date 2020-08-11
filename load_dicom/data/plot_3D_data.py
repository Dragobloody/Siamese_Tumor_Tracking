import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
from pydicom.filereader import dcmread
import numpy as np
 
def plot_3d(image = None, body = None, roi = None, gt = None, pred = None, threshold_i = 1500, threshold_b = 0, threshold_r = 0):    
    # Position the scan upright, 
    # so the head of the patient would be at the top facing the camera   
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    g, y = [], []
    if image is not None: 
        g = np.zeros(image.shape, dtype = np.uint8)
        y = np.zeros(image.shape, dtype = np.uint8)
        p = image
        p = p[:,:,::-1]
        verts_p, faces_p = measure.marching_cubes_classic(p, threshold_i)
        # Fancy indexing: `verts[faces]` to generate a collection of triangles
        mesh_p = Poly3DCollection(verts_p[faces_p], alpha=0.1)    
        mesh_p.set_facecolor([0.0, 0.0, 0.0])
        ax.add_collection3d(mesh_p)
        
        ax.set_xlim(0, p.shape[0])
        ax.set_ylim(0, p.shape[1])
        ax.set_zlim(0, p.shape[2])
        
    if body is not None: 
        g = np.zeros(body.shape, dtype = np.uint8)
        y = np.zeros(body.shape, dtype = np.uint8)
        b = body
        b = b[:,:,::-1]
        verts_b, faces_b = measure.marching_cubes_classic(b, threshold_b)
        # Fancy indexing: `verts[faces]` to generate a collection of triangles
        mesh_b = Poly3DCollection(verts_b[faces_b], alpha=0.1)    
        mesh_b.set_facecolor([0.5, 0.5, 1])
        ax.add_collection3d(mesh_b)
        
        ax.set_xlim(0, b.shape[0])
        ax.set_ylim(0, b.shape[1])
        ax.set_zlim(0, b.shape[2])
        
    if roi is not None: 
        g = np.zeros(roi.shape, dtype = np.uint8)
        y = np.zeros(roi.shape, dtype = np.uint8)
        r = roi
        r = r[:,:,::-1]       
        verts_r, faces_r = measure.marching_cubes_classic(r, threshold_r)       
        # Fancy indexing: `verts[faces]` to generate a collection of triangles
        mesh_r = Poly3DCollection(verts_r[faces_r], alpha=0.1)    
        mesh_r.set_facecolor([0.5, 0.5, 0])
        ax.add_collection3d(mesh_r)
        
        ax.set_xlim(0, r.shape[0])
        ax.set_ylim(0, r.shape[1])
        ax.set_zlim(0, r.shape[2]) 
    
    if gt is not None:       
        gt = np.round(gt).astype(np.int16) 
        gt[2] = g.shape[2] - gt[2]
        g[gt[0], gt[1], gt[2]] = 1
        verts_g, faces_g = measure.marching_cubes_classic(g, 0)       
        # Fancy indexing: `verts[faces]` to generate a collection of triangles
        mesh_g = Poly3DCollection(verts_g[faces_g], alpha=0.1)    
        mesh_g.set_facecolor([0, 1, 0])
        ax.add_collection3d(mesh_g)
        
        ax.set_xlim(0, g.shape[0])
        ax.set_ylim(0, g.shape[1])
        ax.set_zlim(0, g.shape[2]) 
    
    if pred is not None:       
        pred = np.round(pred).astype(np.int16)
        pred[2] = y.shape[2] - pred[2]
        y[pred[0], pred[1], pred[2]] = 1
        verts_y, faces_y = measure.marching_cubes_classic(y, 0)       
        # Fancy indexing: `verts[faces]` to generate a collection of triangles
        mesh_y = Poly3DCollection(verts_y[faces_y], alpha=0.1)    
        mesh_y.set_facecolor([1, 0, 0])
        ax.add_collection3d(mesh_y)
        
        ax.set_xlim(0, y.shape[0])
        ax.set_ylim(0, y.shape[1])
        ax.set_zlim(0, y.shape[2]) 

    plt.show()
    

plot_3d(image = imgs[0], roi = labs[0], threshold_i = 0)


