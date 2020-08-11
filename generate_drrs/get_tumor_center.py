import numpy as np


def get_mask_center(mask):
    horizontal_indicies = np.where(np.any(mask, axis=(1,2)))[0]
    vertical_indicies = np.where(np.any(mask, axis=(0,2)))[0]
    diagonal_indicies = np.where(np.any(mask, axis=(0,1)))[0]
        
    x1, x2 = horizontal_indicies[[0, -1]]
    y1, y2 = vertical_indicies[[0, -1]]
    z1, z2 = diagonal_indicies[[0, -1]]
    x = x1 + (x2 - x1 + 1) // 2       
    y = y1 + (y2 - y1 + 1) // 2       
    z = z1 + (z2 - z1 + 1) // 2       
    
    return x, y, z

       
def get_relative_center(mask):
    x_0, y_0, z_0 = [d // 2 for d in mask.shape]
    x, y, z = get_mask_center(mask)
    
    return x_0 - x, y_0 - y, z_0 - z


def get_patient_relative_center(masks):
    center_x, center_y, center_z = [], [], []
    for i in range(masks.shape[0]):
        x, y, z = get_relative_center(masks[i])
        center_x.append(x)
        center_y.append(y)
        center_z.append(z)
        
    return np.round(np.mean(center_x)), np.round(np.mean(center_y)), np.round(np.mean(center_z))
    
