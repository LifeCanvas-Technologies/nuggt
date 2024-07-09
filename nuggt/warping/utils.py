import json 
import numpy as np 

def read_points_json(json_path):
    '''
    Reads reference and moving points from json file 

    Parameters:
        json_path : str
            Path to the json file containing the correspondence points 

    Returns:
        (np.ndarray, np.ndarray): tuple of fixed points, moving points 
    '''

    with open(json_path, "r") as f:
        data = json.load(f)

    fixed_pts = np.asarray(data['reference']).astype('float32')
    moving_pts = np.asarray(data['moving']).astype('float32')
    return fixed_pts, moving_pts

def write_points_json(json_name, fixed_pts, moving_pts):
    """
    Writes fixed and moving correspondence points to json file. 

    Parameters: 
        json_name : str
            Path to the json to be written
        fixed_pts : np.ndarray
            Fixed (reference) points array of shape (N,D), where N is number of points and D is the dimension
        moving_pts : np.ndarray
            Moving points array of shape (N,D) corresponding to fixed_pts
    
    Returns:
        None
    """
    data = {}
    data['reference'] = fixed_pts.tolist()
    data['moving'] = moving_pts.tolist()
    
    json_object = json.dumps(data)
    with open(json_name, "w") as outfile:
        outfile.write(json_object)

def rescale_points(pts, scale_factor):
    """
    Rescale points by a scale factor

    Parameters:
        pts : np.ndarray | array | tuple 
        scale_factor : tuple

    Returns:
        np.ndarray : rescaled points 
    """
    if type(pts) != np.ndarray:
        pts = np.asarray(pts)
    if len(pts.shape) == 1:
        pts = np.expand_dims(pts,0)
    if pts.shape[1] != len(scale_factor):
        raise ValueError("Points dimension %d does not match scale factor dimension %d"%(pts.shape[1],len(scale_factor)))

    pts_rescaled = pts.astype('float')
    for i in range(len(scale_factor)):
        pts_rescaled[:,i] *= scale_factor[i] 
    return pts_rescaled 

def create_regular_grid(arr_shape, grid_spacing):
    """
    Create an evenly-spaced grid for a given array, and return the points of the grid.

    Parameters:
        arr_shape : tuple of int 
            d-tuple of the shape of array 
        grid_spacing : tuple of int 
            d-tuple (same length of arr_shape) of the grid_spacing in each dimension

    Returns:
        grid : np.ndarray
            array of shape N x d representing each point on the grid 
    """

    num_dims = len(arr_shape)
    num_pts = tuple([int(arr_shape[i]/grid_spacing[i]) for i in range(num_dims)])
    xs = tuple([np.linspace(0,arr_shape[idx]-1,num_pts[idx]) for idx in range(num_dims)])
    grid = np.stack([X.ravel() for X in np.meshgrid(*xs, indexing='ij')], axis=1)
    return grid

def crop_grid(grid, original_shape, zrange=None, yrange=None, xrange=None):
    '''
    Crop a grid based on specified ranges in each dimension.

    Parameters:
        grid (ndarray): The input grid.
        original_shape (tuple): The original shape of the full fixed image. 
        zrange (tuple, optional): The range of values to crop in the z dimension. Defaults to None.
        yrange (tuple, optional): The range of values to crop in the y dimension. Defaults to None.
        xrange (tuple, optional): The range of values to crop in the x dimension. Defaults to None.

    Returns:
        ndarray: The cropped grid.
        zr, yr, xr (tuple): the range of values of the cropped grid in the fixed frame in z, y, x
    '''
    # Get the true grid spacing of the grid (in pixels) 
    linspaces = [np.linspace(0, original_shape[i], grid.shape[i + 3]) for i in range(3)]
    grid_spacing = [linspaces[i][1] - linspaces[i][0] for i in range(3)]
    
    # compute the range in fixed frame
    xargs = np.argwhere((grid[0] >= zrange[0] - grid_spacing[0]) * (grid[0] < zrange[1] + grid_spacing[0]) *
                        (grid[1] >= yrange[0] - grid_spacing[1]) * (grid[1] < yrange[1] + grid_spacing[1]) *
                        (grid[2] >= xrange[0] - grid_spacing[2]) * (grid[2] < xrange[1] + grid_spacing[2]))
    args = [[xargs[:, 2].min(), xargs[:, 2].max()],
            [xargs[:, 3].min(), xargs[:, 3].max()],
            [xargs[:, 4].min(), xargs[:, 4].max()]]

    zr = [int(linspaces[0][args[0][0]]), int(linspaces[0][args[0][1]])]
    yr = [int(linspaces[1][args[1][0]]), int(linspaces[1][args[1][1]])]
    xr = [int(linspaces[2][args[2][0]]), int(linspaces[2][args[2][1]])]

    new_grid = grid[:, :, :, args[0][0]:args[0][1] + 1, args[1][0]:args[1][1] + 1, args[2][0]:args[2][1] + 1]
    new_shape = (zr[1] - zr[0], yr[1] - yr[0], xr[1] - xr[0])
    # print("New fixed image shape:", new_shape)
    # print("New fixed range: z", zr[0], zr[1])
    # print("New fixed range: y", yr[0], yr[1])
    # print("New fixed range: x", xr[0], xr[1])
    
    # translate the new_grid values
    new_grid[2] -= xrange[0]
    new_grid[1] -= yrange[0]
    new_grid[0] -= zrange[0]

    return new_grid, zr, yr, xr 

def convert_array_to_set(arr):
    """
    Convert an array to a set of tuples.

    Parameters:
        arr (ndarray): The input array.

    Returns:
        set: The set of tuples.
    """
    return set([tuple(x) for x in arr])

def get_chunk_coords(z_range, y_range, x_range, chunks, overlap=(0,0,0)):
    '''
    Utility to get the coordinates of chunks, with a given range of coordinates, overlaps, and chunking.
    Different usage than lct_napari.zarr_converter.utils.get_chunk_coords 

    Parameters
    ----------
    z_range : list of int
        The starting and ending z coordinate (first)
    y_range : list of int
        The starting and ending y coordinate (second)
    x_range : list of int
        The starting and ending x coordinate (third)
    chunks : tuple of int
        The chunk size to calculate the coordinate chunking (DOES NOT include overlap)
    overlap : tuple of int
        The amount of overlap between each chunk 

    Returns
    --------
    coords : list of list of list of ints
        Each element of the list is a 3 x 2 list of lists containing the z_range, y_range, x_range for each chunk
    '''
    oz, oy, ox = overlap 
    coords = [[[i,i+chunks[0]],[j,j+chunks[1]],[k,k+chunks[2]]] \
                        for i in np.arange(z_range[0], z_range[1], chunks[0]) \
                        for j in np.arange(y_range[0], y_range[1], chunks[1]) \
                        for k in np.arange(x_range[0], x_range[1], chunks[2])]
    c = np.array(coords)

    c[:,0,0] -= oz; c[:,1,0] -= oy; c[:,2,0] -= ox
    c[:,0,1] += oz; c[:,1,1] += oy; c[:,2,1] += ox

    # filter out of bounds coordinates
    c[c[:,0,1]>z_range[1],0,1] = z_range[1]
    c[c[:,1,1]>y_range[1],1,1] = y_range[1]
    c[c[:,2,1]>x_range[1],2,1] = x_range[1]

    c[c[:,0,0]<z_range[0],0,0] = z_range[0]
    c[c[:,1,0]<y_range[0],1,0] = y_range[0]
    c[c[:,2,0]<x_range[0],2,0] = x_range[0]

    return c.tolist()