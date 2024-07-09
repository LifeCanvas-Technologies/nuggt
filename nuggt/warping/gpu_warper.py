import multiprocessing as mp
import numpy as np
import zarr
import torch
import torch.nn.functional as F
from tqdm import tqdm
import time
import psutil 
from functools import partial 
import os 
import bisect


from .utils import get_chunk_coords

from .warper import Warper
from .utils import read_points_json, rescale_points


LOCAL_INDICES_CACHE = None
MOVING_IMG_SHARED = None


class TorchGridSampler:
    """
    Class to sample a regular grid using Pytorch. 

    Attributes: 
        values (np.ndarray) : 
        grid (np.ndarray) : 
        img_shape (tuple) : 

    Methods:
        __init__
        fit
        sample_grid
    """

    def __init__(self, values, grid, img_shape, mode="bilinear"):
        """
        Constructor for TorchGridSampler.

        Parameters:
            values : np.ndarray 
                array of size to map each coordinate to 
            grid : np.ndarray 
                array of size (1,1,z_dim,y_dim,x_dim) or (1,1,y_dim,x_dim)
            img_shape : array-like
                shape of the array whose grid is being interpolated. 
            mode : str
                "bilinear" or "nearest" -- mode of interpolation for warping 

        Notes:
            PyTorch can only support 4D or 5D grids (corresponding to 2D, 3D images)
        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.values = torch.from_numpy(values).float().to(device)
        if type(grid) == np.ndarray:
            self.grid = torch.from_numpy(grid).float().to(device)
        elif type(grid) == torch.Tensor:
            self.grid = grid 
        
        if type(img_shape) != torch.Tensor:
            self.img_shape = torch.from_numpy(np.array(img_shape)).float().to(device)
        else:
            self.img_shape = img_shape 
        
        self.mode = mode 
            
    def fit(self):
        """
        Interpolates values based on the input grid. 

        Returns:
            ([values_z], values_y, values_x) : tuple of torch.Tensors
                Tensor of size 
        """        
        return tuple([F.grid_sample(value, self.grid, align_corners=True, mode=self.mode)[0] for value in self.values])
 
    def sample_grid(self, padding=2):
        """
        Samples a grid.

        Parameters: 
            padding : int, optional
                Amount of padding on each side of transformed grid to encompass the entire image 

        Returns:
            start : int 
            stop : int 
            new_grid : torch.Tensor
        """
        new_values = self.fit() #(*chunks) tensor

        transformed_start = []; transformed_stop = []
        for value in new_values:
            transformed_start.append(torch.floor(value.min() - padding).to(torch.int16))
            transformed_stop.append(torch.ceil(value.max() + padding).to(torch.int16))
        transformed_start = torch.stack(transformed_start)
        transformed_stop = torch.stack(transformed_stop)

        # Assign points 
        if torch.any(transformed_stop < 0) or torch.any(torch.greater_equal(transformed_start, self.img_shape)):
            return None
        else:
            start = torch.maximum(torch.zeros_like(transformed_start), transformed_start).to(torch.float)
            stop = torch.minimum(self.img_shape, transformed_stop).to(torch.float)
            if torch.any(torch.less_equal(stop - 1, start)):
                return None
            else:
                new_grid = torch.stack([(new_values[idx] - start[idx]) / (stop[idx]-start[idx]-1) * 2 - 1 for idx in np.flip(np.arange(len(new_values)))],
                                       dim = len(self.img_shape)+1)
                return start.to('cpu').numpy().astype('int'), stop.to('cpu').numpy().astype('int'), new_grid



class ZarrWarper:
    """
    Class for warping Zarrs based on pre-computed correspondences 
    """
    def __init__(self, moving_zarr_path, warped_zarr_path, pts_path, fixed_img_size, 
                 grid_values_path=None, moving_voxel_size=(1,1,1), fixed_voxel_size=(1,1,1),
                 mode='bilinear'):
        """
        Constructor

        Parameters:
        mode : str
                "bilinear" or "nearest" -- mode of interpolation for warping 
        """
        if isinstance(moving_zarr_path, zarr.core.Array):
            self.moving_img = moving_zarr_path
        else:
            self.moving_img = zarr.open(moving_zarr_path)
        self.chunks = self.moving_img.chunks
        self.fixed_img_size = fixed_img_size 
        if isinstance(warped_zarr_path, zarr.core.Array):
            self.warped_img = warped_zarr_path
        else:
            self.warped_img = zarr.create(shape=fixed_img_size,
                                        chunks=self.chunks,
                                        dtype=self.moving_img.dtype,
                                        compressor=self.moving_img.compressor,
                                        store=zarr.NestedDirectoryStore(warped_zarr_path),
                                        overwrite=True)
        if isinstance(pts_path, dict):
            self.fixed_pts = pts_path["reference"]
            self.moving_pts = pts_path["moving"]
        else:
            self.fixed_pts, self.moving_pts = read_points_json(pts_path)
        self.fixed_voxel_size = fixed_voxel_size
        self.moving_voxel_size = moving_voxel_size 
        self.num_dims = len(self.fixed_img_size)
        self.grid_values_path = "" if grid_values_path is None else grid_values_path
        self.mode = mode 

    
    def _find_fixed_z_chunks(self, values, max_moving_z_size, padding = 2):
        """
        Gets the z chunks in the fixed image for when we need to do mega-chunking 
        """
        z_chunk = self.chunks[0]
        fixed_z_nodes = np.linspace(0, self.fixed_img_size[0], values.shape[3]) # z coordinates of each node in grid values
        fixed_z_start = z_chunk
        fixed_z_end = z_chunk 
        fixed_z_bounds_max = [] # List of fixed z coordinates representing the actual end for each chunk
        moving_z_bounds_max = [] # List of moving z coordinates that represent the actual end for each chunk (to actually be read in)
        moving_z_bounds_min = []
        moving_z_max = max_moving_z_size
        while fixed_z_end < self.fixed_img_size[0]:
            # Find minimum moving z coordinate of starting point 
            if len(moving_z_bounds_min) == 0:
                moving_z_bounds_min.append(0)
            else:
                idx = bisect.bisect_left(fixed_z_nodes, fixed_z_start)-1 # find first index in grid_values that won't be greater
                moving_z_min = max(int(values[0,:,:,idx].min()-padding), 0)
                moving_z_bounds_min.append(moving_z_min)
                # Update the max possible moving z coordinate for memory
                moving_z_max = min(self.moving_img.shape[0],moving_z_min + max_moving_z_size)

            if self.fixed_img_size[0] - fixed_z_start <= z_chunk:
                fixed_z_end = self.fixed_img_size[0]
                temp_moving_max = self.moving_img.shape[0]
            else:
                # For loop to find which multiple of z_chunk size we can use without overloading moving memory 
                z_iterable = np.arange(fixed_z_start, self.fixed_img_size[0], z_chunk)
                if self.fixed_img_size[0] % z_chunk != 0:
                    z_iterable = np.append(z_iterable, self.fixed_img_size[0])
                for i in z_iterable:
                    idx = bisect.bisect_left(fixed_z_nodes, i) # find the first index in grid_values that is greater than the chunk multiple
                    max_moving_z_in_chunk = min(self.moving_img.shape[0],values[0,:,:,idx].max())
                    if moving_z_max >= max_moving_z_in_chunk:
                        fixed_z_end = i
                        temp_moving_max = max_moving_z_in_chunk
                    else:
                        break
            moving_z_bounds_max.append(min(int(np.ceil(temp_moving_max + padding)), self.moving_img.shape[0]))
            fixed_z_bounds_max.append(fixed_z_end)
            fixed_z_start = fixed_z_end # reset start bound for next chunk 

        fixed_bounds_total = [[0, fixed_z_bounds_max[0]]]
        for i in range(len(fixed_z_bounds_max)-1):
            fixed_bounds_total.append([fixed_z_bounds_max[i], fixed_z_bounds_max[i+1]]) 
        moving_bounds_total = [[moving_z_bounds_min[i], moving_z_bounds_max[i]] for i in range(len(moving_z_bounds_max))]

        return fixed_bounds_total, moving_bounds_total


    def _warp_grid(self, grid_spacing, smooth=2):
        """
        Warp a grid.

        Parameters:
            grid_spacing : tuple 
            smooth : int 

        Returns:
            values : np.ndarray
                Array of shape (3,1,1,Z_fixed,Y_fixed,X_fixed). 
        """

        # Create regular grid to be warped 
        print("Making and warping grid...")
        scaled_warped_img_size = tuple(rescale_points(self.fixed_img_size, self.fixed_voxel_size)[0])
        num_pts = tuple([int(scaled_warped_img_size[i]/grid_spacing[i]) for i in range(self.num_dims)])
        xs = tuple([np.linspace(0,scaled_warped_img_size[idx],num_pts[idx]) for idx in range(self.num_dims)])
        grid = np.stack([X.ravel() for X in np.meshgrid(*xs, indexing='ij')], axis=1)

        # Create warper and approximator 
        warper = Warper(rescale_points(self.fixed_pts, self.fixed_voxel_size),
                        rescale_points(self.moving_pts, self.moving_voxel_size), 
                        smooth=smooth)
        appx_warper = warper.approximate(*xs)
        grid_values = appx_warper(grid)

        grid_values = [np.reshape(grid_values[:,dim], num_pts)/self.moving_voxel_size[dim] \
                        for dim in range(self.num_dims)]
        grid_values = np.expand_dims(np.expand_dims(np.asarray(grid_values),1),1)
        return grid_values

    def _warp_chunk(self, values, coord, zrange=None):
        """
        Warp a single zarr chunk and writes to disk. 

        Parameters:
            values : torch.Tensor 
            coord : np.ndarray 

        Returns: 
            None 
        """

        global LOCAL_INDICES_CACHE
        global MOVING_IMG_SHARED

        chunks = np.asarray(self.chunks)
        img_shape = np.asarray(self.fixed_img_size)
        chunk_shape = np.array([b - a for a, b in coord])

        # Get z, y, x indices for each pixel
        if np.all(chunks == chunk_shape): 
            local_indices = LOCAL_INDICES_CACHE # faster 
        else:
            local_indices = np.indices(chunk_shape)
        global_indices = np.empty_like(local_indices)  # faster than zeros_like
        for i in range(global_indices.shape[0]):
            global_indices[i] = local_indices[i] + coord[i][0]
        global_indices = torch.from_numpy(global_indices).float().to('cuda')

        # Make the grid with normalized coordinates [-1, 1]
        grid = torch.stack([(global_indices[idx] / float(img_shape[idx] - 1)) * 2 - 1 for idx in np.flip(np.arange(len(img_shape)))], 
                            dim=len(img_shape)).unsqueeze(0)

        # Sample the transformation grid
        tgs = TorchGridSampler(values, grid, self.moving_img.shape, mode=self.mode)
        result = tgs.sample_grid()
        if result is not None:
            moving_start, moving_stop, moving_grid = result
            if zrange is None:
                moving_data = MOVING_IMG_SHARED[tuple(np.s_[a:b] for a, b in zip(moving_start, moving_stop))]

                if not np.any(moving_data):
                    interp_chunk = np.zeros(chunk_shape, self.warped_img.dtype)
                else:
                    # interpolate the moving data
                    moving_data = moving_data.reshape((1, 1, *moving_data.shape)).astype('float')
                    moving_data_tensor = torch.from_numpy(moving_data).float().to('cuda')
                    interp_chunk = F.grid_sample(moving_data_tensor, moving_grid, mode=self.mode, align_corners=True).to('cpu').numpy()[0, 0]       
        else:
            interp_chunk = np.zeros(chunk_shape, self.warped_img.dtype)

        self.warped_img[tuple([slice(coord[idx][0],coord[idx][1]) for idx in range(len(img_shape))])] = \
            interp_chunk.astype(self.warped_img.dtype)

    def _warp_zarr(self, values, num_workers=None):
        """
            Warp a zarr using the GPU based on an already computed grid

            Parameters:
                values : np.ndarray
                    array of shape (1,1,Z,Y,X)
                num_workers : int 


            Returns:
                None
            """
        global LOCAL_INDICES_CACHE
        global MOVING_IMG_SHARED

        # Cache local indices
        LOCAL_INDICES_CACHE = np.indices(self.chunks)
        memory_buffer = 0.95 # fraction of memory to use 

        moving_img_memory = self.moving_img.size*self.moving_img.itemsize # Image size in bytes
        max_memory = psutil.virtual_memory().available * memory_buffer # Current workstation memory

        print("Moving image size:",moving_img_memory/1e9,"GB")
        print("Available memory:", max_memory/1e9, "GB")

        if moving_img_memory < max_memory:
            # If we can fit the entire moving image into RAM (fastest option)
            start = time.time()
            MOVING_IMG_SHARED = zarr.create(shape=self.moving_img.shape,
                                        chunks=self.chunks,
                                        dtype=self.moving_img.dtype,
                                        compressor=self.moving_img.compressor)
            MOVING_IMG_SHARED[:] = self.moving_img # see if parallel assignment is faster 
            print("Assigning zarr to shared memory took",time.time()-start,"seconds")

            # Get the chunking for the entire fixed image
            coords = get_chunk_coords([0,self.fixed_img_size[0]], 
                                      [0,self.fixed_img_size[1]],
                                      [0,self.fixed_img_size[2]],
                                      self.chunks)
            
            if num_workers is None:
                for coord in tqdm(coords, total=len(coords)):
                    coord = np.asarray(coord)
                    self._warp_chunk(values, coord, zrange=None)
            else:
                fxn = partial(self._warp_chunk, values, zrange=None)
                with mp.Pool(num_workers) as pool:
                    list(tqdm(pool.imap(fxn, coords), total=len(coords)))
            print("Done!")
        else:
            print("Moving image is too large to fit into memory, warping in large chunks...")
            # print("This functionality has NOT been finished yet, please try again later.")

            start = time.time()
            # Get the size of the z chunks for moving image that should be loaded
            # num_z_chunks = int(np.ceil(moving_img_memory / (max_memory * memory_buffer)))
            # max_moving_z_chunk = np.linspace(0, self.moving_img.shape[0], num_z_chunks + 1, dtype=int)[1]
            max_moving_z_chunk =  int(max_memory * memory_buffer / moving_img_memory * self.moving_img.shape[0])
            print(max_moving_z_chunk)
            # For each max moving z, we find a fixed z range that can be entirely warped in that.
            fixed_z_ranges, moving_z_ranges = self._find_fixed_z_chunks(values, max_moving_z_chunk)
            print("Warping in %d chunks" % len(fixed_z_ranges))
            
            for fixed_z_range, moving_z_range in zip(fixed_z_ranges, moving_z_ranges):
                print("Warping moving z {}-{} to fixed z {}-{}".format(moving_z_range[0], moving_z_range[1],
                                                                       fixed_z_range[0], fixed_z_range[1]))
                MOVING_IMG_SHARED = zarr.create(shape=(moving_z_range[1]-moving_z_range[0], *self.moving_img.shape[1:]),
                                            chunks=self.chunks,
                                            dtype=self.moving_img.dtype,
                                            compressor=self.moving_img.compressor)
                MOVING_IMG_SHARED[:] = self.moving_img[moving_z_range[0]:moving_z_range[1]]
                print("Assigning zarr to shared memory took",time.time()-start,"seconds")
                
                # Update the grid values to subtract by the moving range 
                chunk_values = values.copy()
                chunk_values[0] -= moving_z_range[0]

                # Get the coordinates for this section
                coords = get_chunk_coords([fixed_z_range[0], fixed_z_range[1]], 
                                          [0,self.fixed_img_size[1]],
                                          [0,self.fixed_img_size[2]],
                                          self.chunks)
                if num_workers is None:
                    for coord in tqdm(coords, total=len(coords)):
                        coord = np.asarray(coord)
                        self._warp_chunk(chunk_values, coord, zrange=None)
                else:
                    fxn = partial(self._warp_chunk, chunk_values, zrange=None)
                    with mp.Pool(num_workers) as pool:
                        list(tqdm(pool.imap(fxn, coords), total=len(coords)))
            print("Done")
 
    def warp(self, grid_spacing, smooth=2, num_workers=None):
        '''
        Use Thin Plate Splines to warp image.
        '''
        if not isinstance(self.grid_values_path, np.ndarray):
            if self.grid_values_path == '' or not os.path.exists(self.grid_values_path):
                grid_values = self._warp_grid(grid_spacing, smooth)
                if self.grid_values_path is not None:
                    np.save(self.grid_values_path, grid_values)
            else:
                print("Grid values already exist at {}. Loading...".format(self.grid_values_path))
                grid_values = np.load(self.grid_values_path)
        else:
            grid_values = self.grid_values_path
        
        print("Warping image...")
        self._warp_zarr(grid_values, num_workers=num_workers)


def warp_image(moving_zarr, 
               warped_zarr_path, 
               fixed_pts, 
               moving_pts, 
               fixed_img_size,
               moving_voxel_size,
               fixed_voxel_size,
               num_workers = psutil.cpu_count(logical=False),
               grid_spacing = (320,)*3,
               smooth = 2,
               mode = "bilinear",
               grid_values_path = None):
    
    pts_dict = {"reference": fixed_pts,
                "moving" : moving_pts}
    zarrwarper = ZarrWarper(moving_zarr, warped_zarr_path, pts_dict, fixed_img_size, 
                            grid_values_path=grid_values_path, moving_voxel_size=moving_voxel_size, fixed_voxel_size=fixed_voxel_size,
                            mode = mode)
    zarrwarper.warp(grid_spacing, smooth=smooth, num_workers=num_workers)


def warp_nparray(moving_arr, 
                 fixed_arr_shape, 
                 grid_values,
                 mode = "bilinear"):
    """
    Use the GPU to warp a single numpy array 
    """
    # Get z, y, x indices for each pixel
    local_indices = np.indices(fixed_arr_shape)
    global_indices = torch.from_numpy(local_indices).float().to('cuda')

    # Make the grid with normalized coordinates [-1, 1]
    grid = torch.stack([(global_indices[idx] / float(fixed_arr_shape[idx] - 1)) * 2 - 1 for idx in np.flip(np.arange(len(fixed_arr_shape)))], 
                        dim=len(fixed_arr_shape)).unsqueeze(0)

    # Sample the transformation grid
    tgs = TorchGridSampler(grid_values, grid, moving_arr.shape, mode=mode)
    result = tgs.sample_grid()
    if result is not None:
        moving_start, moving_stop, moving_grid = result

        if not np.any(moving_arr):
            interp_chunk = np.zeros(fixed_arr_shape, moving_arr.dtype)
        else:
            # interpolate the moving data
            moving_data = moving_data.reshape((1, 1, *moving_arr.shape)).astype('float')
            moving_data_tensor = torch.from_numpy(moving_arr).float().to('cuda')
            interp_chunk = F.grid_sample(moving_data_tensor, moving_grid, mode=mode, align_corners=True).to('cpu').numpy()[0, 0]       
    else:
        interp_chunk = np.zeros(fixed_arr_shape, moving_arr.dtype)

    return interp_chunk