"""align.py - an alignment tool


"""

import argparse
from copy import deepcopy
import logging
import json
import neuroglancer
import numpy as np
from neuroglancer import CoordinateSpace
import os
import psutil
import tifffile
import time
import webbrowser
import zarr 
from pathlib import Path

from nuggt.utils.warp import Warper
from nuggt.utils.ngutils import layer, seglayer, pointlayer
from nuggt.utils.ngutils import gray_shader, green_shader
from nuggt.utils.ngutils import soft_max_brightness, get_contrast_limits
from nuggt.warping import warp_image as gpu_warp_image
from nuggt.warping.warper import Warper as SparseWarper
from scipy.interpolate import RegularGridInterpolator

# Monkey-patch neuroglancer.PointAnnotationLayer to have a color

if not hasattr(neuroglancer.PointAnnotationLayer, "annotation_color"):
    from neuroglancer.viewer_state import  wrapped_property, optional, text_type
    neuroglancer.PointAnnotationLayer.annotation_color = \
        wrapped_property('annotationColor', optional(text_type))


def parse_args(raw_args=None):
    parser = argparse.ArgumentParser(description="Neuroglancer Aligner")
    parser.add_argument("--reference-images",
                        help="Paths to reference image files. Is a list of paths to zarrs or tiffs and can be arbitrarily long.",
                        nargs='+',
                        required=False)
    
    parser.add_argument('--edge-image',
                        help="Path to edge image file",
                        required=False)
    
    parser.add_argument("--moving-image",
                        help="Path to image file for image to be aligned",
                        required=False)
    
    parser.add_argument("--warped-zarr",
                        help="Path to zarr file to store warped image",
                        required=False)
    
    parser.add_argument("--segmentation",
                        help="Path to the segmentation file accompanying "
                        "the reference image.",
                        default=None) 
    
    parser.add_argument("--points",
                        help="Path to point-correspondence file for moving image",
                        required=False)

    parser.add_argument("--points-full",
                        help="Path to dense grid point-correspondence file used for warping",
                        required=True)
    
    parser.add_argument("--no-launch",
                        help="Don't launch browsers on startup",
                        default=False,
                        action="store_true")
        
    parser.add_argument("--ip-address",
                        help="IP address interface to use for http",
                        default=None,
                        required=False)
    
    parser.add_argument("--port",
                        type=int,
                        default=None,
                        help="Port # for http. Default = let program choose")
    
    parser.add_argument("--reference-voxel-size",
                        help="X, Y and Z size of voxels, separated by "
                        "commas, e.g. \"1.6,1.6,1.0\"",
                        default="1.0,1.0,1.0",
                       required=False)
    
    parser.add_argument("--moving-voxel-size",
                        help="X, Y and Z size of voxels, separated by "
                        "commas, e.g. \"1.6,1.6,1.0\"",
                        default="1.0,1.0,1.0",
                       required=False)
    
    parser.add_argument("--min-distance",
                        type=float,
                        default=1.0,
                        help="Minimum distance between any two annotations.")
    
    parser.add_argument("--n-workers",
                        type=int,
                        default= psutil.cpu_count(logical=False),
                        required=False,
                        help="# of workers to use during warping")
    
    parser.add_argument("--x-index",
                        help="The index of the x-coordinate in the alignment"
                             " points matrix, e.g. \"0\" if the x and z "
                        "axes were transposed. Defaults to 2.",
                        type=int,
                        default=2,
                       required=False)
    
    parser.add_argument("--y-index",
                        help="The index of the y-coordinate in the alignment"
                             " points matrix, e.g. \"0\" if the y and z "
                        "axes were transposed. Defaults to 1.",
                        type=int,
                        default=1,
                       required=False)
    
    parser.add_argument("--z-index",
                        help="The index of the z-coordinate in the alignment"
                             " points matrix, e.g. \"2\" if the x and z "
                        "axes were transposed. Defaults to 0.",
                        type=int,
                        default=0,
                       required=False)
    
    return parser.parse_args(raw_args)

class ViewerPair:
    """The viewer pair maintains two neuroglancer viewers

    We maintain a reference viewer for the reference image and a moving
    viewer for the image to be aligned. The reference viewer has the following
    layers:
    * reference: the reference image
    * alignment: the image to be aligned, in the reference coordinate frame
    * correspondence-points: the annotation points in the reference frame
    * edit: an annotation layer containing the point being edited

    The moving image viewer has the following layers:
    * image: the moving image
    * correspondence-points: the annotation points in the moving image
                             coordinate frame
    * edit: an annotation layer containing the point being edited
    """

    REFERENCE = "reference"
    EDGE = "edge"
    MOVING = "moving"
    ALIGNMENT = "alignment"
    CORRESPONDENCE_POINTS = "correspondence-points"
    IMAGE = "image"
    EDIT = "edit"
    SEGMENTATION = "segmentation"

    ANNOTATE_ACTION = "annotate"
    
    # BRIGHTER_ACTION = "brighter"
    # DIMMER_ACTION = "dimmer"
    
    CLEAR_ACTION = "clear"
    DONE_ACTION = "done-with-point"
    EDIT_ACTION = "edit"
    NEXT_ACTION = "next"
    PREVIOUS_ACTION = "previous"
    
    # REFERENCE_BRIGHTER_ACTION = "reference-brighter"
    # REFERENCE_DIMMER_ACTION = "reference-dimmer"
    
    
    REFRESH_ACTION = "refresh-view"
    
    REDO_ACTION = "redo"
    SAVE_ACTION = "save-points"
    TRANSLATE_ACTION = "translate-point"
    UNDO_ACTION = "undo"
    WARP_ACTION = "warp"
    

    BRIGHTER_KEY = "shift+equal" 
    CLEAR_KEY = "shift+keyc"
    DIMMER_KEY = "minus"       
    DONE_KEY = "shift+keyd"
    EDIT_KEY = "shift+keye"
    NEXT_KEY = "shift+keyn"
    PREVIOUS_KEY = "shift+keyp"
    
    REFERENCE_BRIGHTER_KEY = "shift+keya" 
    REFERENCE_DIMMER_KEY = "shift+keyx"

    
    REFRESH_KEY = "shift+keyr"
    SAVE_KEY = "shift+keys"
    REDO_KEY = "control+keyy"
    TRANSLATE_KEY = "shift+keyt"
    UNDO_KEY = "control+keyz"
    WARP_KEY = "shift+keyw"

    EDIT_ANNOTATION_COLOR="#FF0000"

    REFERENCE_SHADER = """
void main() {
  emitRGB(vec3(0, toNormalized(getDataValue()), 0));
}
    """

    ALIGNMENT_SHADER = """
void main() {
  emitGrayscale(toNormalized(getDataValue()));
}
        """

    IMAGE_SHADER="""
void main() {
  emitGrayscale(toNormalized(getDataValue()));
}
 """
    #
    # For multiprocessing - dictionaries keyed on id(self)
    #
    moving_images = {}
    warpers = {}
    alignment_buffers = {}

    def __init__(self, reference_images, edge_image, moving_image, warped_image, segmentation,
                 points_file, reference_voxel_size, moving_voxel_size,
                 n_workers=psutil.cpu_count(logical=False), min_distance=1.0,
                 x_index=2, y_index=1, z_index=0, dense_grid_points_file=None):
        if dense_grid_points_file is None:
            raise ValueError("--points-full is required")
        
        """Constructor

        :param reference_image: align to this image
        :param moving_image: align this image
        :param segmentation: the segmentation associated with the reference
        image or None if no segmentation.
        :param points_file: where to load and store points
        :param reference_voxel_size: a 3-tuple giving the X, Y and Z voxel
        size in nanometers.
        :param moving_voxel_size: the voxel size for the moving image
        :param min_distance: the minimum allowed distance between any two points
        :param n_workers: # of workers to use when warping
        """
        self.reference_images = reference_images
        self.edge_image = edge_image

        if self.edge_image is None:
            if self.reference_images is not None:
                self.reference_shape = self.reference_images[0].shape
            else:
                raise ValueError("Either edge_image or reference_images must be provided")
        elif self.reference_images is None:
            self.reference_shape = self.edge_image.shape
        else:
            self.reference_shape = self.reference_images[0].shape


        # path to moving image 
        if isinstance(moving_image, zarr.core.Array):
            self.moving_image_zarr = moving_image 
            self.moving_image = moving_image[:]
        elif isinstance(moving_image, np.ndarray):
            moving_img_path = Path(points_file).parent.parent / "moving_image.zarr"
            self.moving_image_zarr = zarr.create(moving_image.shape, 
                                                 chunks=(128,)*3, 
                                                 dtype=moving_image.dtype,
                                                 store = zarr.NestedDirectoryStore(moving_img_path),
                                                 overwrite = True)
            self.moving_image_zarr[:] = moving_image
            self.moving_image = moving_image
        self.alignment_image = warped_image
        self.segmentation = segmentation
        self.n_workers = n_workers
        self.moving_images[id(self)] = moving_image
        self.decimation = max(1, np.min(self.reference_shape) // 5)
        self.reference_viewer = neuroglancer.Viewer()
        self.moving_viewer = neuroglancer.Viewer()
        self.points_file = points_file
        self.dense_grid_points_file = dense_grid_points_file
        p = Path(dense_grid_points_file)
        self.dense_grid_points_file_corrected = str(p.parent / (p.stem + "_corrected" + p.suffix))
        self.warper = None
        self.reference_voxel_size = reference_voxel_size
        self.moving_voxel_size = moving_voxel_size
        self.edge_brightness = 13030
        self.moving_brightness = soft_max_brightness(self.moving_image, percentile=99)
        self.min_distance = min_distance
        self.load_points()
        self.load_dense_grid_points()
        self.init_state()
        self.init_warper()
        # self.refresh_brightness()
        
      
        self.x_index=x_index
        self.y_index=y_index
        self.z_index=z_index
        

    def load_points(self):
        """Load reference/moving points from the points file"""
        if not os.path.exists(self.points_file):
            self.reference_pts = []
            self.moving_pts = []
        else:
            with open(self.points_file, "r") as fd:
                d = json.load(fd)
            self.reference_pts = d[self.REFERENCE]
            self.moving_pts = d[self.MOVING]

    def load_dense_grid_points(self):
        """Load dense grid reference/moving points from the dense grid points file"""
        with open(self.dense_grid_points_file, "r") as fd:
            d = json.load(fd)
        self.dense_grid_ref_pts = d[self.REFERENCE]
        self.dense_grid_mov_pts_original = d[self.MOVING]
        # Use the corrected file if it exists, otherwise fall back to original
        if os.path.exists(self.dense_grid_points_file_corrected):
            with open(self.dense_grid_points_file_corrected, "r") as fd:
                dc = json.load(fd)
            self.dense_grid_mov_pts = dc[self.MOVING]
        else:
            self.dense_grid_mov_pts = list(self.dense_grid_mov_pts_original)

    def save_points(self):
        """Save reference / moving points to the points file"""
        with open(self.points_file, "w") as fd:
            json.dump({
                self.REFERENCE: self.reference_pts,
                self.MOVING: self.moving_pts
            }, fd, indent=2)
        if len(self.dense_grid_ref_pts) > 0 and len(self.reference_pts) > 0:
            auto_ref = np.array(self.dense_grid_ref_pts)
            auto_mov = np.array(self.dense_grid_mov_pts_original)  # always from original
            manual_ref = np.array(self.reference_pts)
            manual_mov = np.array(self.moving_pts)

            # Infer the regular grid axes from the dense ref points
            zs = np.unique(auto_ref[:, 0])
            ys = np.unique(auto_ref[:, 1])
            xs = np.unique(auto_ref[:, 2])
            grid_shape = (len(zs), len(ys), len(xs))

            # Predict where manual ref points land using the dense grid (no RBF needed)
            corrected_mov = auto_mov.copy()
            for dim in range(3):
                interp = RegularGridInterpolator(
                    (zs, ys, xs), auto_mov[:, dim].reshape(grid_shape),
                    bounds_error=False, fill_value=None)
                manual_mov_predicted = interp(manual_ref)
                corrections = manual_mov[:, dim] - manual_mov_predicted
                # Sparse RBF only on the few manual points
                correction_warper = SparseWarper(manual_ref, corrections[:, np.newaxis], smooth=2)
                corrected_mov[:, dim] += correction_warper.transform(auto_ref)[:, 0]

            self.dense_grid_mov_pts = corrected_mov.tolist()
            with open(self.dense_grid_points_file_corrected, "w") as fd:
                json.dump({
                    self.REFERENCE: self.dense_grid_ref_pts,
                    self.MOVING: self.dense_grid_mov_pts
                }, fd, indent=2)


    def init_state(self):
        #breakpoint()
        """Initialize each of the viewers' states"""
        self.init_actions(self.reference_viewer,
                          self.on_reference_annotate)
        self.init_actions(self.moving_viewer, self.on_moving_annotate)
        self.undo_stack = []
        self.redo_stack = []
        self.selection_index = len(self.reference_pts)
        self.refresh()
        self.update_after_edit()
    
    def init_warper(self):
        """Initialize the warper"""
        self.warper = Warper(self.reference_pts, self.moving_pts)
        inputs = [
            np.arange(0,
                      self.reference_shape[_]+ self.decimation - 1,
                      self.decimation)
            for _ in range(3)]
        self.warper = self.warper.approximate(*inputs)

    @property
    def annotation_reference_pts(self):
        """Reference points in z, y, x order"""
        return self.reference_pts

    @property
    def annotation_moving_pts(self):
        """Moving points in z, y, x order"""
        return self.moving_pts
    

    def init_actions(self, viewer, on_annotate):
        """Initialize the actions for a viewer

        :param viewer: the reference or moving viewer
        :param on_edit: the function to execute when the user wants to edit
        """
      
        viewer.actions.add(self.ANNOTATE_ACTION, on_annotate)
            
        # viewer.actions.add(self.BRIGHTER_ACTION, self.on_brighter)
    
       
        viewer.actions.add(self.CLEAR_ACTION, self.on_clear)
        # viewer.actions.add(self.DIMMER_ACTION, self.on_dimmer)
        viewer.actions.add(self.DONE_ACTION, self.on_done)
        viewer.actions.add(self.EDIT_ACTION, self.on_edit)
        viewer.actions.add(self.NEXT_ACTION, self.on_next)
        viewer.actions.add(self.PREVIOUS_ACTION, self.on_previous)
        
      
        # viewer.actions.add(self.REFERENCE_BRIGHTER_ACTION, self.on_reference_brighter)
        # viewer.actions.add(self.REFERENCE_DIMMER_ACTION,self.on_reference_dimmer) 
     
        
       
        viewer.actions.add(self.REFRESH_ACTION, self.on_refresh)
        viewer.actions.add(self.SAVE_ACTION, self.on_save)
        viewer.actions.add(self.REDO_ACTION, self.on_redo)
        viewer.actions.add(self.UNDO_ACTION, self.on_undo)
        viewer.actions.add(self.WARP_ACTION, self.on_warp)
        
        if viewer == self.reference_viewer:
            viewer.actions.add(self.TRANSLATE_ACTION, self.on_translate)
        with viewer.config_state.txn() as s:
            bindings_viewer = s.input_event_bindings.viewer
            # bindings_viewer[self.BRIGHTER_KEY] = self.BRIGHTER_ACTION
            bindings_viewer[self.CLEAR_KEY] = self.CLEAR_ACTION
            # bindings_viewer[self.DIMMER_KEY] = self.DIMMER_ACTION
            bindings_viewer[self.DONE_KEY] = self.DONE_ACTION
            bindings_viewer[self.EDIT_KEY] = self.EDIT_ACTION
            bindings_viewer[self.NEXT_KEY] = self.NEXT_ACTION
            bindings_viewer[self.PREVIOUS_KEY] = self.PREVIOUS_ACTION
            
            # bindings_viewer[self.REFERENCE_BRIGHTER_KEY] = \
            #     self.REFERENCE_BRIGHTER_ACTION
            
            # bindings_viewer[self.REFERENCE_DIMMER_KEY] = \
            #     self.REFERENCE_DIMMER_ACTION
            
            
            bindings_viewer[self.REFRESH_KEY] = self.REFRESH_ACTION
            bindings_viewer[self.SAVE_KEY] = self.SAVE_ACTION
            bindings_viewer[self.REDO_KEY] = self.REDO_ACTION
            bindings_viewer[self.UNDO_KEY] = self.UNDO_ACTION
            bindings_viewer[self.WARP_KEY] = self.WARP_ACTION
            if viewer == self.reference_viewer:
                bindings_viewer[self.TRANSLATE_KEY] = self.TRANSLATE_ACTION

    def on_reference_annotate(self, s):
        """Handle an edit in the reference viewer

        :param s: the current state
        """
        point = s.mouse_voxel_coordinates
        reference_points = np.array(self.reference_pts)
        if len(reference_points) > 0:
            distances = np.sqrt(np.sum(np.square(
                reference_points - point[np.newaxis, ::-1]), 1))
            if np.min(distances) < self.min_distance:
                self.post_message(
                    self.reference_viewer, self.EDIT,
                    "Point at %d %d %d is too close to some other point" %
                    tuple(point.tolist()))
                return
        msg = "Edit point: %d %d %d" %  tuple(point.tolist())
        self.post_message(self.reference_viewer, self.EDIT, msg)

        with self.reference_viewer.txn() as txn:
            layer = neuroglancer.PointAnnotationLayer(
                points=[point.tolist()],
                annotation_color=self.EDIT_ANNOTATION_COLOR)
            txn.layers[self.EDIT] = layer

    def post_message(self, viewer, kind, msg):
        """Post a message to a viewer

        :param viewer: the reference or moving viewer
        :param kind: the kind of message - a name slot for the message
        :param msg: the message to post
        """
        if viewer is None:
            self.post_message(self.reference_viewer, kind, msg)
            self.post_message(self.moving_viewer, kind, msg)
        else:
            with viewer.config_state.txn() as cs:
                cs.status_messages[kind] = msg
        
    def on_moving_annotate(self, s):
        """Handle an edit in the moving viewer

        :param s: the current state
        """
        point = s.mouse_voxel_coordinates
        moving_points = np.array(self.moving_pts)
        if len(moving_points) > 0:
            distances = np.sqrt(np.sum(np.square(
                moving_points - point[np.newaxis, ::-1]), 1))
            if np.min(distances) < self.min_distance:
                self.post_message(
                    self.moving_viewer, self.EDIT,
                    "Point at %d %d %d is too close to some other point" %
                    tuple(point.tolist()))
                return
        msg = "Edit point: %d %d %d" %  tuple(point.tolist())
        self.post_message(self.moving_viewer, self.EDIT, msg)

        with self.moving_viewer.txn() as txn:
            layer = neuroglancer.PointAnnotationLayer(
                points=[point.tolist()],
                annotation_color=self.EDIT_ANNOTATION_COLOR)
            txn.layers[self.EDIT] = layer

    def on_clear(self, s):
        """Clear the current edit annotation"""
        self.clear()

    def clear(self):
        """Clear the edit annotation from the UI"""
        with self.reference_viewer.txn() as txn:
            txn.layers[self.EDIT] = neuroglancer.PointAnnotationLayer(
                annotation_color=self.EDIT_ANNOTATION_COLOR)
        with self.moving_viewer.txn() as txn:
            txn.layers[self.EDIT] = neuroglancer.PointAnnotationLayer(
                annotation_color=self.EDIT_ANNOTATION_COLOR)

    def on_edit(self, s):
        """Transfer the currently selected point to the edit annotation"""
        layer = s.viewerState.layers[self.CORRESPONDENCE_POINTS].layer
        d = layer.to_json()
        if "selectedAnnotation" in d:
            idx = int(d["selectedAnnotation"])
            self.selection_index = idx
            self.edit_point(self.selection_index)
            self.redo_stack.clear()

    def edit_point(self, idx, add_to_undo=True):
        reference, moving= self.remove_point(idx, add_to_undo)
            
        with self.reference_viewer.txn() as txn:
            pointlayer(
                txn, self.EDIT,
                [reference[0]], [reference[1]], [reference[2]],
                color=self.EDIT_ANNOTATION_COLOR,
                voxel_size=self.reference_voxel_size)
            txn.position = reference[::-1]
        with self.moving_viewer.txn() as txn:
            pointlayer(
                txn, self.EDIT,
                [moving[0]], [moving[1]], [moving[2]],
                color=self.EDIT_ANNOTATION_COLOR,
                voxel_size=self.moving_voxel_size)
            txn.position = moving[::-1]
            


    def on_next(self, s):
        """Save the current edit and move to the next point"""
        self.on_done(s)
        if len(self.reference_pts) > 0:
            self.selection_index = \
                (self.selection_index + 1) % len(self.reference_pts)
            self.edit_point(self.selection_index)
            self.redo_stack.clear()

    def on_previous(self, s):
        """Save the current edit and move to the previous point"""
        self.on_done(s)
        if len(self.reference_pts) > 0:
            self.selection_index = \
                len(self.reference_pts) - 1 if self.selection_index == 0 \
                else self.selection_index - 1
            self.edit_point(self.selection_index)
            self.redo_stack.clear()

    def on_done(self, s):
        """Handle editing done"""
        self.rp = self.get_reference_edit_point()
        self.mp = self.get_moving_edit_point()
            
        if self.mp and self.rp: 
            self.add_point(self.selection_index, self.rp, self.mp)
            self.clear()
            self.redo_stack.clear()
            

    def get_moving_edit_point(self):
        try:
            ma = self.moving_viewer.state.layers[self.EDIT].annotations
        except AttributeError:
            ma = self.moving_viewer.state.layers[self.EDIT].points
        if len(ma) == 0:
            return None
        if isinstance(ma[0], np.ndarray):
            points = ma[0].tolist()[::-1] 
        else: 
            points= ma[0].point.tolist()[::-1]
        self.post_message(self.moving_viewer, self.EDIT,
                "Get the moving edit point at %d, %d, %d" %
                tuple(points)[::-1])
        return points

    def get_reference_edit_point(self):
        """Get the current edit point in the reference frame"""
        try:
            ra = self.reference_viewer.state.layers[self.EDIT].annotations
        except AttributeError:
            ra = self.reference_viewer.state.layers[self.EDIT].points
        if len(ra) == 0:
            return None
        if isinstance(ra[0], np.ndarray):
            points = ra[0].tolist()[::-1] 
        else: 
            points= ra[0].point.tolist()[::-1]
        self.post_message(self.reference_viewer, self.EDIT,
                "Get the reference edit point at %d, %d, %d" %
                tuple(points)[::-1])
        return points


    def add_point(self, idx, reference_point, moving_point, add_to_undo=True):
        """Add a point to the reference and moving points list

        :param idx: where to add the point
        :param reference_point: the point to add in the reference space,
               in Z, Y, X order
        :param moving_point: the point to add in the moving space,
               in Z, Y, X order
        :param add_to_undo: True to add a "delete" operation to the undo stack,
        False to add it to the redo stack.
        """
        self.reference_pts.insert(idx, reference_point)
        self.post_message(self.reference_viewer, self.EDIT,
                          "Added point at %d, %d, %d" %
                          tuple(reference_point[::-1]))
        self.moving_pts.insert(idx, moving_point)
        self.post_message(self.moving_viewer, self.EDIT,
                          "Added point at %d, %d, %d" %
                          tuple(moving_point[::-1]))
        entry = (self.edit_point, (idx, not add_to_undo))
        if add_to_undo:
            self.undo_stack.append(entry)
        else:
            self.redo_stack.append(entry)
        self.update_after_edit()


    def remove_point(self, idx, add_to_undo=True):
        """Remove the point at the given index

        :param idx: the index into the reference_pts and moving_pts array
        :param add_to_undo: True to add the undo operation to the undo stack,
        false to add it to the redo stack.
        :returns: the reference and moving points removed.
        """
        reference_point = self.reference_pts.pop(idx)
        self.post_message(self.reference_viewer, self.EDIT,
                          "removed point %d at %d, %d, %d" %
                          tuple([idx] + list(reference_point[::-1])))
        moving_point = self.moving_pts.pop(idx)
        self.post_message(self.moving_viewer, self.EDIT,
                          "removed point %d at %d, %d, %d" %
                          tuple([idx] + list(moving_point[::-1])))
        entry = (self.add_point, (idx, reference_point, moving_point,
                                  not add_to_undo)) 
        
        if add_to_undo:
            self.undo_stack.append(entry)
        else:
            self.redo_stack.append(entry)
        self.update_after_edit()
        return reference_point, moving_point

    def update_after_edit(self):
        #
        viewer_points_voxelsize=((self.reference_viewer,
                                            self.annotation_reference_pts,
                                            self.reference_voxel_size),
                                           (self.moving_viewer,
                                            self.annotation_moving_pts,
                                            self.moving_voxel_size),
                                           )
        
        for viewer, points, voxel_size in viewer_points_voxelsize:
            points = np.array(points, dtype=np.float32)
            if len(points) == 0:
                points = points.reshape(0, 3)
            with viewer.txn() as txn:
                pointlayer(
                    txn, self.CORRESPONDENCE_POINTS,
                    points[:, 0], points[:, 1], points[:, 2],
                    voxel_size=voxel_size)
                pointlayer(
                    txn, self.EDIT, np.zeros(0), np.zeros(0), np.zeros(0),
                    color=self.EDIT_ANNOTATION_COLOR, voxel_size=voxel_size)

    def on_refresh(self, s):
        self.refresh()
        
    def refresh(self):
        """Refresh both views"""
        with self.moving_viewer.txn() as s:
            s.dimensions = CoordinateSpace(
                  names=["x", "y", "z"],
                  units=["µm"],
                  scales=self.moving_voxel_size)
            
            # Try to preserve current contrast limits
            try:
                layer_obj = self.moving_viewer.state.layers[self.IMAGE]
                old_limits = get_contrast_limits(layer_obj)
            except:
                old_limits = [0, self.moving_brightness]
            layer(s, self.IMAGE, self.moving_image, gray_shader,
                  voxel_size=self.moving_voxel_size,
                  contrast_limits = old_limits)
        with self.reference_viewer.txn() as s:
            s.dimensions = CoordinateSpace(
                  names=["x", "y", "z"],
                  units=["µm"],
                  scales=self.reference_voxel_size)
            if self.reference_images is not None:
                for i, reference_image in enumerate(self.reference_images):
                    key = f"{self.REFERENCE}_{i}"
                    try:
                        layer_obj = self.reference_viewer.state.layers[key]
                        old_limits = get_contrast_limits(layer_obj)
                    except:
                        old_limits = [0, soft_max_brightness(reference_image, percentile=99)]
                    layer(s, f"{self.REFERENCE}_{i}", reference_image, green_shader,
                          voxel_size=self.reference_voxel_size,
                          contrast_limits = old_limits)
                    
            if self.edge_image is not None:
                # try:
                #     layer_obj = self.reference_viewer.state.layers[self.EDGE]
                #     old_limits = get_contrast_limits(layer_obj)
                # except:
                #     old_limits = [0, self.edge_brightness]
                layer(s, self.EDGE, self.edge_image, green_shader,
                    voxel_size=self.reference_voxel_size,
                    contrast_limits = [0, self.edge_brightness])
            
            # Preserve old limits
            try:
                layer_obj = self.reference_viewer.state.layers[self.ALIGNMENT]
                old_align_limits = get_contrast_limits(layer_obj)
            except Exception as e:
                print(e)
                old_align_limits = [0, self.moving_brightness]
            layer(s, self.ALIGNMENT, self.alignment_image, gray_shader,
                  voxel_size=self.reference_voxel_size,
                  contrast_limits = old_align_limits)
            if self.segmentation is not None:
                seglayer(s, self.SEGMENTATION, self.segmentation)

    def on_undo(self, s):
        """Undo the last operation"""
        if len(self.undo_stack) > 0:
            undo = self.undo_stack.pop(-1)
            undo[0](*undo[1])
        else:
            self.post_message(None, self.EDIT, "Nothing to undo")

    def on_redo(self, s):
        """Redo the last undo operation"""
        if len(self.redo_stack) > 0:
            redo = self.redo_stack.pop(-1)
            redo[0](*redo[1])
        else:
            self.post_message(None, self.EDIT, "Nothing to redo")


    def on_translate(self, s):
        """Translate the editing coordinate in the reference frame to moving"""
        rp = self.get_reference_edit_point()
        if self.warper is not None and rp:
            self.mp = self.warper(np.atleast_2d(rp))[0]
            with self.moving_viewer.txn() as txn:
                txn.layers[self.EDIT] = neuroglancer.PointAnnotationLayer(
                    points=[self.mp[::-1]],
                    annotation_color=self.EDIT_ANNOTATION_COLOR)
                txn.position = self.mp[::-1]


    def on_save(self, s):
        """Handle a point-save action

        :param s: the current state of whatever viewer
        """
        self.save_points()
        viewers= (self.reference_viewer, self.moving_viewer)
        for viewer in viewers:
            self.post_message(viewer, self.EDIT, "Saved point state")

    def on_warp(self, s):
        cs, generation = \
            self.reference_viewer.config_state.state_and_generation
        cs = deepcopy(cs)

        cs.status_messages[self.WARP_ACTION] = \
            "Warping alignment image to reference... (patience please)"
        self.reference_viewer.config_state.set_state(
             cs, existing_generation=generation)
        try:
            self.post_message(self.reference_viewer, self.WARP_ACTION, "Warping alignment image to reference... (patience please)")
            self.align_image()
            # Preserve old limits
            try:
                layer_obj = self.reference_viewer.state.layers[self.ALIGNMENT]
                old_align_limits = get_contrast_limits(layer_obj)
            except Exception as e:
                print(e)
                old_align_limits = [0, self.moving_brightness]
            with self.reference_viewer.txn() as txn:
                layer(txn, self.ALIGNMENT, self.alignment_image,
                      gray_shader, voxel_size=self.reference_voxel_size, contrast_limits=old_align_limits),
            # self.refresh_brightness()
            self.post_message(self.reference_viewer, self.WARP_ACTION,
                    "Warping complete, thank you for your patience.")
        except:
            self.post_message(self.reference_viewer, self.WARP_ACTION,
                    "Oh my, something went wrong. See console log for details.")
            raise

    def align_image(self):
        """Warp the moving image into the reference image's space"""
        self.save_points() # Make sure we save the points (and update the dense grid points) before warping
        self.init_warper() # reinitialize based on the points
        warp_path = Path(self.points_file).parent / "registered_manual.zarr"
        if len(self.dense_grid_ref_pts) > 0:
            # Reconstruct grid_values directly from the dense points — no RBF fitting needed
            auto_ref = np.array(self.dense_grid_ref_pts)
            auto_mov = np.array(self.dense_grid_mov_pts)
            zs = np.unique(auto_ref[:, 0])
            ys = np.unique(auto_ref[:, 1])
            xs = np.unique(auto_ref[:, 2])
            grid_shape = (len(zs), len(ys), len(xs))
            grid_values = np.expand_dims(np.expand_dims(
                np.array([auto_mov[:, dim].reshape(grid_shape) for dim in range(3)]),
                1), 1)  # shape: (3, 1, 1, Z, Y, X)
            gpu_warp_image(
                moving_zarr=self.moving_image_zarr,
                warped_zarr_path=str(warp_path),
                fixed_pts=self.reference_pts,
                moving_pts=self.moving_pts,
                fixed_img_size=self.reference_shape,
                moving_voxel_size=(1,1,1),
                fixed_voxel_size=(1,1,1),
                grid_spacing=(32,32,32),
                num_workers=None,
                grid_values_path=grid_values
            )
        else:
            gpu_warp_image(
                moving_zarr=self.moving_image_zarr,
                warped_zarr_path=str(warp_path),
                fixed_pts=self.reference_pts,
                moving_pts=self.moving_pts,
                fixed_img_size=self.reference_shape,
                moving_voxel_size=(1,1,1),
                fixed_voxel_size=(1,1,1),
                grid_spacing=(32,32,32),
                num_workers=None
            )

        self.alignment_image = zarr.open(str(warp_path))[:]

    def print_viewers(self):
        """Print the URLs of the viewers to the console"""
        print("Reference viewer: %s" % repr(self.reference_viewer))
        print("Moving viewer: %s" % repr(self.moving_viewer))


    def launch_viewers(self):
        """Launch webpages for each of the viewers"""
        webbrowser.open_new(self.reference_viewer.get_viewer_url())
        webbrowser.open_new(self.moving_viewer.get_viewer_url())



def main(raw_args=None):
    logging.basicConfig(level=logging.INFO)
    args = parse_args(raw_args)
    if args.ip_address is not None and args.port is not None:
        neuroglancer.set_server_bind_address(
            bind_address=args.ip_address,
            bind_port=args.port)
    elif args.ip_address is not None:
        neuroglancer.set_server_bind_address(
            bind_address=args.ip_address)
    elif args.port is not None:
        neuroglancer.set_server_bind_address(bind_port=args.port)
        
    reference_voxel_size = \
        [float(_)*1 for _ in args.reference_voxel_size.split(",")]
    moving_voxel_size = \
        [float(_)*1 for _ in args.moving_voxel_size.split(",")]
    
    if args.reference_images is not None:
        logging.info("Reading reference images")
        reference_images = []
        for ref_img_path in args.reference_images:
            if ref_img_path.endswith(".zarr") or os.path.isdir(ref_img_path):
                reference_image = zarr.open(ref_img_path, mode = "r")
            else:
                reference_image = tifffile.imread(ref_img_path)
            reference_images.append(reference_image)
    else:
        reference_images = None
    if args.edge_image is not None:
        logging.info("Reading edge image")
        edge_image = tifffile.imread(args.edge_image)
    else:
        edge_image = None
    logging.info("Reading moving image")
    if args.moving_image.endswith(".zarr") or os.path.isdir(args.moving_image):
        moving_image = zarr.open(args.moving_image, mode = "r")
    else:
        moving_image = tifffile.imread(args.moving_image)
    logging.info("Reading in warped image")
    logging.info(args.warped_zarr)
    if args.warped_zarr is not None:
        if args.warped_zarr.endswith(".zarr") or os.path.isdir(args.warped_zarr):
            warped_zarr = zarr.open(args.warped_zarr, mode = "r")[:]
        else:
            warped_zarr = tifffile.imread(args.warped_zarr)
    else:
        warped_zarr = None 

    if args.segmentation is not None:
        logging.info("Reading segmentation")
        segmentation = tifffile.imread(args.segmentation).astype(np.uint32)
    else:
        segmentation = None

    vp = ViewerPair(reference_images, edge_image, moving_image, warped_zarr, segmentation, args.points,
                    reference_voxel_size, moving_voxel_size, n_workers=args.n_workers,
                    x_index=args.x_index, y_index=args.y_index, z_index=args.z_index,
                    dense_grid_points_file=args.points_full)
    
    if not args.no_launch:
        vp.launch_viewers()
    vp.print_viewers()
    print("Hit ctrl+C to exit")
    while True:
        time.sleep(.1)

if __name__ == "__main__":
    main()
    
