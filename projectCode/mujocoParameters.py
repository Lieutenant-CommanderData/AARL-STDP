from typing import Optional
import mediapy as media
import mujoco as mj
import mujoco.viewer as mjv

# Camera options for the fun of it
mj_camera = ['fixed', 'Angle2', 'Angle3', 'Angle4', 'Angle5', 'Angle6']

class MuJoCoExtras():
    '''Some extra functions for using MuJoCo!
    '''
    def __init__(self):
        pass
    # Show image of XML file for reference
    def showModel(self, mjmodel, mjdata, camera: Optional[int]=0):
        '''Renders the passed MuJoCo model using the specified camera to create a .png of the model.
        
        Args:
            mjmodel >>> Which MuJoCo model you want rendered
            mjdata >>> The data that goes along with the model
            camera >>> Selects which MuJoCo camera to use for the render. Defaults to camera 0
        '''
        if camera > 5: camera = 0
        renderer = mj.Renderer(mjmodel) # Create MuJoCo renderer
        mj.mj_forward(mjmodel, mjdata) # Activate the model into its first timestep
        renderer.update_scene(mjdata, camera=mj_camera[camera]) # Render the scene
        media.write_image('./results/mjModel.png', renderer.render()) # Save to image
        renderer.close() # MUST CLOSE TO PREVENT CRASH

    # Pop the model out into the interactive MuJoCo window.
    def popOutModel(self, path):
        '''Pops a MuJoCo model out into the interactive MuJoCo viewer.
        
        Args:
            path >>> The path to the desired XML file you want loaded into the viewer
            '''
        m = mj.MjModel.from_xml_path(path)
        d = mj.MjData(m)
        v = mjv.launch(m, d)
