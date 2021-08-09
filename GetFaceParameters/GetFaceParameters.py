# Remember to open the `TEXT COMMANDS' window under the `View' menu.
# Information will be logged there.
import adsk.core, adsk.fusion, adsk.cam, traceback

_ui = None
_app = None

class UiLogger:
    def __init__(self, forceUpdate):
        # _app = adsk.core.Application.get()
        # _ui  = _app.userInterface
        global _ui
        palettes = _ui.palettes
        self.textPalette = palettes.itemById("TextCommands")
        self.forceUpdate = forceUpdate
        self.textPalette.isVisible = True

    def print(self, text):
        self.textPalette.writeText(str(text))
        if (self.forceUpdate):
            adsk.doEvents()

def run(context):
    global _ui, _app
    try:
        _app = adsk.core.Application.get()
        _ui  = _app.userInterface
        logger = UiLogger(True)

        # Have a face selected.
        faceSel = _ui.selectEntity('Select a face', 'Faces')
        if faceSel:
            face = adsk.fusion.BRepFace.cast(faceSel.entity)
        if not face:
            _ui.messageBox('A face must be selected.')

        # Get the face geometry.
        geom = face.geometry
        if geom.classType() == adsk.core.Cylinder.classType():
            logger.print("Cylinder selected.  R = {} [mm]".format(geom.radius * 10))
            norm = [geom.axis.x, geom.axis.y, geom.axis.z]
            r = geom.radius * 10
        elif geom.classType() == adsk.core.Sphere.classType():
            logger.print("Sphere selected.  R = {} [mm]".format(geom.radius * 10))
            norm = [0.0, 0.0, 1.0]
            r = geom.radius * 10
        else:
            norm = [geom.normal.x, geom.normal.y, geom.normal.z]
            r = 0.0
        origin = geom.origin

        logger.print("Origin [mm]: ({}, {}, {}), norm/axis: ({}, {}, {}), r [mm]: {}".format(
            origin.x*10, origin.y*10, origin.z*10,
            norm[0], norm[1], norm[2], r))
    except:
        if _ui:
            _ui.messageBox('Failed:\n{}'.format(traceback.format_exc()))
