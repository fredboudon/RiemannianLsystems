from pgljupyter import *
from openalea.lpy import *
from openalea.lpy.lsysparameters import *
from IPython.display import Code, Markdown, display
from ipywidgets import HBox, VBox, Layout

import warnings
warnings.filterwarnings("ignore")

def get_lsystem_parameters(self):
    return self._LsystemWidget__lp

LsystemWidget.get_lsystem_parameters = get_lsystem_parameters
del get_lsystem_parameters

def LsystemEditor(fname, *args, **kwds):
        from pgljupyter.editors import make_color_editor, make_scalar_editor, make_curve_editor
        plane = kwds.get('plane',False)
        if plane in kwds : del kwds['plane']
            
        lsw = LsystemWidget(fname, *args, **kwds)
        lp = lsw.get_lsystem_parameters()
        lsw.plane = plane
        editors = []

        def on_value_changed(param):
            print('on_value_changed', param)
            def fn(change):
                print('change', change)
                if 'new' in change:
                    value = change['new']
                    name = change['name']
                    if isinstance(param, BaseScalar) and hasattr(param, name):
                        setattr(param, name, value)
                        if name == 'value':
                            lsw.set_parameters(lp.dumps())
                    elif isinstance(param, tuple) and isinstance(param[1], (pgl.NurbsCurve2D, pgl.BezierCurve2D, pgl.Polyline2D)):
                        new_len = len(value)

                        if isinstance(param[1], (pgl.NurbsCurve2D, pgl.BezierCurve2D)):
                            prev_len = len(param[1].ctrlPointList)
                            param[1].ctrlPointList = pgl.Point3Array([pgl.Vector3(p[0], p[1], 1) for p in value])
                        elif isinstance(param[1], pgl.Polyline2D):
                            prev_len = len(param[1].pointList)
                            param[1].pointList = pgl.Point2Array([pgl.Vector2(p[0], p[1]) for p in value])

                        if prev_len != new_len:
                            param[1].setKnotListToDefault()
                        print('set_parameters')
                        lsw.set_parameters(lp.dumps())

            return fn


        if lp:
            for scalar in lp.get_category_scalars():
                editor = make_scalar_editor(scalar, no_name=True, extended_editor=False)
                if editor:
                    editor.observe(on_value_changed(scalar))
                    editors.append(editor)

            for param in lp.get_category_graphicalparameters():
                editor = make_curve_editor(param, no_name=True)
                if editor:
                    editor.observe(on_value_changed(param), 'value')
                    editors.append(editor)

        w, h = lsw.size_display
        if len(editors):
            return HBox((
                HBox([lsw], layout=Layout(margin='10px', min_width=str(w) + 'px', min_height=str(h) + 'px')),
                HBox(editors, layout=Layout(margin='0', flex_flow='row wrap'))
            ))
        else:
            return lsw
        
        

def display_example(filename, caption = None, subcaption = None, size_world=50, animate=True, plane = False, codedisplay = True):
      print()
      filename =  '../Lsystems/'+filename
      if not filename.endswith('.lpy'):
            filename += '.lpy'
      code = open(filename,'r').read()
      code = code.split('###### INITIALISATION ######')[0]
      lcaption = None
      lsubcaptions = []
      for i, codeline in enumerate(code.splitlines()):
          if codeline.startswith('#'):
              if i == 0:
                  lcaption = codeline[1:]
              else:
                 lsubcaptions.append(codeline[1:])
          elif len(codeline) > 0:
              if i > 0:
                code = '\n'.join(code.splitlines()[i:])
              break
      lsubcaption = '\n'.join(lsubcaptions)

      widgets = []
      if not caption is None:
        widgets += [Markdown('\n### '+caption)] 
        if subcaption:
            widgets += [Markdown(subcaption)]
      elif not lcaption is None:
        widgets += [Markdown('\n### '+lcaption)] 
        if lsubcaption:
            widgets += [Markdown(lsubcaption)]

      lw = LsystemEditor(filename, size_world=size_world, animate=animate, plane=plane)
      if codedisplay:
        widgets += [Markdown('#### Lsystem:'),Code(data=code, language='python')]
      widgets += [Markdown('#### Output:'),lw]
      for widget in widgets:
        display(widget)

def display_examples(examples, size_world=50, animate=True, codedisplay = True):
    for example in examples:
        if type(example) == str:
           filename = example
           caption, subcaption = None, None
        elif len(example) == 1:
           filename = example[0]
           caption, subcaption = None, None
        elif len(example) == 2:
           filename, caption = example
           subcaption = None
        elif len(example) == 3:
           filename, caption, subcaption = example
        else:
            raise ValueError(example)
        display_example(filename, caption, subcaption, size_world=size_world, animate=animate, codedisplay = codedisplay)
