import os, tempfile

import Tkinter as tk
import Image   as image
import numpy   as np


tk_roots    = {}
tk_def_root = None
tk_cur_root = None
tk_max      = 1
def set_window(id = None):
    global tk_roots, tk_def_root, tk_cur_root
    if id is None: id = tk_def_root
    if id is None: new_window()
    else:
        assert (id in tk_roots)
        tk_cur_root = id

def ensure_window(width=None, height=None, title=None):
    global tk_roots, tk_cur_root
    r = tk_cur_root
    if r is None:
        kwargs = {}
        if width:  kwargs['width']=width
        if height: kwargs['height']=height
        if title:  kwargs['title']=title
        new_window(**kwargs)
    else:
        r = tk_roots[r]
        if width or height:
            if not height: height = r.geometry().split('+')[0].split('x')[1]
            if not width:   width = r.geometry().split('+')[0].split('x')[0]
            r.geometry('%sx%s' % (width, height))
        if title:
            r.title(title)

def new_window(width=800, height=600, title = 'Python Window'):
    global tk_roots, tk_cur_root, tk_def_root, tk_max
    
    root = tk.Tk(); root.geometry('%dx%d'%(width, height)); root.title(title)

    id = tk_max; tk_max +=1
    root.id = id

    tk_roots[id] = root
    if tk_def_root is None: 
        tk_def_root = id
    tk_cur_root = id
    
    canvas = tk.Canvas(root, width=width, height=height, bg = 'white')
    canvas.pack(expand=tk.YES, fill=tk.BOTH)
    canvas.images = []
    
    root.canvas = canvas
    
    def root_destr():
        global tk_roots, tk_def_root, tk_cur_root
        id = root.id
        cls(id); canvas.destroy(); root.destroy(); del tk_roots[id]
        next_id = None if not tk_roots else tk_roots.keys()[0]
        if tk_def_root == id: tk_def_root = next_id
        if tk_cur_root == id: tk_cur_root = tk_def_root
    
    root.protocol("WM_DELETE_WINDOW", root_destr)
    return root.id

def cls(id = None):
    ''' clear window '''
    global tk_roots, tk_cur_root
    if id is None: id = tk_cur_root
    if id is None: return
    canvas = tk_roots[id].canvas
    for i in canvas.find_all(): canvas.delete(i)
    canvas.images = []

def draw_mat(mat, x=0, y=0, minv=None, maxv=None, scale=1.0):
    global tk_roots, tk_cur_root
    assert (len(mat.shape) == 2 or (len(mat.shape) == 3 and mat.shape[2] == 3))
    if tk_cur_root is None: new_window()

    if minv is None: minv = mat.min()
    if maxv is None: maxv = mat.max()

    if maxv - minv < 1e-9:
        print '+++ Warning (draw_mat): maxv = minv'
        mat = np.zeros(mat.shape, np.uint8)
    else:
        mat = (mat.clip(minv, maxv) - minv) * (255. / (maxv - minv))
        mat = mat.astype(np.uint8)
    
    img = image.fromarray(mat)
    if scale != 1.0:
        width, height = img.size
        img = img.resize((width * scale, height * scale))

    # Python mangles the image if we try to send it directly
    fname = tempfile.mktemp('.ppm', 'tkimg_')
    img.save(fname, 'PPM')
    
    root = tk_roots[tk_cur_root]
    tkimg = tk.PhotoImage(file=fname, format='PPM', master=root)
    os.unlink(fname)

    root.canvas.create_image(x, y, anchor=tk.NW, image=tkimg)
    root.canvas.images.append(tkimg)

def draw_lines(x1, y1, x2, y2, *rest):
    global tk_roots, tk_cur_root
    if tk_cur_root is None: new_window()

    root = tk_roots[tk_cur_root]
    root.canvas.create_line(x1, y1, x2, y2, *rest)

def draw_text(x, y, text):
    global tk_roots, tk_cur_root
    if tk_cur_root is None: new_window()

    root = tk_roots[tk_cur_root]
    root.canvas.create_text(x, y, text=text, anchor=tk.SW, font=('courier',8))
