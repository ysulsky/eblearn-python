from eblearn.util import *
from eblearn import state

def plot_filters(m, shape_in,
                 transpose=False, orig_x=5, orig_y=5, max_x=390, scale = 1.0):
    assert (len(shape_in) == 2)
    ensure_window(title = 'PSD Filters')
    h, w = shape_in; h *= scale; w *= scale
    padding, pos_x, pos_y = 5, 0, 0
    filters =m.parameter.states[0].x
    if transpose: filters = filters.transpose(0,*(range(filters.ndim)[-1:0:-1]))
    minv = filters.min()
    maxv = filters.max()
    for k in filters:
        k = k.reshape(shape_in)
        if pos_x + w > max_x:
            pos_x  = 0; pos_y += h + padding
        draw_mat(k, orig_x + pos_x, orig_y + pos_y,
                 minv = minv, maxv = maxv, scale = scale)
        pos_x += w + padding
    pos_y += h + padding + 16
    draw_text(orig_x, orig_y + pos_y, 'Values range in %g .. %g' % (minv,maxv))

def plot_reconstructions(ds, machine, n = 1000,
                         orig_x=5, orig_y=5, max_x=795, scale = 1.0):
    shape_in, shape_out = ds.shape()
    if len(shape_in) == len(shape_out) == 3:
        assert (shape_in[0] == shape_out[0] == 1)
        shape_in=shape_in[1:]
        shape_out=shape_out[1:]
    assert (len(shape_in) == len(shape_out) == 2)
    ensure_window(title = 'PSD Reconstructions')
    spacing, padding = 2, 10
    pic = empty((max(shape_in[0], shape_out[0]),
                 shape_in[1] + spacing + shape_out[1]))
    h, w = pic.shape; h *= scale; w *= scale
    pos_x, pos_y = 0, 0
    ds.seek(0)
    inp, tgt = state(()), state(())
    for i in xrange(min(n, ds.size())):
        ds.fprop(inp, tgt)
        machine.encoder.fprop(inp, machine.encoder_out)
        machine.decoder.fprop(machine.encoder_out, machine.decoder_out)
        ds.next()
        inpx = inp.x
        recx = machine.decoder_out.x
        white = max( inpx.max(), recx.max() )
        pic.fill(white)
        pic[:shape_in[0],:shape_in[1]] = inpx
        pic[:shape_out[0],shape_in[1] + spacing:] = recx
        if pos_x + w > max_x:
            pos_x  = 0; pos_y += h + padding
        draw_mat(pic, orig_x + pos_x, orig_y + pos_y,
                 maxv = white, scale = scale)
        pos_x += w + padding
