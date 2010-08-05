import numpy as np
import opencv.highgui  as hg
import opencv.cv       as cv

FORMAT_BGR  = 0
FORMAT_GRAY = 1
FORMAT_RGB  = 2
FORMAT_HSV  = 3
NFORMATS    = 4

class video_base (object):
    def __init__(self, format, *args):
        self._init_cap(args)
        self.set_format(format)

        self.own_data = True # set to false if you want to do less copying
        
        self.width   = int(hg.cvGetCaptureProperty(self.cap,
                                                   hg.CV_CAP_PROP_FRAME_WIDTH))
        self.height  = int(hg.cvGetCaptureProperty(self.cap,
                                                   hg.CV_CAP_PROP_FRAME_HEIGHT))
        self.fps     = -1.0
        self.nframes = -1

    def _init_cap(self, args = None):
        self.framepos = -1
        if args is not None:
            self._cap_args = args
        self.cap = self.create_capture( *self._cap_args )
        if self.cap is None:
            raise IOError("couldn't initialize capture")
    
    def _destr_cap(self):
        hg.cvReleaseCapture(self.cap)
        
    def next(self):
        ret = hg.cvGrabFrame(self.cap)
        if ret: self.framepos += 1
        return ret

    def seek(self, pos):
        if pos < self.framepos:
            raise Exception('unable to seek backward')
        ret = 1
        while (ret and self.framepos < pos):
            ret = self.next()
        
        return ret

    def set_format(self, format):
        if 0 <= format < NFORMATS:
            self.format = format
        else:
            raise Exception('unknown format')


    def frame(self):
        if self.framepos == -1:
            raise Exception('call next before the first frame!')
        
        format = self.format
        img = hg.cvRetrieveFrame(self.cap)
        nchannels = 1 if format == FORMAT_GRAY else 3
        shape = \
            (img.height, img.width) if nchannels == 1 else \
            (img.height, img.width, nchannels)
        
        if format == FORMAT_BGR: # default format
            frame = np.ndarray(shape = shape, dtype = np.uint8, 
                               buffer = img.imageData)
            if self.own_data: frame = frame.copy()
            return frame
        
        size = cv.cvSize(img.width, img.height)
        img2 = cv.cvCreateImage(size, 8, nchannels)
        cvt_type = -1
        if format == FORMAT_GRAY:
            cvt_type = cv.CV_BGR2GRAY
        elif format == FORMAT_RGB:
            cvt_type = cv.CV_BGR2RGB
        elif format == FORMAT_HSV:
            cvt_type = cv.CV_BGR2HSV
        else: assert(0)
        
        cv.cvCvtColor(img, img2, cvt_type)
        
        frame = np.ndarray(shape = shape, dtype = np.uint8,
                           buffer = img2.imageData)
        if self.own_data: frame = frame.copy()
        return frame

    def __del__(self):
        self._destr_cap()


class video_file (video_base):
    def __init__(self, fname, format=FORMAT_RGB):
        self.create_capture = hg.cvCreateFileCapture
        super(video_file, self).__init__(format, fname)
        
        self.fps    =     hg.cvGetCaptureProperty(self.cap,
                                                  hg.CV_CAP_PROP_FPS)
        self.nframes= int(hg.cvGetCaptureProperty(self.cap,
                                                  hg.CV_CAP_PROP_FRAME_COUNT))
        
    def seek(self, pos):
        #if pos < self.framepos:
        #    self._destr_cap()
        #    self._init_cap()
        if pos < self.framepos:
            hg.cvSetCaptureProperty(self.cap, hg.CV_CAP_PROP_POS_FRAMES, 0.0)
            self.framepos = -1
        return super(video_file, self).seek(pos)


class video_cam (video_base):
    def __init__(self, camno = -1, format=FORMAT_RGB):
        self.create_capture = hg.cvCreateCameraCapture
        super(video_cam, self).__init__(format, camno)
        
