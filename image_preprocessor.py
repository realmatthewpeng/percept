import pulse2percept as p2p

class ImagePreprocessor:
    doInvert = False
    doRGBtoGray = False
    doResize = {}
    doScale = 0.0
    doShift = {}
    doRotate = 0.0
    doTrim = {}
    doThreshold = {}
    doFilter = {}

    def __init__(self, doInvert=False, doRGBtoGray=False, doResize={}, doScale=0.0, doShift={}, doRotate=0.0, doTrim={}, doThreshold={}, doFilter={}):
        self.doInvert = doInvert
        self.doRGBtoGray = doRGBtoGray
        self.doResize = doResize
        self.doScale = doScale
        self.doShift = doShift
        self.doRotate = doRotate
        self.doTrim = doTrim
        self.doThreshold = doThreshold
        self.doFilter = doFilter
    
    def process_image(self, image: p2p.stimuli.ImageStimulus) -> p2p.stimuli.ImageStimulus:
        if self.doInvert:
            image = image.invert()
        if self.doRGBtoGray:
            image = image.rgb2gray()
        if self.doResize:
            image = image.resize(**self.doResize)
        if self.doScale:
            image = image.scale(self.doScale)
        if self.doShift:
            image = image.shift(**self.doShift)
        if self.doRotate:
            image = image.rotate(self.doRotate)
        if self.doTrim:
            image = image.trim(**self.doTrim)
        if self.doThreshold:
            image = image.threshold(**self.doThreshold)
        if self.doFilter:
            image = image.filter(**self.doFilter)
        return image
