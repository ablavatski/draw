#!/ysr/bin/env python 

from __future__ import division

import numpy as np

import theano
import theano.tensor as T

from theano import tensor


# -----------------------------------------------------------------------------

def my_batched_dot(A, B):
    """Batched version of dot-product.     
       
    For A[dim_1, dim_2, dim_3] and B[dim_1, dim_3, dim_4] this         
    is \approx equal to:       
               
    for i in range(dim_1):     
        C[i] = tensor.dot(A[i], B[i])
       
    Returns        
    -------        
        C : shape (dim_1 \times dim_2 \times dim_4)        
    """
    C = A.dimshuffle([0, 1, 2, 'x']) * B.dimshuffle([0, 'x', 1, 2])
    return C.sum(axis=-2)


# -----------------------------------------------------------------------------

class ZoomableAttentionWindow(object):
    def __init__(self, channels, img_height, img_width, N):
        # A zoomable attention window for images.

        self.channels = channels
        self.img_height = img_height
        self.img_width = img_width
        self.N = N

    def filterbank_matrices(self, center_y, center_x, deltaY, deltaX, sigmaY, sigmaX):
        tol = 1e-4
        N = self.N

        muX = center_x.dimshuffle([0, 'x']) + deltaX.dimshuffle([0, 'x']) * (T.arange(N, dtype=theano.config.floatX) - N / 2 - 0.5)
        muY = center_y.dimshuffle([0, 'x']) + deltaY.dimshuffle([0, 'x']) * (T.arange(N, dtype=theano.config.floatX) - N / 2 - 0.5)

        a = tensor.arange(self.img_width, dtype=theano.config.floatX)
        b = tensor.arange(self.img_height, dtype=theano.config.floatX)

        FX = tensor.exp(-(a - muX.dimshuffle([0, 1, 'x'])) ** 2 / 2. / sigmaX.dimshuffle([0, 'x', 'x']) ** 2)
        FY = tensor.exp(-(b - muY.dimshuffle([0, 1, 'x'])) ** 2 / 2. / sigmaY.dimshuffle([0, 'x', 'x']) ** 2)
        FX = FX / (FX.sum(axis=-1).dimshuffle(0, 1, 'x') + tol)
        FY = FY / (FY.sum(axis=-1).dimshuffle(0, 1, 'x') + tol)

        return FY, FX

    def read(self, images, center_y, center_x, deltaY, deltaX, sigmaY, sigmaX):
        channels = self.channels
        batch_size = images.shape[0]

        # Reshape input into proper 2d images
        I = images.reshape((batch_size * channels, self.img_height, self.img_width))

        # Get separable filterbank
        FY, FX = self.filterbank_matrices(center_y, center_x, deltaY, deltaX, sigmaY, sigmaX)

        FY = T.repeat(FY, channels, axis=0)
        FX = T.repeat(FX, channels, axis=0)

        # apply to the batch of images
        W = my_batched_dot(my_batched_dot(FY, I), FX.transpose([0, 2, 1]))

        return W.reshape((batch_size, channels * self.N * self.N))

    def write(self, windows, center_y, center_x, deltaY, deltaX, sigmaY, sigmaX):
        channels = self.channels
        batch_size = windows.shape[0]

        # Reshape input into proper 2d windows
        W = windows.reshape((batch_size * channels, self.N, self.N))

        # Get separable filterbank
        FY, FX = self.filterbank_matrices(center_y, center_x, deltaY, deltaX, sigmaY, sigmaX)

        FY = T.repeat(FY, channels, axis=0)
        FX = T.repeat(FX, channels, axis=0)

        I = my_batched_dot(my_batched_dot(FY.transpose([0, 2, 1]), W), FX)

        return I.reshape((batch_size, channels * self.img_height * self.img_width))

    def nn2att(self, l):
        """Convert neural-net outputs to attention parameters
    
        Parameters
        ----------
        layer : :class:`~tensor.TensorVariable`
            A batch of neural net outputs with shape (batch_size x 5)
    
        Returns
        -------
        center_y : :class:`~tensor.TensorVariable` 
        center_x : :class:`~tensor.TensorVariable` 
        delta : :class:`~tensor.TensorVariable` 
        sigma : :class:`~tensor.TensorVariable` 
        gamma : :class:`~tensor.TensorVariable` 
        """
        center_y = l[:, 0]
        center_x = l[:, 1]
        log_deltaY = l[:, 2]
        log_deltaX = l[:, 3]
        log_sigmaY = l[:, 4]
        log_sigmaX = l[:, 5]
        log_gamma = l[:, 6]

        deltaY = T.exp(log_deltaY)
        deltaX = T.exp(log_deltaX)
        sigmaY = T.exp(log_sigmaY / 2.)
        sigmaX = T.exp(log_sigmaX / 2.)
        gamma = T.exp(log_gamma).dimshuffle(0, 'x')

        # normalize coordinates
        center_x = (center_x + 1.) / 2. * self.img_width
        center_y = (center_y + 1.) / 2. * self.img_height
        deltaY = (self.img_height - 1) / (self.N - 1) * deltaY
        deltaX = (self.img_width - 1) / (self.N - 1) * deltaX

        return center_y, center_x, deltaY, deltaX, sigmaY, sigmaX, gamma

    def nn2att_wn(self, l):
        center_y = l[:, 0]
        center_x = l[:, 1]
        log_deltaY = l[:, 2]
        log_deltaX = l[:, 3]

        deltaY = T.exp(log_deltaY)
        deltaX = T.exp(log_deltaX)

        center_x = (center_x + 1.) / 2.
        center_y = (center_y + 1.) / 2.
        deltaY = deltaY / (self.N - 1)
        deltaX = deltaX / (self.N - 1)

        return center_y, center_x, deltaY, deltaX


# =============================================================================

if __name__ == "__main__":
    from PIL import Image

    N = 40
    channels = 3
    height = 480
    width = 640

    # ------------------------------------------------------------------------
    att = ZoomableAttentionWindow(channels, height, width, N)

    I_ = T.matrix()
    center_y_ = T.vector()
    center_x_ = T.vector()
    deltaY_ = T.vector()
    deltaX_ = T.vector()
    sigmaY_ = T.vector()
    sigmaX_ = T.vector()
    W_ = att.read(I_, center_y_, center_x_, deltaY_, deltaX_, sigmaY_, sigmaX_)

    do_read = theano.function(inputs=[I_, center_y_, center_x_, deltaY_, deltaX_, sigmaY_, sigmaX_],
                              outputs=W_, allow_input_downcast=True)

    W_ = T.matrix()
    center_y_ = T.vector()
    center_x_ = T.vector()
    deltaY_ = T.vector()
    deltaX_ = T.vector()
    sigmaY_ = T.vector()
    sigmaX_ = T.vector()
    I_ = att.write(W_, center_y_, center_x_, deltaY_, deltaX_, sigmaY_, sigmaX_)

    do_write = theano.function(inputs=[W_, center_y_, center_x_, deltaY_, deltaX_, sigmaY_, sigmaX_],
                               outputs=I_, allow_input_downcast=True)

    # ------------------------------------------------------------------------

    I = Image.open("cat.jpg")
    I = I.resize((640, 480))  # .convert('L')

    I = np.asarray(I).transpose([2, 0, 1])
    I = I.reshape((channels * width * height))
    I = I / 255.

    center_y = 200.5
    center_x = 330.5
    deltaY = 1.
    deltaX = 5.
    sigmaY = 2
    sigmaX = 2


    def vectorize(*args):
        return [a.reshape((1,) + a.shape) for a in args]


    I, center_y, center_x, deltaY, deltaX, sigmaY, sigmaX = \
        vectorize(I, np.array(center_y), np.array(center_x), np.array(deltaY), np.array(deltaX), np.array(sigmaY), np.array(sigmaX))

    # import ipdb; ipdb.set_trace()

    W = do_read(I, center_y, center_x, deltaY, deltaX, sigmaY, sigmaX)
    I2 = do_write(W, center_y, center_x, deltaY, deltaX, sigmaY, sigmaX)


    def imagify(flat_image, h, w):
        image = flat_image.reshape([channels, h, w])
        image = image.transpose([1, 2, 0])
        return image / image.max()


    import pylab

    pylab.figure()
    pylab.gray()
    pylab.imshow(imagify(I, height, width), interpolation='nearest')

    pylab.figure()
    pylab.gray()
    pylab.imshow(imagify(W, N, N), interpolation='nearest')

    pylab.figure()
    pylab.gray()
    pylab.imshow(imagify(I2, height, width), interpolation='nearest')
    pylab.show(block=True)

    # import ipdb; ipdb.set_trace()
