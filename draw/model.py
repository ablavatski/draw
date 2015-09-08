from __future__ import division, print_function

import theano.tensor as T
from blocks.bricks.base import application
from blocks.bricks.recurrent import BaseRecurrent, recurrent
from blocks.bricks import Random, Initializable, MLP, Linear
from blocks.bricks import Identity

from attention import ZoomableAttentionWindow


# -----------------------------------------------------------------------------

class LocatorSampler(Initializable, Random):
    def __init__(self, input_dim, output_dim, **kwargs):
        super(LocatorSampler, self).__init__(**kwargs)

        self.prior_mean = 0.
        self.prior_log_sigma = 0.

        self.mean_transform = Linear(name=self.name + '_mean', input_dim=input_dim, output_dim=output_dim, weights_init=self.weights_init, biases_init=self.biases_init,
                                     use_bias=True)

        self.log_sigma_transform = Linear(name=self.name + '_log_sigma', input_dim=input_dim, output_dim=output_dim, weights_init=self.weights_init, biases_init=self.biases_init,
                                          use_bias=True)

        self.children = [self.mean_transform, self.log_sigma_transform]

    def get_dim(self, name):
        if name == 'input':
            return self.mean_transform.get_dim('input')
        elif name == 'output':
            return self.mean_transform.get_dim('output')
        else:
            raise ValueError

    @application(inputs=['x', 'u'], outputs=['z'])
    def sample(self, x, u):
        """Return a samples and the corresponding KL term

        Parameters
        ----------
        x : 

        Returns
        -------
        z : tensor.matrix
            Samples drawn from Q(z|x)
        
        """
        mean = self.mean_transform.apply(x)
        log_sigma = self.log_sigma_transform.apply(x)

        # Sample from mean-zeros std.-one Gaussian
        # u = self.theano_rng.normal(
        #            size=mean.shape, 
        #            avg=0., std=1.)

        # ... and scale/translate samples
        z = mean + T.exp(log_sigma) * u

        return z

    # @application(inputs=['n_samples'])
    @application(inputs=['u'], outputs=['z_prior'])
    def sample_from_prior(self, u):
        """Sample z from the prior distribution P_z.

        Parameters
        ----------
        u : tensor.matrix
            gaussian random source 

        Returns
        -------
        z : tensor.matrix
            samples 

        """
        z_dim = self.mean_transform.get_dim('output')

        # Sample from mean-zeros std.-one Gaussian
        # u = self.theano_rng.normal(
        #            size=(n_samples, z_dim),
        #            avg=0., std=1.)

        # ... and scale/translate samples
        z = self.prior_mean + T.exp(self.prior_log_sigma) * u
        # z.name("z_prior")

        return z


# -----------------------------------------------------------------------------


class LocatorReader(Initializable):
    def __init__(self, x_dim, dec_dim, channels, height, width, N, **kwargs):
        super(LocatorReader, self).__init__(name="reader", **kwargs)

        self.img_height = height
        self.img_width = width
        self.N = N
        self.x_dim = x_dim
        self.dec_dim = dec_dim
        self.output_dim = channels * N * N

        self.zoomer = ZoomableAttentionWindow(channels, height, width, N)
        self.readout = MLP(activations=[Identity()], dims=[dec_dim, 5], **kwargs)

        self.children = [self.readout]

    def get_dim(self, name):
        if name == 'input':
            return self.dec_dim
        elif name == 'x_dim':
            return self.x_dim
        elif name == 'output':
            return self.output_dim
        else:
            raise ValueError

    @application(inputs=['x', 'h_dec'], outputs=['r'])
    def apply(self, x, h_dec):
        l = self.readout.apply(h_dec)

        center_y, center_x, delta, sigma, gamma = self.zoomer.nn2att(l)

        w = gamma * self.zoomer.read(x, center_y, center_x, delta, sigma)

        return w


# -----------------------------------------------------------------------------

class LocatorWriter(Initializable):
    def __init__(self, input_dim, output_dim, channels, width, height, N, **kwargs):
        super(LocatorWriter, self).__init__(name="writer", **kwargs)

        self.channels = channels
        self.img_width = width
        self.img_height = height
        self.N = N
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.zoomer = ZoomableAttentionWindow(channels, height, width, N)
        self.z_trafo = Linear(name=self.name + '_ztrafo', input_dim=input_dim, output_dim=5, weights_init=self.weights_init, biases_init=self.biases_init, use_bias=True)

        self.children = [self.z_trafo]

    @application(inputs=['h'], outputs=['center_y', 'center_x', 'delta'])
    def apply_detailed(self, h):
        l = self.z_trafo.apply(h)

        center_y, center_x, delta, sigma, gamma = self.zoomer.nn2att(l)

        return center_y, center_x, delta


# -----------------------------------------------------------------------------


class LocatorModel(BaseRecurrent, Initializable, Random):
    def __init__(self, n_iter, reader,
                 encoder_mlp, encoder_rnn, sampler,
                 decoder_mlp, decoder_rnn, writer, **kwargs):
        super(LocatorModel, self).__init__(**kwargs)
        self.n_iter = n_iter

        self.reader = reader
        self.encoder_mlp = encoder_mlp
        self.encoder_rnn = encoder_rnn
        self.sampler = sampler
        self.decoder_mlp = decoder_mlp
        self.decoder_rnn = decoder_rnn
        self.writer = writer

        self.children = [self.reader, self.encoder_mlp, self.encoder_rnn, self.sampler,
                         self.writer, self.decoder_mlp, self.decoder_rnn]

    def get_dim(self, name):
        if name == 'h_enc':
            return self.encoder_rnn.get_dim('states')
        elif name == 'c_enc':
            return self.encoder_rnn.get_dim('cells')
        elif name in ['z', 'z_mean', 'z_log_sigma']:
            return self.sampler.get_dim('output')
        elif name == 'h_dec':
            return self.decoder_rnn.get_dim('states')
        elif name == 'c_dec':
            return self.decoder_rnn.get_dim('cells')
        elif name == 'center_y':
            return 0
        elif name == 'center_x':
            return 0
        elif name == 'delta':
            return 0
        else:
            super(LocatorModel, self).get_dim(name)

    # ------------------------------------------------------------------------

    @recurrent(sequences=['u'], contexts=['x'], states=['center_y', 'center_x', 'delta', 'h_enc', 'c_enc', 'z', 'h_dec', 'c_dec'],
               outputs=['center_y', 'center_x', 'delta', 'h_enc', 'c_enc', 'z', 'h_dec', 'c_dec'])
    def apply(self, u, center_y, center_x, delta, h_enc, c_enc, z, h_dec, c_dec, x):
        r = self.reader.apply(x, h_dec)
        i_enc = self.encoder_mlp.apply(T.concatenate([r, h_dec], axis=1))
        h_enc, c_enc = self.encoder_rnn.apply(states=h_enc, cells=c_enc, inputs=i_enc, iterate=False)
        z = self.sampler.sample(h_enc, u)

        i_dec = self.decoder_mlp.apply(z)
        h_dec, c_dec = self.decoder_rnn.apply(states=h_dec, cells=c_dec, inputs=i_dec, iterate=False)
        center_y, center_x, delta = self.writer.apply_detailed(h_dec)

        return center_y, center_x, delta, h_enc, c_enc, z, h_dec, c_dec

    # @recurrent(sequences=['u'], contexts=[], states=['c', 'h_dec', 'c_dec'], outputs=['c', 'h_dec', 'c_dec'])
    # def decode(self, u, c, h_dec, c_dec):
    #     batch_size = c.shape[0]
    #
    #     z = self.sampler.sample_from_prior(u)
    #     i_dec = self.decoder_mlp.apply(z)
    #     h_dec, c_dec = self.decoder_rnn.apply(
    #         states=h_dec, cells=c_dec,
    #         inputs=i_dec, iterate=False)
    #     c = c + self.writer.apply(h_dec)
    #     return c, h_dec, c_dec

    # ------------------------------------------------------------------------

    @application(inputs=['features'], outputs=['center_y', 'center_x', 'delta'])
    def calculate(self, features):
        batch_size = features.shape[0]
        dim_z = self.get_dim('z')

        # Sample from mean-zeros std.-one Gaussian
        u = self.theano_rng.normal(size=(self.n_iter, batch_size, dim_z), avg=0., std=1.)

        center_y, center_x, delta, h_enc, c_enc, z, h_dec, c_dec = self.apply(x=features, u=u)

        center_y = center_y[-1, :]
        center_y.name = "center_y"

        center_x = center_x[-1, :]
        center_x.name = "center_x"

        delta = delta[-1, :]
        delta.name = 'delta'

        return center_y, center_x, delta

        # @application(inputs=['n_samples'], outputs=['samples'])
        # def sample(self, n_samples):
        #     """Sample from model.
        #
        #     Returns
        #     -------
        #
        #     samples : tensor3 (n_samples, n_iter, x_dim)
        #     """
        #
        #     # Sample from mean-zeros std.-one Gaussian
        #     u_dim = self.sampler.mean_transform.get_dim('output')
        #     u = self.theano_rng.normal(
        #         size=(self.n_iter, n_samples, u_dim),
        #         avg=0., std=1.)
        #
        #     c, _, _, = self.decode(u)
        #     # c, _, _, center_y, center_x, delta = self.decode(u)
        #     return T.nnet.sigmoid(c)
