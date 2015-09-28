from __future__ import division, print_function

import theano.tensor as T
from blocks.bricks.base import application
from blocks.bricks.recurrent import BaseRecurrent, recurrent
from blocks.bricks import Random, Initializable, MLP, Linear
from blocks.bricks import Identity, Rectifier, Tanh

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
                 decoder_mlp, decoder_rnn, **kwargs):
        super(LocatorModel, self).__init__(**kwargs)
        self.n_iter = n_iter

        self.reader = reader
        self.encoder_mlp = encoder_mlp
        self.encoder_rnn = encoder_rnn
        self.sampler = sampler
        self.decoder_mlp = decoder_mlp
        self.decoder_rnn = decoder_rnn

        self.children = [self.reader, self.encoder_mlp, self.encoder_rnn, self.sampler,
                         self.decoder_mlp, self.decoder_rnn]

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
        center_y, center_x, delta = self.reader.obtain(h_dec)

        return center_y, center_x, delta, h_enc, c_enc, z, h_dec, c_dec

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

    # ------------------------------------------------------------------------

    @application(inputs=['features', 'batch_size'], outputs=['center_y', 'center_x', 'delta'])
    def find(self, features, batch_size):
        dim_z = self.get_dim('z')

        # Sample from mean-zeros std.-one Gaussian
        u = self.theano_rng.normal(size=(self.n_iter, batch_size, dim_z), avg=0., std=1.)
        center_y, center_x, delta, h_enc, c_enc, z, h_dec, c_dec = self.apply(x=features, u=u)

        return center_y, center_x, delta


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
        # elif name == 'output':
        #     return self.output_dim
        else:
            raise ValueError

    @application(inputs=['x', 'h_dec'], outputs=['r', 'l'])
    def apply(self, x, h_dec):
        l = self.readout.apply(h_dec)

        center_y, center_x, delta, sigma, gamma = self.zoomer.nn2att(l)

        w = gamma * self.zoomer.read(x, center_y, center_x, delta, sigma)

        return w, l

    @application(inputs=['h_dec'], outputs=['center_y', 'center_x', 'delta'])
    def apply_l(self, h_dec):
        l = self.readout.apply(h_dec)

        center_y, center_x, delta = self.zoomer.nn2att_wn(l)

        return center_y, center_x, delta


class Representer(Initializable):
    def __init__(self, representation_mlp, **kwargs):
        super(Representer, self).__init__(name="representer", **kwargs)

        self.representation_mlp = representation_mlp
        self.r_trafo = Linear(name=representation_mlp.name + '_trafo', input_dim=representation_mlp.output_dim, output_dim=representation_mlp.output_dim, weights_init=self.weights_init,
                              biases_init=self.biases_init, use_bias=True)

        self.children = [self.representation_mlp, self.r_trafo]

    def get_dim(self, name):
        if name == 'input':
            return self.representation_mlp.input_dim
        elif name == 'output':
            return self.representation_mlp.output_dim
        else:
            raise ValueError

    @application(inputs=['r'], outputs=['l_repr'])
    def apply(self, r):

        i_repr = self.representation_mlp.apply(r)
        l_repr = self.r_trafo.apply(i_repr)

        return l_repr


class Locater(Initializable):
    def __init__(self, location_mlp, **kwargs):
        super(Locater, self).__init__(name="locater", **kwargs)

        self.location_mlp = location_mlp

        self.l_trafo = Linear(name=location_mlp.name + '_trafo', input_dim=location_mlp.output_dim, output_dim=location_mlp.output_dim, weights_init=self.weights_init,
                              biases_init=self.biases_init,
                              use_bias=True)

        self.children = [self.location_mlp, self.l_trafo]

    def get_dim(self, name):
        if name == 'input':
            return self.location_mlp.input_dim
        elif name == 'output':
            return self.location_mlp.output_dim
        else:
            raise ValueError

    @application(inputs=['l'], outputs=['l_loc'])
    def apply(self, l):

        i_loc = self.location_mlp.apply(l)
        l_loc = self.l_trafo.apply(i_loc)

        return l_loc


class SimpleLocatorModel1LSTM(BaseRecurrent, Initializable, Random):
    def __init__(self, n_iter, reader, locater, representer, decoder_mlp, decoder_rnn, **kwargs):
        super(SimpleLocatorModel1LSTM, self).__init__(**kwargs)
        self.n_iter = n_iter

        self.reader = reader
        self.representer = representer
        self.locater = locater
        self.decoder_mlp = decoder_mlp
        self.decoder_rnn = decoder_rnn

        self.children = [self.reader, self.locater, self.representer, self.decoder_mlp, self.decoder_rnn]

    def get_dim(self, name):
        if name == 'h_dec':
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
            super(SimpleLocatorModel1LSTM, self).get_dim(name)

    @recurrent(sequences=[], contexts=['x', 'n_steps', 'batch_size'], states=['h_dec', 'c_dec'],
               outputs=['center_y', 'center_x', 'delta', 'h_dec', 'c_dec'])
    def apply(self, h_dec, c_dec, x, n_steps, batch_size):
        r, l = self.reader.apply(x, h_dec)

        l_repr = self.representer.apply(r)
        l_loc = self.locater.apply(l)

        i_dec = self.decoder_mlp.apply(T.concatenate([l_repr, l_loc], axis=1))
        h_dec, c_dec = self.decoder_rnn.apply(states=h_dec, cells=c_dec, inputs=i_dec, iterate=False)

        center_y, center_x, delta = self.reader.apply_l(h_dec)

        return center_y, center_x, delta, h_dec, c_dec

    @application(inputs=['features'], outputs=['center_y', 'center_x', 'delta'])
    def calculate(self, features):
        batch_size = features.shape[0]

        center_y, center_x, delta, h_dec, c_dec = self.apply(x=features, n_steps=self.n_iter, batch_size=batch_size)

        # center_y = center_y[-1, :]
        # center_y.name = "center_y"

        # center_x = center_x[-1, :]
        # center_x.name = "center_x"

        # delta = delta[-1, :]
        # delta.name = 'delta'

        return center_y, center_x, delta

    @application(inputs=['features', 'batch_size'], outputs=['center_y', 'center_x', 'delta'])
    def find(self, features, batch_size):

        center_y, center_x, delta, h_dec, c_dec = self.apply(x=features, n_steps=self.n_iter, batch_size=batch_size)

        return center_y, center_x, delta


class SimpleLocatorModel2LSTM(BaseRecurrent, Initializable):
    def __init__(self, n_iter, reader, encoder_mlp, encoder_rnn, decoder_mlp, decoder_rnn, **kwargs):
        super(SimpleLocatorModel2LSTM, self).__init__(**kwargs)
        self.n_iter = n_iter

        self.reader = reader
        self.encoder_mlp = encoder_mlp
        self.encoder_rnn = encoder_rnn
        self.decoder_mlp = decoder_mlp
        self.decoder_rnn = decoder_rnn

        self.children = [self.reader, self.encoder_mlp, self.encoder_rnn, self.decoder_mlp, self.decoder_rnn]

    def get_dim(self, name):
        if name == 'h_enc':
            return self.encoder_rnn.get_dim('states')
        elif name == 'c_enc':
            return self.encoder_rnn.get_dim('cells')
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
            super(SimpleLocatorModel2LSTM, self).get_dim(name)

    @recurrent(sequences=[], contexts=['x', 'n_steps', 'batch_size'], states=['h_enc', 'c_enc', 'h_dec', 'c_dec'],
               outputs=['center_y', 'center_x', 'delta', 'h_enc', 'c_enc', 'h_dec', 'c_dec'])
    def apply(self, h_enc, c_enc, h_dec, c_dec, x, n_steps, batch_size):
        r, l = self.reader.apply(x, h_dec)

        i_enc = self.encoder_mlp.apply(T.concatenate([r, h_dec], axis=1))
        h_enc, c_enc = self.encoder_rnn.apply(states=h_enc, cells=c_enc, inputs=i_enc, iterate=False)

        i_dec = self.decoder_mlp.apply(h_enc)
        h_dec, c_dec = self.decoder_rnn.apply(states=h_dec, cells=c_dec, inputs=i_dec, iterate=False)

        center_y, center_x, delta = self.reader.apply_l(h_dec)

        return center_y, center_x, delta, h_enc, c_enc, h_dec, c_dec

    @application(inputs=['features'], outputs=['center_y', 'center_x', 'delta'])
    def calculate(self, features):
        batch_size = features.shape[0]

        center_y, center_x, delta, h_enc, c_enc, h_dec, c_dec = self.apply(x=features, n_steps=self.n_iter, batch_size=batch_size)

        # center_y = center_y[-1, :]
        # center_y.name = "center_y"

        # center_x = center_x[-1, :]
        # center_x.name = "center_x"

        # delta = delta[-1, :]
        # delta.name = 'delta'

        return center_y, center_x, delta

    @application(inputs=['features', 'batch_size'], outputs=['center_y', 'center_x', 'delta'])
    def find(self, features, batch_size):

        center_y, center_x, delta, h_enc, c_enc, h_dec, c_dec = self.apply(x=features, n_steps=self.n_iter, batch_size=batch_size)

        return center_y, center_x, delta
