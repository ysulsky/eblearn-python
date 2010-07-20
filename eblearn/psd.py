from eblearn import *

# TODO code optimizers

class psd_codec (module_2_1):
    def __init__(self, 
                 encoder, encoder_cost, code_penalty, decoder, decoder_cost,
                 weight_encoder = 1., weight_code  = 1., weight_decoder = 1.,
                 code_optimizer = None):
        vals = dict(locals())
        del vals['self']
        self.__dict__.update(vals)

        self.encoder_out = state(())
        self.code        = state(())
        self.decoder_out = state(())
        
        self.encoder_energy = state((1,))
        self.code_energy    = state((1,))
        self.decoder_energy = state((1,))
        
        # you probably want to train these with separate update parameters
        # so set {encoder,decoder}.parameter.indep = True
        self._merge_parameters(encoder)
        self._merge_parameters(encoder_cost)
        self._merge_parameters(code_penalty)
        self._merge_parameters(decoder)
        self._merge_parameters(decoder_cost)
        
    def forget(self):
        self.encoder.forget()
        self.encoder_cost.forget()
        self.code_penalty.forget()
        self.decoder.forget()
        self.decoder_cost.forget()
        
        self.decoder.normalize()

    def normalize(self):
        self.decoder.normalize()

    def fprop(self, input, output, energy):
        encoder_out, code, decoder_out = \
            self.encoder_out, self.code, self.decoder_out
        
        # input                         -> encoder_out
        self.encoder.fprop(input, encoder_out)

        # encoder_out                   -> code
        code.resize(encoder_out.shape)
        code.x[:] = encoder_out.x

        # code                          -> decoder_out
        self.decoder.fprop(code, decoder_out)

        # decoder_out, output           -> decoder_energy
        self.decoder_cost.fprop(decoder_out, output, self.decoder_energy)

        # encoder_out, code             -> encoder_energy
        self.encoder_cost.fprop(encoder_out, code, self.encoder_energy)

        # code                          -> code_energy
        self.code_penalty.fprop(code, self.code_energy)

        # {encoder,cost,decoder}_energy -> energy
        energy.resize((1,))
        energy.x[:] = (self.weight_encoder * self.encoder_energy.x +
                       self.weight_code    * self.code_energy.x    +
                       self.weight_decoder * self.decoder_energy.x)

        # optimize code
        if self.code_optimizer is not None:
            self.code_optimizer.opt(input, output, energy)
    
    
    def bprop_input(self, input, output, energy):
        encoder_out, code, decoder_out = \
            self.encoder_out, self.code, self.decoder_out

        encoder_out.clear_dx()
        code.clear_dx()
        decoder_out.clear_dx()

        # {encoder,cost,decoder}_energy <- energy
        self.encoder_energy.dx[:] = energy.dx * self.weight_encoder
        self.code_energy.dx[:]    = energy.dx * self.weight_code
        self.decoder_energy.dx[:] = energy.dx * self.weight_decoder
        
        # code                          <- code_energy
        self.code_penalty.bprop_input(code, self.code_energy)
        
        # encoder_out, code             <- encoder_energy
        self.encoder_cost.bprop_input(encoder_out, code, self.encoder_energy)
        
        # decoder_out, output           <- decoder_energy
        self.decoder_cost.bprop_input(decoder_out, output, self.decoder_energy)

        # code                          <- decoder_out
        self.decoder.bprop_input(code, decoder_out)

        # input                         <- encoder_out
        self.encoder.bprop_input(input, encoder_out)

    
    def bprop_param(self, input, output, energy):
        encoder_out, code, decoder_out = \
            self.encoder_out, self.code, self.decoder_out

        # code                          <- code_energy
        self.code_penalty.bprop_param(code, self.code_energy)
        
        # encoder_out, code             <- encoder_energy
        self.encoder_cost.bprop_param(encoder_out, code, self.encoder_energy)
        
        # decoder_out, output           <- decoder_energy
        self.decoder_cost.bprop_param(decoder_out, output, self.decoder_energy)

        # code                          <- decoder_out
        self.decoder.bprop_param(code, decoder_out)

        # input                         <- encoder_out
        self.encoder.bprop_param(input, encoder_out)


    def bbprop_input(self, input, output, energy):
        encoder_out, code, decoder_out = \
            self.encoder_out, self.code, self.decoder_out

        encoder_out.clear_ddx()
        code.clear_ddx()
        decoder_out.clear_ddx()

        # {encoder,cost,decoder}_energy <- energy
        self.encoder_energy.ddx[:] = energy.ddx
        self.code_energy.ddx[:]    = energy.ddx
        self.decoder_energy.ddx[:] = energy.ddx
        
        # code                          <- code_energy
        self.code_penalty.bbprop_input(code, self.code_energy)
        
        # encoder_out, code             <- encoder_energy
        self.encoder_cost.bbprop_input(encoder_out, code, self.encoder_energy)
        
        # decoder_out, output           <- decoder_energy
        self.decoder_cost.bbprop_input(decoder_out, output, self.decoder_energy)

        # code                          <- decoder_out
        self.decoder.bbprop_input(code, decoder_out)

        # input                         <- encoder_out
        self.encoder.bbprop_input(input, encoder_out)

    
    def bbprop_param(self, input, output, energy):
        encoder_out, code, decoder_out = \
            self.encoder_out, self.code, self.decoder_out

        # code                          <- code_energy
        self.code_penalty.bbprop_param(code, self.code_energy)
        
        # encoder_out, code             <- encoder_energy
        self.encoder_cost.bbprop_param(encoder_out, code, self.encoder_energy)
        
        # decoder_out, output           <- decoder_energy
        self.decoder_cost.bbprop_param(decoder_out, output, self.decoder_energy)

        # code                          <- decoder_out
        self.decoder.bbprop_param(code, decoder_out)

        # input                         <- encoder_out
        self.encoder.bbprop_param(input, encoder_out)

