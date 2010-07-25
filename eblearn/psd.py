from module     import *
from arch       import *
from datasource import *
from trainer    import *
from parameter  import *

class psd_codec (module_2_1):

    class dummy_dsource (eb_dsource):
        def __init__(self):
            self.cur_out = None
            super(psd_codec.dummy_dsource, self).__init__()
        def set_output(self, output):
            self.cur_out = output
        def size(self):  return 1
        def shape(self): raise NotImplementedError()
        def fprop(self, input, output):
            # input isn't touched, set directly in trainer
            output.resize(self.cur_out.shape)
            output.x[:] = self.cur_out.x[:]

    class ebm_decoder (ebm_2):
        def __init__(self, decoder, cost):
            super(psd_codec.ebm_decoder, self).__init__(decoder, cost)
        def bprop(self, input1, input2, output):
            self.bprop_input(input1, input2, output)
        def bbprop(self, input1, input2, output):
            self.bbprop_input(input1, input2, output)

    class psd_costs (module_2_1):
        def __init__(self,
                     encoder_cost,   code_penalty, decoder_cost,
                     weight_encoder, weight_code,  weight_decoder,
                     encoder_out,    code):
            vals = dict(locals())
            del vals['self']
            self.__dict__.update(vals)

            self.encoder_energy = state((1,))
            self.code_energy    = state((1,))
            self.decoder_energy = state((1,))

        def forget(self):
            self.encoder_cost.forget()
            self.code_penalty.forget()
            self.decoder_cost.forget()

        def normalize():
            self.encoder_cost.normalize()
            self.code_penalty.normalize()
            self.decoder_cost.normalize()
        
        def fprop(self, decoder_out, output, energy):
            encoder_out, code = self.encoder_out, self.code
            
            self.decoder_cost.fprop(decoder_out, output, self.decoder_energy)
            self.encoder_cost.fprop(encoder_out, code, self.encoder_energy)
            self.code_penalty.fprop(code, self.code_energy)

            energy.resize((1,))
            energy.x[:] = (self.weight_encoder * self.encoder_energy.x +
                           self.weight_code    * self.code_energy.x    +
                           self.weight_decoder * self.decoder_energy.x)

        def bprop_input(self, decoder_out, output, energy):
            encoder_out, code = self.encoder_out, self.code
            encoder_energy, code_energy, decoder_energy =\
                self.encoder_energy, self.code_energy, self.decoder_energy
            
            encoder_energy.dx[:] = energy.dx * self.weight_encoder
            code_energy.dx[:]    = energy.dx * self.weight_code
            decoder_energy.dx[:] = energy.dx * self.weight_decoder
        
            self.code_penalty.bprop_input(code, code_energy)
            self.encoder_cost.bprop_input(encoder_out, code,   encoder_energy)
            self.decoder_cost.bprop_input(decoder_out, output, decoder_energy)

        def bprop_param(self, decoder_out, output, energy):
            encoder_out, code = self.encoder_out, self.code
            encoder_energy, code_energy, decoder_energy =\
                self.encoder_energy, self.code_energy, self.decoder_energy

            self.code_penalty.bprop_param(code, code_energy)
            self.encoder_cost.bprop_param(encoder_out, code,   encoder_energy)
            self.decoder_cost.bprop_param(decoder_out, output, decoder_energy)

        def bbprop_input(self, decoder_out, output, energy):
            encoder_out, code = self.encoder_out, self.code
            encoder_energy, code_energy, decoder_energy =\
                self.encoder_energy, self.code_energy, self.decoder_energy
            
            encoder_energy.dx[:] = energy.ddx
            code_energy.dx[:]    = energy.ddx
            decoder_energy.dx[:] = energy.ddx
        
            self.code_penalty.bbprop_input(code, code_energy)
            self.encoder_cost.bbprop_input(encoder_out, code,   encoder_energy)
            self.decoder_cost.bbprop_input(decoder_out, output, decoder_energy)

        def bbprop_param(self, decoder_out, output, energy):
            encoder_out, code = self.encoder_out, self.code
            encoder_energy, code_energy, decoder_energy =\
                self.encoder_energy, self.code_energy, self.decoder_energy

            self.code_penalty.bbprop_param(code, code_energy)
            self.encoder_cost.bbprop_param(encoder_out, code,   encoder_energy)
            self.decoder_cost.bbprop_param(decoder_out, output, decoder_energy)

    
    def __init__(self, 
                 encoder, encoder_cost, code_penalty, decoder, decoder_cost,
                 weight_encoder = 1., weight_code  = 1., weight_decoder = 1.,
                 optimize_code = True):
        vals = dict(locals())
        del vals['self']
        self.__dict__.update(vals)

        self.encoder_out = state(())
        self.code        = state(())
        self.decoder_out = state(())

        self.costs = \
            psd_codec.psd_costs(encoder_cost, code_penalty, decoder_cost,
                                weight_encoder, weight_code,  weight_decoder,
                                self.encoder_out, self.code)
        
        self.code_parameter = parameter()
        self.code_parameter.append(self.code)

        self.code_trainer = None
        if optimize_code:
            self.code_trainer = \
                eb_trainer(self.code_parameter,
                           psd_codec.ebm_decoder(self.decoder, self.costs),
                           psd_codec.dummy_dsource(),
                           hess_interval    = 0,
                           report_interval  = 0,
                           do_normalization = False,
                           quiet            = True,
                           auto_forget      = False)
            self.code_trainer.input  = self.code

            self.code_feval = feval_from_trainer(self.code_trainer)
            self.code_parameter.updater = gd_linesearch_update(self.code_feval)
    
    def forget(self):
        self.encoder.forget()
        self.decoder.forget()
        self.costs.forget()
        
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
        
        # optimize code
        if self.code_trainer is not None:
            self.code_trainer.ds_train.set_output(output)
            self.code_trainer.train()
        
        # code                          -> decoder_out
        self.decoder.fprop(code, decoder_out)
        
        self.costs.fprop(decoder_out, output, energy)
    
    
    def bprop_input(self, input, output, energy):
        encoder_out, code, decoder_out = \
            self.encoder_out, self.code, self.decoder_out

        encoder_out.clear_dx()
        code.clear_dx()
        decoder_out.clear_dx()

        self.costs.bprop_input(decoder_out, output, energy)
        self.decoder.bprop_input(code, decoder_out)
        self.encoder.bprop_input(input, encoder_out)

    
    def bprop_param(self, input, output, energy):
        encoder_out, code, decoder_out = \
            self.encoder_out, self.code, self.decoder_out

        self.costs.bprop_param(decoder_out, output, energy)
        self.decoder.bprop_param(code, decoder_out)
        self.encoder.bprop_param(input, encoder_out)


    def bbprop_input(self, input, output, energy):
        encoder_out, code, decoder_out = \
            self.encoder_out, self.code, self.decoder_out

        encoder_out.clear_ddx()
        code.clear_ddx()
        decoder_out.clear_ddx()

        self.costs.bbprop_input(decoder_out, output, energy)
        self.decoder.bbprop_input(code, decoder_out)
        self.encoder.bbprop_input(input, encoder_out)

    
    def bbprop_param(self, input, output, energy):
        encoder_out, code, decoder_out = \
            self.encoder_out, self.code, self.decoder_out

        self.costs.bbprop_param(decoder_out, output, energy)
        self.decoder.bbprop_param(code, decoder_out)
        self.encoder.bbprop_param(input, encoder_out)

# for pickling
dummy_dsource = psd_codec.dummy_dsource
ebm_decoder   = psd_codec.ebm_decoder
psd_costs     = psd_codec.psd_costs

