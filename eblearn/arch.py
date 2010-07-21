from module import *

class layers (module_1_1):
    
    def __init__(self, *layers):
        self.layers = layers
        self.hidden = [state(()) for lyr in layers[:-1]]

    def forget(self):
        for lyr in self.layers: lyr.forget()

    def normalize(self):
        for lyr in self.layers: lyr.normalize()
    
    def fprop(self, input, output):
        inputs  = [input] + self.hidden
        outputs = self.hidden + [output]
        for (lyr, inp, outp) in zip(self.layers, inputs, outputs):
            lyr.fprop(inp, outp)
    
    def bprop_input(self, input, output):
        for h in self.hidden: h.clear_dx()
        inputs  = [input] + self.hidden
        outputs = self.hidden + [output]
        for (lyr, inp, outp) in zip(self.layers, inputs, outputs)[::-1]:
            lyr.bprop_input(inp, outp)
    def bprop_param(self, input, output):
        inputs  = [input] + self.hidden
        outputs = self.hidden + [output]
        for (lyr, inp, outp) in zip(self.layers, inputs, outputs)[::-1]:
            lyr.bprop_param(inp, outp)
    
    def bbprop_input(self, input, output):
        for h in self.hidden: h.clear_ddx()
        inputs  = [input] + self.hidden
        outputs = self.hidden + [output]
        for (lyr, inp, outp) in zip(self.layers, inputs, outputs)[::-1]:
            lyr.bbprop_input(inp, outp)
    def bbprop_param(self, input, output):
        inputs  = [input] + self.hidden
        outputs = self.hidden + [output]
        for (lyr, inp, outp) in zip(self.layers, inputs, outputs)[::-1]:
            lyr.bbprop_param(inp, outp)


class ebm_2 (module_2_1):
    def __init__(self, machine, cost):
        self.machine = machine
        self.cost    = cost
        self.machine_out = state(())

    def forget(self):
        self.machine.forget()
        self.cost.forget()

    def normalize(self):
        self.machine.normalize()
        self.cost.normalize()

    def fprop(self, input1, input2, output):
        self.machine.fprop(input1, self.machine_out)
        self.cost.fprop(self.machine_out, input2, output)

    def bprop_input(self, input1, input2, output):
        self.machine_out.clear_dx()
        self.cost.bprop_input(self.machine_out, input2, output)
        self.machine.bprop_input(input1, self.machine_out)
    def bprop_param(self, input1, input2, output):
        self.cost.bprop_param(self.machine_out, input2, output)
        self.machine.bprop_param(input1, self.machine_out)

    def bbprop_input(self, input1, input2, output):
        self.machine_out.clear_ddx()
        self.cost.bbprop_input(self.machine_out, input2, output)
        self.machine.bbprop_input(input1, self.machine_out)
    def bbprop_param(self, input1, input2, output):
        self.cost.bbprop_param(self.machine_out, input2, output)
        self.machine.bbprop_param(input1, self.machine_out)

