from module import *

class layers_1 (module_1_1):
    
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

layers = layers_1


class layers_2 (module_2_2):
    
    def __init__(self, *layers):
        self.layers = layers
        self.hidden = [(state(()), state(())) for lyr in layers[:-1]]

    def forget(self):
        for lyr in self.layers: lyr.forget()

    def normalize(self):
        for lyr in self.layers: lyr.normalize()
    
    def fprop(self, input1, input2, output1, output2):
        inputs  = [(input1, input2)] + self.hidden
        outputs = self.hidden + [(output1, output2)]
        for (lyr, inp, outp) in zip(self.layers, inputs, outputs):
            lyr.fprop(inp[0], inp[1], outp[0], outp[1])
    
    def bprop_input(self, input, output):
        for (h1,h2) in self.hidden:
            h1.clear_dx(); h2.clear_dx()
        inputs  = [(input1, input2)] + self.hidden
        outputs = self.hidden + [(output1, output2)]
        for (lyr, inp, outp) in zip(self.layers, inputs, outputs)[::-1]:
            lyr.bprop_input(inp[0], inp[1], outp[0], outp[1])
    def bprop_param(self, input, output):
        inputs  = [(input1, input2)] + self.hidden
        outputs = self.hidden + [(output1, output2)]
        for (lyr, inp, outp) in zip(self.layers, inputs, outputs)[::-1]:
            lyr.bprop_param(inp[0], inp[1], outp[0], outp[1])
    
    def bbprop_input(self, input, output):
        for (h1,h2) in self.hidden:
            h1.clear_ddx(); h2.clear_ddx()
        inputs  = [(input1, input2)] + self.hidden
        outputs = self.hidden + [(output1, output2)]
        for (lyr, inp, outp) in zip(self.layers, inputs, outputs)[::-1]:
            lyr.bbprop_input(inp[0], inp[1], outp[0], outp[1])
    def bbprop_param(self, input, output):
        inputs  = [(input1, input2)] + self.hidden
        outputs = self.hidden + [(output1, output2)]
        for (lyr, inp, outp) in zip(self.layers, inputs, outputs)[::-1]:
            lyr.bbprop_param(inp[0], inp[1], outp[0], outp[1])


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

class filter_output_2_1 (module_2_1):
    def __init__(self, machine, filter):
        self.machine = machine
        self.filter  = filter
        self.hidden  = state(())

    def forget(self):
        self.machine.forget()
        self.filter.forget()

    def normalize(self):
        self.machine.forget()
        self.filter.forget()

    def fprop(self, input1, input2, output):
        self.machine.fprop(input1, input2, self.hidden)
        self.filter.fprop(self.hidden, output)

    def bprop_input(self, input1, input2, output):
        self.hidden.clear_dx()
        self.filter.bprop_input(self.hidden, output)
        self.machine.bprop_input(input1, input2, self.hidden)
    def bprop_param(self, input1, input2, output):
        self.filter.bprop_param(self.hidden, output)
        self.machine.bprop_param(input1, input2, self.hidden)

    def bbprop_input(self, input1, input2, output):
        self.hidden.clear_ddx()
        self.filter.bbprop_input(self.hidden, output)
        self.machine.bbprop_input(input1, input2, self.hidden)
    def bbprop_param(self, input1, input2, output):
        self.filter.bbprop_param(self.hidden, output)
        self.machine.bbprop_param(input1, input2, self.hidden)

class filter_input_2_1 (module_2_1):
    def __init__(self, machine, filter1, filter2):
        self.machine = machine
        self.filter1 = filter1
        self.filter2 = filter2
        self.hidden1 = state(())
        self.hidden2 = state(())

    def forget(self):
        self.filter1.forget()
        self.filter2.forget()
        self.machine.forget()

    def normalize(self):
        self.filter1.normalize()
        self.filter2.normalize()
        self.machine.normalize()

    def fprop(self, input1, input2, output):
        self.filter1.fprop(input1, self.hidden1)
        self.filter2.fprop(input2, self.hidden2)
        self.machine.fprop(self.hidden1, self.hidden2, output)

    def bprop_input(self, input1, input2, output):
        self.hidden1.clear_dx()
        self.hidden2.clear_dx()
        self.machine.bprop_input(self.hidden1, self.hidden2, output)
        self.filter1.bprop_input(input1, self.hidden1)
        self.filter2.bprop_input(input2, self.hidden2)
    def bprop_param(self, input1, input2, output):
        self.machine.bprop_param(self.hidden1, self.hidden2, output)
        self.filter1.bprop_param(input1, self.hidden1)
        self.filter2.bprop_param(input2, self.hidden2)

    def bbprop_input(self, input1, input2, output):
        self.hidden1.clear_ddx()
        self.hidden2.clear_ddx()
        self.machine.bbprop_input(self.hidden1, self.hidden2, output)
        self.filter1.bbprop_input(input1, self.hidden1)
        self.filter2.bbprop_input(input2, self.hidden2)
    def bbprop_param(self, input1, input2, output):
        self.machine.bbprop_param(self.hidden1, self.hidden2, output)
        self.filter1.bbprop_param(input1, self.hidden1)
        self.filter2.bbprop_param(input2, self.hidden2)

