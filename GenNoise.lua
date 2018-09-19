require 'nn'
----------------------------------------------------------
-- GenNoise 
----------------------------------------------------------
-- Generates a new tensor with noise of spatial size as `input`
-- Forgets about `input` returning 0 gradInput.

local GenNoise, parent = torch.class('nn.GenNoise', 'nn.Module')

function  GenNoise:__init(num_planes)
    self.num_planes = num_planes
    self.mult = 1.0/255/2
end
function GenNoise:updateOutput(input)
    self.sz = input:size()

    self.sz_ = input:size()
    self.sz_[2] = self.num_planes

    self.output = self.output or input.new()
    self.output:resize(self.sz_)
    
    -- It is concated with normed data, so gen from N(0,1)
    self.output:normal(0,1):mul(2):add(-1):mul(self.mult)

   return self.output
end

function GenNoise:updateGradInput(input, gradOutput)
   self.gradInput = self.gradInput or gradOutput.new()
   self.gradInput:resizeAs(input):zero()
   
   return self.gradInput
end