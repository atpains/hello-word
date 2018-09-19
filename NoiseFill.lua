require 'nn'
----------------------------------------------------------
-- NoiseFill 
----------------------------------------------------------
-- Fills last `num_noise_channels` channels of an existing `input` tensor with noise. 
local NoiseFill, parent = torch.class('nn.NoiseFill', 'nn.Module')

function NoiseFill:__init(num_noise_channels, mult)
  parent.__init(self)

  -- last `num_noise_channels` maps will be filled with noise
  self.num_noise_channels = num_noise_channels
  self.mult = mult
  self.buffer = torch.Tensor()
end

function NoiseFill:updateOutput(input)
  --self.output = self.output or input:new()
  --self.output:resizeAs(input)

  local N = input:size(1)
  local C, H, W
  C, H, W = input:size(2), input:size(3), input:size(4)
  self.output:resize(N, C + self.num_noise_channels, H, W)

  self.output:narrow(2,1,C):copy(input:narrow(2,1,C))

  self.output:narrow(2,C+1, self.num_noise_channels):uniform():mul(2):add(-1):mul(self.mult)

  --[[
  -- copy non-noise part
  if self.num_noise_channels ~= input:size(2) then
    local ch_to_copy = input:size(2) - self.num_noise_channels
    self.output:narrow(2,1,ch_to_copy):copy(input:narrow(2,1,ch_to_copy))
  end


  -- fill noise
  if self.num_noise_channels > 0 then
    local num_channels = input:size(2)
    local first_noise_channel = num_channels - self.num_noise_channels + 1
    self.output:narrow(2,first_noise_channel, self.num_noise_channels):uniform():mul(2):add(-1):mul(self.mult)
  end
  ]]--
  return self.output
end

function NoiseFill:updateGradInput(input, gradOutput)
  --[[
  local N = input:size(1)
  local C, H, W
  C, H, W = input:size(2), input:size(3), input:size(4)
  self.buffer:resize(N, C, H, W)
  self.buffer:copy(gradOutput:narrow(2,1,C))
  self.gradInput = self.buffer
  ]]--
  self.gradInput = gradOutput:narrow(2,1,input:size(2))
  return self.gradInput
end