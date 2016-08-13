require 'nn'
require 'dpnn'
require 'nngraph'
local nninit = require 'nninit'

local nmse, parent = torch.class('nn.NormMSECriterion', 'nn.Criterion')
function nmse:__init()
  parent.__init(self)
  self.mean = nil
  self.std = nil
  self.mse = nn.MSECriterion()
end
function nmse:setMeanStd(mean, std)
  self.mean = mean:view(1, -1):clone()
  self.std = std:view(1, -1):clone()
end
-- target is clean, input is noisy
function nmse:updateOutput(input, target)
  if self.mean and self.std then
    self.input_norm = (input - self.mean:expand(input:size())):cdiv(self.std:expand(input:size()))
    self.output = self.mse:updateOutput(self.input_norm, target)
  else
    self.output = self.mse:updateOutput(input, target)
  end
  return self.output
end
function nmse:updateGradInput(input, target)
  if self.mean and self.std then
    self.gradInput = self.mse:updateGradInput(self.input_norm, target)
    self.gradInput:cdiv(self.std:expand(input:size())) -- bug??
  else
    self.gradInput = self.mse:updateGradInput(input, target)
  end
  return self.gradInput
end
function nmse:type(...)
  parent.type(self, ...)
  self.mse:type(...)
end

function createLadderAE(opt)
  local noise_level = opt.noise_level or 0.3
  local layer_sizes = {1000,500,250,250,250}
  layer_sizes[0] = 784

  -- Encoder
  local z, z_bn, z_noise = {}, {}, {}
  local input = nn.Identity()()
  z[0] = nn.Reshape(layer_sizes[0], true)(input)
  z_noise[0] = nn.WhiteNoise(0, noise_level)(z[0])
  prev_out = z_noise[0]

  local bn_layers = {}

  for i = 1,#layer_sizes do
    local sz = layer_sizes[i]
    z[i] = nn.Linear(layer_sizes[i-1], sz)(prev_out)
    bn_layers[i] = nn.BatchNormalization(sz, nil, nil, false) 
    z_bn[i] = bn_layers[i](z[i])
    z_noise[i] = nn.WhiteNoise(0, noise_level)(z_bn[i])
    prev_out = nn.ReLU(true)(nn.Add(sz)(z_noise[i]))
  end
  local y = nn.Linear(250, 10)(prev_out)
  local y_bn = nn.BatchNormalization(10)(y)
  local y_softmax = nn.SoftMax()(y_bn)

  -- Decoder
  local up_size = 10
  local up_layer = y_softmax
  local u, z_hat = {}, {}, {}, {}
  for i = #layer_sizes,0,-1 do
    local sz = layer_sizes[i]

    u[i] = nn.BatchNormalization(sz, nil, nil, false)
            (nn.Linear(up_size, layer_sizes[i])(up_layer))
   
    local g

    local function getMul(sz, i) 
      if i == nil then return nn.CMul(sz) end
      return nn.CMul(sz):init('weight', nninit.constant, i)
    end
    local function getAdd(sz, i)
      if i == nil then return nn.Add(sz) end
      return nn.Add(sz):init('bias', nninit.constant, i)
    end
    if opt.comb_func == 'vanilla' then
      g = function(z_noise, u)
        local function AffineMul(sz, x, y)
          local xy = nn.CMulTable()({x, y})
          return getAdd(sz,0)(nn.CAddTable()({getMul(sz,1)(x), getMul(sz,0)(y), getMul(sz,0)(xy)}))
        end
        local a1 = AffineMul(sz, z_noise, u)
        local a2 = AffineMul(sz, z_noise, u)
        return nn.CAddTable()({a1, getMul(sz,1)(nn.Sigmoid()(a2))})
      end
    elseif opt.comb_func == 'vanilla-randinit' then
      g = function(z_noise, u)
        local function AffineMul(sz, x, y)
          local xy = nn.CMulTable()({x, y})
          return getAdd(sz)(nn.CAddTable()({getMul(sz)(x), getMul(sz)(y), getMul(sz)(xy)}))
        end
        local a1 = AffineMul(sz, z_noise, u)
        local a2 = AffineMul(sz, z_noise, u)
        return nn.CAddTable()({a1, getMul(sz)(nn.Sigmoid()(a2))})
      end
    elseif opt.comb_func == 'gaussian' then
      local function AddTwo(x, y) return nn.CAddTable()({x,y}) end
      local function SubTwo(x, y) return nn.CSubTable()({x,y}) end
      local function MulTwo(x, y) return nn.CMulTable()({x,y}) end
      local function Affine(sz, x) return getAdd(sz)(getMul(sz)(x)) end
      g = function (z_noise, u)
        local mu = AddTwo(Affine(sz, nn.Sigmoid()(Affine(sz, u))), getMul(sz)(u))
        local nu = AddTwo(Affine(sz, nn.Sigmoid()(Affine(sz, u))), getMul(sz)(u))
        return AddTwo(MulTwo(SubTwo(z_noise, mu), nu), mu)
      end
    else
      error('unrecognized combinator function')
    end
    z_hat[i] = g(z_noise[i], u[i]) 

    up_size = sz
    up_layer = z_hat[i]
  end

  local outputs = {y_bn, z[0]}
  for i=1,#z_bn do table.insert(outputs, z_bn[i]) end
  for i=0,#z_hat do table.insert(outputs, z_hat[i]) end

  local model = nn.gModule({input}, outputs)

  model.layer_sizes = layer_sizes
  model.bn_layers = bn_layers

  -- construct criterion
  local criterion = nn.ParallelCriterion()
  criterion:add(nn.CrossEntropyCriterion())

  -- dummy criterion for noisy output
  for i = 0,#layer_sizes do
    criterion:add(nn.MSECriterion(), 0) 
  end

  for i = 0,#layer_sizes do
    criterion:add(nn.NormMSECriterion())
  end
  return model, criterion
end
