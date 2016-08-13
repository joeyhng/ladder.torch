require 'optim'
require 'model'
require 'MnistLoader'
local Trainer = require 'train'

torch.setdefaulttensortype('torch.FloatTensor')

cmd = torch.CmdLine()
cmd:option('-gpuid', 0, 'GPU ID (only using cuda)')
cmd:option('-batch_size', 100, 'batch size')
cmd:option('-num_labels', 100, 'batch size')
cmd:option('-learning_rate', 0.0002, 'learning rate')
cmd:option('-comb_func', 'vanilla-randinit', 'combinator function g')

cmd:option('-lr_decay_iter', 50000, 'learning rate decay iter')
cmd:option('-max_iterations', 75000, 'number of training iteration')
opt = cmd:parse(arg)

model, criterion = createLadderAE{noise_level=0.3, comb_func=opt.comb_func}

if opt.gpuid >= 0 then
  require 'cunn'
  cutorch.setDevice(opt.gpuid+1)
  model:cuda()
  criterion:cuda()
end

local trainer = Trainer.new(model, criterion, opt)

local test_loader = MnistLoader('test', opt.batch_size, -1)
function test()
  test_loader:reset()
  model:evaluate()
  local cfm = optim.ConfusionMatrix(10)
  for t = 1,test_loader.num_batches do
    local x, y = test_loader:next_batch(opt.gpuid)
    local pred = model:forward(x)[1]
    cfm:batchAdd(pred, y)
  end
  print('Test confusion matrix:')
  print(cfm)
  return cfm.totalValid
end

for i = 1,opt.max_iterations do
  trainer:train()
end

