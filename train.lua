local Trainer = torch.class('Trainer')

function Trainer:__init(model, criterion, opt)
  self.iter = 0
  self.params, self.grad_params = model:getParameters()
  self.opt = opt
  self.optim_state = {learningRate=opt.learning_rate}
  self.init_learning_rate = opt.learning_rate
  self.lr_decay_iter = opt.lr_decay_iter or -1
  self.max_iter = opt.max_iterations

  self.model = model
  self.criterion = criterion


  self.loader = MnistLoader('train', opt.batch_size, opt.num_labels, opt.norm_input)
  self.unlabeled_loader = MnistLoader('train', opt.batch_size, -1, opt.norm_input)
  self.targets = {}

  self.loss_sum, self.dn_loss_sum = 0, 0
end

local confusion = optim.ConfusionMatrix(10)
function Trainer:train()
  self.iter = self.iter + 1
  local i = self.iter
  local opt = self.opt
  local model, criterion = self.model, self.criterion

  local targets = self.targets

  local function feval()
    return self.criterion.output, self.grad_params
  end

  local loss, dn_loss = nil, nil
  self.grad_params:zero()

  -------------------------------------------------------------
  -- Unlabled training
  local x_unlabeled = self.unlabeled_loader:next_batch(opt.gpuid)

  -- Compute clean activations (without noise)
  model:evaluate()
  for i = 1,#model.layer_sizes do
    model.bn_layers[i]:training()
  end

  local output = model:forward(x_unlabeled)

  -- dummy targets
  targets[1] = torch.zeros(opt.batch_size):typeAs(x_unlabeled)
  for i = 2,#output do
    targets[i] = targets[i] or output[i].new():resizeAs(output[i])
    targets[i]:copy(output[i])
  end

  for i = 0,#model.layer_sizes do
    targets[#model.layer_sizes+3+i]:copy(output[2+i])
    if i >= 1 then
      self.criterion.criterions[i+#model.layer_sizes+3]
        :setMeanStd(model.bn_layers[i].save_mean,
                    model.bn_layers[i].save_std)
    end
  end

  model:training()

  -- set criterion weights
  for i = 1,#criterion.weights do criterion.weights[i] = 0 end
  criterion.weights[#model.layer_sizes+3] = 1000 
  criterion.weights[#model.layer_sizes+4] = 10 
  for i = 2,#model.layer_sizes do
    criterion.weights[#model.layer_sizes+3+i] = 0.1
  end

  local output_noisy = model:forward(x_unlabeled)
  dn_loss = criterion:forward(output_noisy, targets)
  local d = criterion:backward(output_noisy, targets)
  model:backward(x_unlabeled, d)

  -------------------------------------------------------------
  -- Labeled training
  model:training()
  local x_labeled, y = self.loader:next_batch(opt.gpuid)
  local out = model:forward(x_labeled)

  for i = 1,#criterion.weights do criterion.weights[i] = 0 end
  criterion.weights[1] = 1
  targets[1]:copy(y)

  loss = criterion:forward(out, targets)
  local d = criterion:backward(out, targets)
  model:backward(x_labeled, d)
  confusion:batchAdd(out[1], y)

  optim.adam(feval, self.params, self.optim_state)

  if i % 100 == 0 then
    self.loss_sum = self.loss_sum + loss
    local loss_mean = self.loss_sum / 100
    self.loss_sum = 0

    self.dn_loss_sum = self.dn_loss_sum + dn_loss
    dn_loss_mean = self.dn_loss_sum / 100
    self.dn_loss_sum = 0

    print(('iteration %d: cls_loss = %f, denoise_loss = %f')
            :format(i, loss_mean, dn_loss_mean))
  end
  if i % 1000 == 0 then
    confusion:updateValids()
    print('Train accuracy:', confusion.totalValid)
    local test_accuracy = test()
  end
  if i * opt.batch_size % opt.num_labels == 0 then
    confusion:zero()
  end

  if self.lr_decay_iter > 0 and i > self.lr_decay_iter then
    local decay_rate = 1 - (i-self.lr_decay_iter) / (self.max_iter-self.lr_decay_iter)
    self.optim_state.learningRate =
      math.max(1e-8, self.init_learning_rate * decay_rate)
  end
end

return Trainer
