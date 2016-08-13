local MnistLoader, parent = torch.class('MnistLoader')

DATA_PATH = 'mnist.t7/'

function MnistLoader:__init(split, batch_size, num_labels)
  local num_labels = num_labels or -1
  assert(split == 'train' or split == 'test', 'split must be train or test')
  assert(num_labels <= 0 or num_labels % 10 == 0, 'num labels must be divisible by 10')

  self.batch_size = batch_size
  self.dataset = torch.load(('%s/%s_32x32.t7'):format(DATA_PATH, split), 'ascii')
  self.dataset.data = self.dataset.data[{{},{},{3,30},{3,30}}]:float()
  self.dataset.data:div(255)

  if num_labels > 0 then
    local selection = torch.LongTensor(num_labels)
    for i = 1,10 do
      local idx = self.dataset.labels:eq(i):nonzero():squeeze():totable()
      selection[{{(i-1)*num_labels/10+1,i*num_labels/10}}] = 
        torch.Tensor(idx):index(1, torch.randperm(#idx)[{{1,num_labels/10}}]:long())
    end
    self.dataset.data = self.dataset.data:index(1, selection)
    self.dataset.labels = self.dataset.labels:index(1, selection)
  end

  local randperm = torch.randperm(self.dataset.data:size(1)):long()
  self.dataset.data = self.dataset.data:index(1, randperm):squeeze():float()
  self.dataset.labels = self.dataset.labels:index(1, randperm)
  self.num_batches = math.floor(self.dataset.data:size(1) / self.batch_size)

  self.batch_idx = 0
  self.num_class = 10
  self.num_labels = self.dataset.data:size(1)
  return self
end

function MnistLoader:reset()
  local randperm = torch.randperm(self.dataset.data:size(1)):long()
  self.dataset.data = self.dataset.data:index(1, randperm):squeeze()
  self.dataset.labels = self.dataset.labels:index(1, randperm)
  self.batch_idx = 0
end

function MnistLoader:next_batch(gpuid)
  if self.batch_idx == self.num_batches then
    self:reset()
  end
  self.batch_idx = self.batch_idx % self.num_batches + 1
  local idx1 = (self.batch_idx - 1) * self.batch_size + 1
  local idx2 = self.batch_idx * self.batch_size 
  local x = self.dataset.data:sub(idx1, idx2)
  local y = self.dataset.labels:sub(idx1, idx2)
  if gpuid >= 0 then
    x = x:cuda()
    y = y:cuda()
  end
  return x, y
end

return MnistLoader
