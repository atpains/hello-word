-- usage example: DATA_ROOT=/path/to/data/ which_direction=BtoA name=expt1 th train.lua 
--
-- code derived from https://github.com/soumith/dcgan.torch
--

require 'torch'
require 'nn'
require 'optim'
util = paths.dofile('util/util.lua')
require 'image'
require 'cutorch'
require 'cunn'

require 'fast_neural_style.DataLoader'
require 'fast_neural_style.PerceptualCriterion'
require 'fast_neural_style.TotalVariation'


local utils = require 'fast_neural_style.utils'
local preprocess = require 'fast_neural_style.preprocess'


opt = {
    DATA_ROOT = '/data/artflow/git/pix2pix.percep_loss/datasets/s1/AB', -- path to images (should have subfolders 'train', 'val', etc)
    batchSize = 1, -- # images in batch
    loadSize = 286, -- scale images to this size
    fineSize = 256, --  then crop to this size
    ngf = 64, -- #  of gen filters in first conv layer
    ndf = 64, -- #  of discrim filters in first conv layer
    input_nc = 3, -- #  of input image channels
    output_nc = 3, -- #  of output image channels
    niter = 200, -- #  of iter at starting learning rate
    lr = 0.0002, -- initial learning rate for adam
    beta1 = 0.5, -- momentum term of adam
    ntrain = math.huge, -- #  of examples per epoch. math.huge for full dataset
    flip = 0, -- if flip the images for data argumentation
    display = 1, -- display samples while training. 0 = false
    display_id = 10, -- display window id.
    display_plot = 'errL1', -- which loss values to plot over time. Accepted values include a comma seperated list of: errL1, errG, and errD
    gpu = 1, -- gpu = 0 is CPU mode. gpu=X is GPU mode on GPU X
    name = 'pix2pix_percep_loss', -- name of the experiment, should generally be passed on the command line
    which_direction = 'AtoB', -- AtoB or BtoA
    phase = 'train', -- train, val, test, etc
    preprocess = 'regular', -- for special purpose preprocessing, e.g., for colorization, change this (selects preprocessing functions in util.lua)
    nThreads = 2, -- # threads for loading data
    save_epoch_freq = 1, -- save a model every save_epoch_freq epochs (does not overwrite previously saved models)
    save_latest_freq = 5000, -- save the latest model every latest_freq sgd iterations (overwrites the previous latest model)
    print_freq = 50, -- print the debug information every print_freq iterations
    display_freq = 100, -- display the current results every display_freq iterations
    save_display_freq = 5000, -- save the current display of results every save_display_freq_iterations
    continue_train = 0, -- if continue training, load the latest model: 1: true, 0: false
    serial_batches = 0, -- if 1, takes images in order to make batches, otherwise takes them randomly
    serial_batch_iter = 1, -- iter into serial image list
    checkpoints_dir = './checkpoints', -- models are saved here
    cudnn = 1, -- set to 0 to not use cudnn
    condition_GAN = 1, -- set to 0 to use unconditional discriminator
    use_GAN = 1, -- set to 0 to turn off GAN term
    use_L1 = 1, -- set to 0 to turn off L1 term
    which_model_netD = 'basic', -- selects model to use for netD
    which_model_netG = 'unet', -- selects model to use for netG. 1. unet, 2. unet_noise, 3. unet_noise_resize, 4. unet_resize, 5. unet_resize_5
    n_layers_D = 0, -- only used if which_model_netD=='n_layers'
    lambda = 5, -- weight on L1 term in objective

    -- percep loss parameters
    percep_loss_weight = 1.0,
    content_weights = '5.0',
    content_layers = '9',
    style_image = 's1.jpg',
    style_image_size = 512,
    style_weights = '4000.0',
    style_layers = '9,14,21,25',
    style_target_type = 'gram',
    task = 'style',
    tanh_constant = 150,
    preprocessing = 'vgg',
    loss_network = '/data/artflow/models/vgg19.t7',
    mult = 0.1,
    init_epoch_num = 2,
    use_real_A = 0,
    sr_type = 'post', -- pre or post
    pooling_type = 'max', -- avg or max
    down_kernal = 3,
    down_padding = 1,
    sr_grad_upsample = 1,
    tv = 0,
    upsample_lambda = 10,
    shift = 0
}

-- one-line argument parser. parses enviroment variables to override the defaults
for k, v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)

local lambda_tmp = opt.lambda
local content_weights_tmp = tonumber(opt.content_weights)
local style_weights_tmp = tonumber(opt.style_weights)

local input_nc = opt.input_nc
local output_nc = opt.output_nc
-- translation direction
local idx_A = nil
local idx_B = nil

if opt.which_direction == 'AtoB' then
    idx_A = { 1, input_nc }
    idx_B = { input_nc + 1, input_nc + output_nc }
elseif opt.which_direction == 'BtoA' then
    idx_A = { input_nc + 1, input_nc + output_nc }
    idx_B = { 1, input_nc }
else
    error(string.format('bad direction %s', opt.which_direction))
end

if opt.display == 0 then opt.display = false end

opt.manualSeed = torch.random(1, 10000) -- fix seed
print("Random Seed: " .. opt.manualSeed)
torch.manualSeed(opt.manualSeed)
torch.setdefaulttensortype('torch.FloatTensor')

-- create data loader
local data_loader = paths.dofile('data/data.lua')
print('#threads...' .. opt.nThreads)
local data = data_loader.new(opt.nThreads, opt)
print("Dataset Size: ", data:size())
tmp_d, tmp_paths = data:getBatch()

----------------- Initialization of percep loss net--------------------------
-- Parse layer strings and weights
opt.style_weights = tostring(opt.style_weights)
opt.content_weights = tostring(opt.content_weights)
opt.content_layers = tostring(opt.content_layers)
opt.content_layers, opt.content_weights = utils.parse_layers(opt.content_layers, opt.content_weights)
opt.style_layers, opt.style_weights = utils.parse_layers(opt.style_layers, opt.style_weights)

-- Figure out preprocessing
if not preprocess[opt.preprocessing] then
    local msg = 'invalid -preprocessing "%s"; must be "vgg" or "resnet"'
    error(string.format(msg, opt.preprocessing))
end
preprocess = preprocess[opt.preprocessing]

-- Figure out the backend
cutorch.setDevice(opt.gpu)
local dtype = 'torch.CudaTensor'
local use_cudnn = true

-- Set up the perceptual loss function
local percep_crit
if opt.percep_loss_weight > 0 then
    local loss_net = torch.load(opt.loss_network)
    local crit_args = {
        cnn = loss_net,
        style_layers = opt.style_layers,
        style_weights = opt.style_weights,
        content_layers = opt.content_layers,
        content_weights = opt.content_weights,
        agg_type = opt.style_target_type,
	shift = tonumber(opt.shift),
    }
    percep_crit = nn.PerceptualCriterion(crit_args):type(dtype)

    if opt.task == 'style' then
        -- Load the style image and set it
        local style_image = image.load(opt.style_image, 3, 'float')
        style_image = image.scale(style_image, opt.style_image_size)
        local H, W = style_image:size(2), style_image:size(3)
        local perm = torch.LongTensor { 3, 2, 1 }
        style_image = style_image:index(1, perm)
        style_image = style_image:mul(2):add(-1)
        --style_image = preprocess.preprocess(style_image:view(1, 3, H, W))
        percep_crit:setStyleTarget(style_image:type(dtype))
    end
end




------------------ Initialization of netG and netD---------------------------
require 'models_noise_sr'
local function weights_init(m)
    local name = torch.type(m)
    if name:find('Convolution') then
        m.weight:normal(0.0, 0.02)
        m.bias:fill(0)
    elseif name:find('BatchNormalization') then
        if m.weight then m.weight:normal(1.0, 0.02) end
        if m.bias then m.bias:fill(0) end
    end
end


local ndf = opt.ndf
local ngf = opt.ngf
local real_label = 1
local fake_label = 0

function defineG(input_nc, output_nc, ngf)
    local netG = nil
    if opt.which_model_netG == "encoder_decoder" then netG = defineG_encoder_decoder(input_nc, output_nc, ngf)
    elseif opt.which_model_netG == "unet" then netG = defineG_unet(input_nc, output_nc, ngf)
    elseif opt.which_model_netG == "unet_noise" then netG = defineG_unet_noise(input_nc, output_nc, ngf, opt.mult)
    elseif opt.which_model_netG == "unet_noise_resize" then netG = defineG_unet_noise_resize(input_nc, output_nc, ngf, opt.mult)
    elseif opt.which_model_netG == "unet_resize" then netG = defineG_unet_resize(input_nc, output_nc, ngf)
    elseif opt.which_model_netG == "unet_resize_5" then netG = defineG_unet_resize_5(input_nc, output_nc, ngf)
    elseif opt.which_model_netG == "unet_128" then netG = defineG_unet_128(input_nc, output_nc, ngf)
    elseif opt.which_model_netG == "unet_resize_sr" then netG = defineG_unet_resize_super_resolution(input_nc, output_nc, ngf)
    elseif opt.which_model_netG == "unet_resize_sr_x4" then netG = defineG_unet_resize_super_resolution_x4(input_nc, output_nc, ngf)
    elseif opt.which_model_netG == "unet_noise_resize_sr" then netG = defineG_unet_noise_resize_super_resolution(input_nc, output_nc, ngf, opt.mult)
    elseif opt.which_model_netG == "unet_noise_resize_sr_x4" then netG = defineG_unet_noise_resize_super_resolution_x4(input_nc, output_nc, ngf, opt.mult)
    else error("unsupported netG model")
    end

    netG:apply(weights_init)

    return netG
end

function defineD(input_nc, output_nc, ndf)
    local netD = nil
    if opt.condition_GAN == 1 then
        input_nc_tmp = input_nc
    else
        input_nc_tmp = 0 -- only penalizes structure in output channels
    end

    if opt.which_model_netD == "basic" then netD = defineD_basic(input_nc_tmp, output_nc, ndf)
    elseif opt.which_model_netD == "n_layers" then netD = defineD_n_layers(input_nc_tmp, output_nc, ndf, opt.n_layers_D)
    else error("unsupported netD model")
    end

    netD:apply(weights_init)

    return netD
end


-- load saved models and finetune
if opt.continue_train == 1 then
    print('loading previously trained netG...')
    netG = util.load(paths.concat(opt.checkpoints_dir, opt.name, 'latest_net_G.t7'), opt)
    print('loading previously trained netD...')
    netD = util.load(paths.concat(opt.checkpoints_dir, opt.name, 'latest_net_D.t7'), opt)
else
    print('define model netG...')
    netG = defineG(input_nc, output_nc, ngf)
    print('define model netD...')
    netD = defineD(input_nc, output_nc, ndf)
end

print(netG)
print(netD)


local criterion = nn.BCECriterion()
local criterionAE = nn.AbsCriterion()
local criterionUpsampleMSE = nn.AbsCriterion()
---------------------------------------------------------------------------
optimStateG = {
    learningRate = opt.lr,
    beta1 = opt.beta1,
}
optimStateD = {
    learningRate = opt.lr,
    beta1 = opt.beta1,
}
----------------------------------------------------------------------------
local real_A = torch.Tensor(opt.batchSize, input_nc, opt.fineSize, opt.fineSize)
local real_A_downsample = torch.Tensor(opt.batchSize, input_nc, math.floor(opt.fineSize / 2), math.floor(opt.fineSize / 2))
local real_B = torch.Tensor(opt.batchSize, output_nc, opt.fineSize, opt.fineSize)
local fake_B_upsample = torch.Tensor(opt.batchSize, input_nc, math.floor(opt.fineSize * 2), math.floor(opt.fineSize * 2))
local fake_B = torch.Tensor(opt.batchSize, output_nc, opt.fineSize, opt.fineSize)
local real_AB = torch.Tensor(opt.batchSize, output_nc + input_nc * opt.condition_GAN, opt.fineSize, opt.fineSize)
local fake_AB = torch.Tensor(opt.batchSize, output_nc + input_nc * opt.condition_GAN, opt.fineSize, opt.fineSize)
local errD, errG, errL1 = 0, 0, 0
local epoch_tm = torch.Timer()
local tm = torch.Timer()
local data_tm = torch.Timer()
----------------------------------------------------------------------------

if opt.gpu > 0 then
    print('transferring to gpu...')
    --require 'cunn'
    --cutorch.setDevice(opt.gpu)
    real_A = real_A:cuda();
    real_A_downsample = real_A_downsample:cuda();
    fake_B_upsample = fake_B_upsample:cuda();
    real_B = real_B:cuda(); fake_B = fake_B:cuda();
    real_AB = real_AB:cuda(); fake_AB = fake_AB:cuda();
    if opt.cudnn == 1 then
        netG = util.cudnn(netG); netD = util.cudnn(netD);
    end
    netD:cuda(); netG:cuda(); criterion:cuda(); criterionAE:cuda();criterionUpsampleMSE:cuda();
    print('done')
else
    print('running model on CPU')
end


local parametersD, gradParametersD = netD:getParameters()
local parametersG, gradParametersG = netG:getParameters()



if opt.display then disp = require 'display' end

local downsampling = nn.Sequential()
if opt.which_model_netG == "unet_resize_sr_x4" or opt.which_model_netG == "unet_noise_resize_sr_x4" then
    if opt.pooling_type == "avg" then
        downsampling = downsampling:add(nn.SpatialReflectionPadding(0, opt.down_padding, 0, opt.down_padding))
        downsampling = downsampling:add(nn.SpatialAveragePooling(opt.down_kernal, opt.down_kernal, 2, 2, 0, 0))
        downsampling = downsampling:add(nn.SpatialReflectionPadding(0, opt.down_padding, 0, opt.down_padding))
        downsampling = downsampling:add(nn.SpatialAveragePooling(opt.down_kernal, opt.down_kernal, 2, 2, 0, 0))
    else
        downsampling = downsampling:add(nn.SpatialReflectionPadding(0, opt.down_padding, 0, opt.down_padding))
        downsampling = downsampling:add(nn.SpatialMaxPooling(opt.down_kernal, opt.down_kernal, 2, 2, 0, 0))
        downsampling = downsampling:add(nn.SpatialReflectionPadding(0, opt.down_padding, 0, opt.down_padding))
        downsampling = downsampling:add(nn.SpatialMaxPooling(opt.down_kernal, opt.down_kernal, 2, 2, 0, 0))
    end
else
    if opt.pooling_type == "avg" then
        downsampling = downsampling:add(nn.SpatialReflectionPadding(0, opt.down_padding, 0, opt.down_padding))
        downsampling = downsampling:add(nn.SpatialAveragePooling(opt.down_kernal, opt.down_kernal, 2, 2, 0, 0))
    else
        downsampling = downsampling:add(nn.SpatialReflectionPadding(0, opt.down_padding, 0, opt.down_padding))
        downsampling = downsampling:add(nn.SpatialMaxPooling(opt.down_kernal, opt.down_kernal, 2, 2, 0, 0))
    end
end
print('Downsampling:')
print(downsampling)
downsampling = util.cudnn(downsampling);
downsampling = downsampling:cuda();


local upsampling = nn.SpatialUpSamplingBilinear(2)
print('Upsampling:')
print(upsampling)
upsampling = util.cudnn(upsampling);
upsampling = upsampling:cuda();

local total_variation = nn.TotalVariation(opt.tv)
print('Total Variation:')
print(total_variation)
total_variation = util.cudnn(total_variation);
total_variation = total_variation:cuda();

local zeros = torch.Tensor(fake_B_upsample:size()):zero():cuda()

function createRealFake()
    -- load real
    data_tm:reset(); data_tm:resume()
    local real_data, data_path = data:getBatch()
    data_tm:stop()

    real_A:copy(real_data[{ {}, idx_A, {}, {} }])
    real_B:copy(real_data[{ {}, idx_B, {}, {} }])

    if opt.condition_GAN == 1 then
        real_AB = torch.cat(real_A, real_B, 2)
    else
        real_AB = real_B -- unconditional GAN, only penalizes structure in B
    end

    -- create fake
    if opt.which_model_netG == "unet_resize_sr" or opt.which_model_netG == "unet_resize_sr_x4" or opt.which_model_netG == "unet_noise_resize_sr" or opt.which_model_netG == "unet_noise_resize_sr_x4" then
        if opt.sr_type == "pre" then
            real_A_downsample = downsampling:forward(real_A)
            fake_B = netG:forward(real_A_downsample)
        else
            fake_B_upsample = netG:forward(real_A)
            fake_B = downsampling:forward(fake_B_upsample)
        end
    else
        fake_B = netG:forward(real_A)
    end

    if opt.condition_GAN == 1 then
        fake_AB = torch.cat(real_A, fake_B, 2)
    else
        fake_AB = fake_B -- unconditional GAN, only penalizes structure in B
    end
end

-- create closure to evaluate f(X) and df/dX of discriminator
local fDx = function(x)
    netD:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
    netG:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)

    gradParametersD:zero()

    -- Real
    local output = netD:forward(real_AB)
    local label = torch.FloatTensor(output:size()):fill(real_label)
    if opt.gpu > 0 then
        label = label:cuda()
    end

    local errD_real = criterion:forward(output, label)
    local df_do = criterion:backward(output, label)
    netD:backward(real_AB, df_do)

    -- Fake
    local output = netD:forward(fake_AB)
    label:fill(fake_label)
    local errD_fake = criterion:forward(output, label)
    local df_do = criterion:backward(output, label)
    netD:backward(fake_AB, df_do)

    errD = (errD_real + errD_fake) / 2

    return errD, gradParametersD
end

-- create closure to evaluate f(X) and df/dX of generator
local fGx = function(x)
    netD:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
    netG:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)

    gradParametersG:zero()

    -- GAN loss
    local df_dg = torch.zeros(fake_B:size())
    if opt.gpu > 0 then
        df_dg = df_dg:cuda();
    end

    if opt.use_GAN == 1 then
        local output = netD.output -- netD:forward{input_A,input_B} was already executed in fDx, so save computation
        local label = torch.FloatTensor(output:size()):fill(real_label) -- fake labels are real for generator cost
        if opt.gpu > 0 then
            label = label:cuda();
        end
        errG = criterion:forward(output, label)
        local df_do = criterion:backward(output, label)
        df_dg = netD:updateGradInput(fake_AB, df_do):narrow(2, fake_AB:size(2) - output_nc + 1, output_nc)
    else
        errG = 0
    end

    -- unary loss
    local df_do_AE = torch.zeros(fake_B:size())
    if opt.gpu > 0 then
        df_do_AE = df_do_AE:cuda();
    end
    if opt.use_L1 == 1 then
        errL1 = criterionAE:forward(fake_B, real_B)
        df_do_AE = criterionAE:backward(fake_B, real_B)
    else
        errL1 = 0
    end

    local real_B_upsample = upsampling:forward(real_B)
    local loss_UpsampleMSE = criterionUpsampleMSE:forward(fake_B_upsample, real_B_upsample)
    local grad_UpsampleMSE = criterionUpsampleMSE:backward(fake_B_upsample, real_B_upsample)
    upsampling:clearState()
    
    -- Total Variation
    local tv_grad = total_variation:backward(fake_B_upsample, zeros)

    -- Compute perceptual loss and gradient
    local grad_out = nil
    local percep_loss = 0
    if percep_crit then
        percep_crit:setStyleWeight(opt.style_weights)
        percep_crit:setContentWeight(opt.content_weights)
        --percep_crit:setStyleTarget(real_B)
        local target = { content_target = real_B } -- real_A or real_B
        if use_real_A == 1 then
            target = { content_target = real_A }
        end
        percep_loss = percep_crit:forward(fake_B, target)
        percep_loss = percep_loss * opt.percep_loss_weight
        local grad_out_percep = percep_crit:backward(fake_B, target)
        if grad_out then
            grad_out:add(opt.percep_loss_weight, grad_out_percep)
        else
            grad_out_percep:mul(opt.percep_loss_weight)
            grad_out = grad_out_percep
        end
    end

    errL1 = errL1 * opt.lambda + percep_loss + loss_UpsampleMSE * opt.upsample_lambda
    --errG = errG + percep_loss

    if opt.which_model_netG == "unet_resize_sr" or opt.which_model_netG == "unet_resize_sr_x4" or opt.which_model_netG == "unet_noise_resize_sr" or opt.which_model_netG == "unet_noise_resize_sr_x4" then
        if opt.sr_type == "pre" then
            netG:backward(real_A_downsample, df_dg + df_do_AE:mul(opt.lambda) + grad_out)
        else
            -- local grad_to_netG = downsampling:backward(fake_B_upsample, df_dg + df_do_AE:mul(opt.lambda) + grad_out)
            local grad_to_netG = nil
            if opt.sr_grad_upsample == 0 then
                grad_to_netG = downsampling:backward(fake_B_upsample, df_dg + df_do_AE:mul(opt.lambda) + grad_out)
            elseif opt.sr_grad_upsample == 1 then
                grad_to_netG = upsampling:forward(df_dg + df_do_AE:mul(opt.lambda) + grad_out)
            else 
                local target_fake_B = fake_B - df_dg - df_do_AE:mul(opt.lambda) - grad_out
                local target_fake_B_upsample = upsampling:forward(target_fake_B)
                grad_to_netG = fake_B_upsample - target_fake_B_upsample
            end
            netG:backward(real_A, grad_to_netG + tv_grad + grad_UpsampleMSE:mul(opt.upsample_lambda))
        end
    else
        netG:backward(real_A, df_dg + df_do_AE:mul(opt.lambda) + grad_out)
    end

    return errG, gradParametersG
end




-- train
local best_err = nil
paths.mkdir(opt.checkpoints_dir)
paths.mkdir(opt.checkpoints_dir .. '/' .. opt.name)

-- save opt
file = torch.DiskFile(paths.concat(opt.checkpoints_dir, opt.name, 'opt.txt'), 'w')
file:writeObject(opt)
file:close()

-- parse diplay_plot string into table
opt.display_plot = string.split(string.gsub(opt.display_plot, "%s+", ""), ",")
for k, v in ipairs(opt.display_plot) do
    if not util.containsValue({ "errG", "errD", "errL1" }, v) then
        error(string.format('bad display_plot value "%s"', v))
    end
end

-- display plot config
local plot_config = {
    title = "Loss over time",
    labels = { "epoch", unpack(opt.display_plot) },
    ylabel = "loss",
}

-- display plot vars
local plot_data = {}
local plot_win

opt.lambda = 100
opt.content_weights = 100.0
opt.style_weights = 0.0

local counter = 0
for epoch = 1, opt.niter do
    if epoch == opt.init_epoch_num then
        opt.lambda = lambda_tmp
        opt.content_weights = content_weights_tmp
        opt.style_weights = style_weights_tmp

        print("------------------------------------------------------------------")
        print(string.format('lambda=%d, opt.content_weights=%f,opt.style_weights=%f', opt.lambda, opt.content_weights, opt.style_weights))


        optimStateG = {
            learningRate = opt.lr,
            beta1 = opt.beta1,
        }
        optimStateD = {
            learningRate = opt.lr,
            beta1 = opt.beta1,
        }
    end

    epoch_tm:reset()
    for i = 1, math.min(data:size(), opt.ntrain), opt.batchSize do

        tm:reset()

        -- load a batch and run G on that batch
        createRealFake()

        -- (1) Update D network: maximize log(D(x,y)) + log(1 - D(x,G(x)))
        if opt.use_GAN == 1 then optim.adam(fDx, parametersD, optimStateD) end

        -- (2) Update G network: maximize log(D(x,G(x))) + L1(y,G(x))
        optim.adam(fGx, parametersG, optimStateG)

        -- display
        counter = counter + 1
        if counter % opt.display_freq == 0 and opt.display then
            createRealFake()
            if opt.preprocess == 'colorization' then
                local real_A_s = util.scaleBatch(real_A:float(), 100, 100)
                local fake_B_s = util.scaleBatch(fake_B:float(), 100, 100)
                local real_B_s = util.scaleBatch(real_B:float(), 100, 100)
                disp.image(util.deprocessL_batch(real_A_s), { win = opt.display_id, title = opt.name .. ' input' })
                disp.image(util.deprocessLAB_batch(real_A_s, fake_B_s), { win = opt.display_id + 1, title = opt.name .. ' output' })
                disp.image(util.deprocessLAB_batch(real_A_s, real_B_s), { win = opt.display_id + 2, title = opt.name .. ' target' })
            else
                disp.image(util.deprocess_batch(util.scaleBatch(real_A:float(), 100, 100)), { win = opt.display_id, title = opt.name .. ' input' })
                disp.image(util.deprocess_batch(util.scaleBatch(fake_B:float(), 100, 100)), { win = opt.display_id + 1, title = opt.name .. ' output' })
                disp.image(util.deprocess_batch(util.scaleBatch(real_B:float(), 100, 100)), { win = opt.display_id + 2, title = opt.name .. ' target' })
            end
        end

        -- write display visualization to disk
        --  runs on the first batchSize images in the opt.phase set
        if counter % opt.save_display_freq == 0 and opt.display then
            local serial_batches = opt.serial_batches
            opt.serial_batches = 1
            opt.serial_batch_iter = 1

            local image_out = nil
            local N_save_display = 10
            local N_save_iter = torch.max(torch.Tensor({ 1, torch.floor(N_save_display / opt.batchSize) }))
            for i3 = 1, N_save_iter do

                createRealFake()
                print('save to the disk')
                if opt.preprocess == 'colorization' then
                    for i2 = 1, fake_B:size(1) do
                        if image_out == nil then image_out = torch.cat(util.deprocessL(real_A[i2]:float()), util.deprocessLAB(real_A[i2]:float(), fake_B[i2]:float()), 3) / 255.0
                        else image_out = torch.cat(image_out, torch.cat(util.deprocessL(real_A[i2]:float()), util.deprocessLAB(real_A[i2]:float(), fake_B[i2]:float()), 3) / 255.0, 2)
                        end
                    end
                else
                    for i2 = 1, fake_B:size(1) do
                        if image_out == nil then image_out = torch.cat(util.deprocess(real_A[i2]:float()), util.deprocess(fake_B[i2]:float()), 3)
                        else image_out = torch.cat(image_out, torch.cat(util.deprocess(real_A[i2]:float()), util.deprocess(fake_B[i2]:float()), 3), 2)
                        end
                    end
                end
            end
            image.save(paths.concat(opt.checkpoints_dir, opt.name, counter .. '_train_res.png'), image_out)

            opt.serial_batches = serial_batches
        end

        -- logging and display plot
        if counter % opt.print_freq == 0 then
            local loss = { errG = errG and errG or -1, errD = errD and errD or -1, errL1 = errL1 and errL1 or -1 }
            local curItInBatch = ((i - 1) / opt.batchSize)
            local totalItInBatch = math.floor(math.min(data:size(), opt.ntrain) / opt.batchSize)
            print(('Epoch: [%d][%8d / %8d]\t Time: %.3f  DataTime: %.3f  '
                    .. '  Err_G: %.4f  Err_D: %.4f  ErrL1: %.4f'):format(epoch, curItInBatch, totalItInBatch,
                tm:time().real / opt.batchSize, data_tm:time().real / opt.batchSize,
                errG, errD, errL1))

            local plot_vals = { epoch + curItInBatch / totalItInBatch }
            for k, v in ipairs(opt.display_plot) do
                if loss[v] ~= nil then
                    plot_vals[#plot_vals + 1] = loss[v]
                end
            end

            -- update display plot
            if opt.display then
                table.insert(plot_data, plot_vals)
                plot_config.win = plot_win
                plot_win = disp.plot(plot_data, plot_config)
            end
        end

        -- save latest model
        if counter % opt.save_latest_freq == 0 then
            print(('saving the latest model (epoch %d, iters %d)'):format(epoch, counter))
            torch.save(paths.concat(opt.checkpoints_dir, opt.name, 'latest_net_G.t7'), netG:clearState())
            torch.save(paths.concat(opt.checkpoints_dir, opt.name, 'latest_net_D.t7'), netD:clearState())

            torch.save(paths.concat(opt.checkpoints_dir, opt.name, 'Iter_' .. counter .. '_net_G.t7'), netG:clearState())
            torch.save(paths.concat(opt.checkpoints_dir, opt.name, 'Iter_' .. counter .. '_net_D.t7'), netD:clearState())
        end
    end


    parametersD, gradParametersD = nil, nil -- nil them to avoid spiking memory
    parametersG, gradParametersG = nil, nil

    if epoch % opt.save_epoch_freq == 0 then
        torch.save(paths.concat(opt.checkpoints_dir, opt.name, epoch .. '_net_G.t7'), netG:clearState())
        torch.save(paths.concat(opt.checkpoints_dir, opt.name, epoch .. '_net_D.t7'), netD:clearState())
    end

    print(('End of epoch %d / %d \t Time Taken: %.3f'):format(epoch, opt.niter, epoch_tm:time().real))
    parametersD, gradParametersD = netD:getParameters() -- reflatten the params and get them
    parametersG, gradParametersG = netG:getParameters()
end
