# ddpm.py代码解析

latent diffusion.py文件下有四个类，DDPM，LatentDiffusion，DiffusionWrapper，Layout2ImgDiffusion。



```
instantiate_from_config(config)
```

这个函数是通过配置文件索引相应的模型类位置，并填入配置参数创建网络模型

## DDPM类



## LatentDiffusion类

latentdiffusion模块主要由三部分组成，vae部分，条件控制部分，扩散部分组成。

分别起名为` self.first_stage_model`,`self.cond_stage_model`,` self.model`





主线调用流程training_step（只有ddpm有），调用shared_step（ddpm与ldm都有，ldm重构）,

shared_step调用get_input与forward

get_input在ddpm与ldm中都有，ldm继承重构了

forward在ddpm与ldm中都有，ldm重构了



### training_step(存在于ddpm类中)



```
def training_step(self, batch, batch_idx):
    loss, loss_dict = self.shared_step(batch)

    self.log_dict(loss_dict, prog_bar=True,
                  logger=True, on_step=True, on_epoch=True)

    self.log("global_step", self.global_step,
             prog_bar=True, logger=True, on_step=True, on_epoch=False)

    if self.use_scheduler:
        lr = self.optimizers().param_groups[0]['lr']
        self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)

    return loss
```



### shared_step

```
def shared_step(self, batch, **kwargs):
    x, c = self.get_input(batch, self.first_stage_key)
    loss = self(x, c)
    return loss
```





### get_input

```python
def get_input(self, batch, k, return_first_stage_outputs=False, force_c_encode=False,
              cond_key=None, return_original_cond=False, bs=None):
    x = super().get_input(batch, k)
    if bs is not None:
        x = x[:bs]
    x = x.to(self.device)
    encoder_posterior = self.encode_first_stage(x)
    z = self.get_first_stage_encoding(encoder_posterior).detach()

    if self.model.conditioning_key is not None:
        if cond_key is None:
            cond_key = self.cond_stage_key
        if cond_key != self.first_stage_key:
            if cond_key in ['caption', 'coordinates_bbox','txt']:
                xc = batch[cond_key]
            elif cond_key == 'class_label':
                xc = batch
            else:
                xc = super().get_input(batch, cond_key).to(self.device)
        else:
            xc = x
        if not self.cond_stage_trainable or force_c_encode:
            if isinstance(xc, dict) or isinstance(xc, list):
                # import pudb; pudb.set_trace()
                c = self.get_learned_conditioning(xc)
            else:
                c = self.get_learned_conditioning(xc.to(self.device))
        else:
            c = xc
        if bs is not None:
            c = c[:bs]

        if self.use_positional_encodings:
            pos_x, pos_y = self.compute_latent_shifts(batch)
            ckey = __conditioning_keys__[self.model.conditioning_key]
            c = {ckey: c, 'pos_x': pos_x, 'pos_y': pos_y}

    else:
        c = None
        xc = None
        if self.use_positional_encodings:
            pos_x, pos_y = self.compute_latent_shifts(batch)
            c = {'pos_x': pos_x, 'pos_y': pos_y}
    out = [z, c]
    if return_first_stage_outputs:
        xrec = self.decode_first_stage(z)
        out.extend([x, xrec])
    if return_original_cond:
        out.append(xc)
    return out
```

z是图像压缩后编码，条件如果是不可训练的则调用get_learned_conditioning进行编码，如果是可训练的，则输入原本的文本



通过这样写，将可训练的部分都写在了forward里面





### instantiate_first_stage方法

```python
    def instantiate_first_stage(self, config):
        model = instantiate_from_config(config)
        self.first_stage_model = model.eval()
        self.first_stage_model.train = disabled_train
        for param in self.first_stage_model.parameters():
            param.requires_grad = False
```

该方法初始化self.first_stage_model

### instantiate_cond_stage方法



```python
    def instantiate_cond_stage(self, config):
        if not self.cond_stage_trainable:
            if config == "__is_first_stage__":
                print("Using first stage also as cond stage.")
                self.cond_stage_model = self.first_stage_model
            elif config == "__is_unconditional__":
                print(f"Training {self.__class__.__name__} as an unconditional model.")
                self.cond_stage_model = None
                # self.be_unconditional = True
            else:
                model = instantiate_from_config(config)
                self.cond_stage_model = model.eval()
                self.cond_stage_model.train = disabled_train
                for param in self.cond_stage_model.parameters():
                    param.requires_grad = False
        else:
            assert config != '__is_first_stage__'
            assert config != '__is_unconditional__'
            model = instantiate_from_config(config)
            self.cond_stage_model = model
```

该方法初始化self.cond_stage_model，首先看配置文件condation是否是可训练的，如果是不可训练的，分为三种，

不可训练

1. 用first_stage作为条件（图生图？）
2. 无条件
3. 用别人的模型（实际用这个）

可训练

 	1. 自己创建一个可训练条件编码模型。



###	forward方法

```python
def forward(self, x, c, *args, **kwargs):
    t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
    if self.model.conditioning_key is not None:
        assert c is not None
        if self.cond_stage_trainable:
            c = self.get_learned_conditioning(c)
        if self.shorten_cond_schedule:  # TODO: drop this option
            tc = self.cond_ids[t].to(self.device)
            c = self.q_sample(x_start=c, t=tc, noise=torch.randn_like(c.float()))
    return self.p_losses(x, c, t, *args, **kwargs)
```



如果条件是可训练的，那么self.get_learned_conditioning(c)方法进行编码，自己



这样p_losses输入的x,c都是经过编码的，x通过了vae,c通过了get_learned_conditioning



```python
def get_learned_conditioning(self, c):
    if self.cond_stage_forward is None:
        if hasattr(self.cond_stage_model, 'encode') and callable(self.cond_stage_model.encode):
            c = self.cond_stage_model.encode(c)
            if isinstance(c, DiagonalGaussianDistribution):
                c = c.mode()
        else:
            c = self.cond_stage_model(c)
    else:
        assert hasattr(self.cond_stage_model, self.cond_stage_forward)
        c = getattr(self.cond_stage_model, self.cond_stage_forward)(c)
    return c
```





` self.first_stage_model`,`self.cond_stage_model`分为初始化与调用通过网络后输出。

其中instantiate_first_stage，instantiate_cond_stage是初始化



encode_first_stage与get_learned_conditioning是调用网络后输出

调用encode_first_stage后，紧跟着调用get_first_stage_encoding，这两算是一个整体

get_first_stage_encoding是调用encode_first_stage后检查其是不是某种自定义的网络编码，是的话执行某个操作，不是的话，返回原值





