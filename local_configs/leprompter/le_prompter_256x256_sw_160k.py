from local_configs._base_.models.le_prompter import prompts_steps

_base_ = [
    '../_base_/models/le_prompter.py', '../_base_/datasets/prompt_sw_256x256.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]

model = dict(pretrained=None, decode_head=dict(num_classes=2))

# optimizer
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

samples_per_gpu = 16
workers_per_gpu = 4
# train_step1_step会比sam_leformer中的prompts_steps先降到0，加上35可以让train_step1_step后降到0
train_step1_steps = prompts_steps * workers_per_gpu + 35
data = dict(samples_per_gpu=samples_per_gpu, workers_per_gpu=workers_per_gpu)

