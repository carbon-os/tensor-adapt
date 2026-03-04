#include <tensor/trainer/adamw.hpp>
#include <tensor/backend/cuda/ops.hpp>

namespace tensor::trainer {

using namespace backend::cuda;
namespace ops = backend::cuda::ops;

void AdamW::step(
    adapter::LoraPair& lp,
    float lr, int step,
    const Device& dev) const
{
    int nA = lp.rank * lp.in_dim;
    int nB = lp.out_dim * lp.rank;

    ops::adamw_step(
        lp.A_f32.f32(), lp.A_bf16.bf16(),
        lp.mA.f32(), lp.vA.f32(), lp.gA.f32(),
        nA, lr, beta1, beta2, eps, weight_decay, step, dev.stream());

    ops::adamw_step(
        lp.B_f32.f32(), lp.B_bf16.bf16(),
        lp.mB.f32(), lp.vB.f32(), lp.gB.f32(),
        nB, lr, beta1, beta2, eps, weight_decay, step, dev.stream());
}

void AdamW::step_all(
    adapter::AdapterModel& model,
    float lr, int step,
    const Device& dev) const
{
    for (auto& la : model.layers) {
        for (auto* lp : {&la.lora_q, &la.lora_k, &la.lora_v, &la.lora_o}) {
            this->step(*lp, lr, step, dev);
        }
    }
}

} // namespace tensor::trainer