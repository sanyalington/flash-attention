import torch
from ck_layer_norm import oldmain, ln_fwd

if __name__ == '__main__':
    start_ev = torch.cuda.Event(enable_timing=True)
    stop_ev = torch.cuda.Event(enable_timing=True)
    epsilon = 1e-5
    device = torch.device("cuda")
    fp32 = torch.float32
    fp16 = torch.float16
    #bf16 = torch.bfloat16
    BS=1
    hidden_size=4096
    torch.manual_seed(1)

    x0 = torch.randn((1001,BS, hidden_size), dtype=fp16, device=device)
    beta0 = torch.randn((1001,hidden_size), dtype=fp16, device=device)
    gamma0 = torch.randn((1001,hidden_size), dtype=fp16, device=device)
    
    x = torch.randn((BS, hidden_size), dtype=fp16, device=device)
    beta = torch.randn(hidden_size, dtype=fp16, device=device)
    gamma = torch.randn(hidden_size, dtype=fp16, device=device)

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        z0 = ln_fwd(x0[100], gamma0[100], beta0[100], epsilon)
        z = ln_fwd(x, gamma, beta, epsilon)
        stream.record_event(start_ev)
        for i in range(1000):
            z1 = ln_fwd(x0[i], gamma0[i], beta0[i], epsilon)
        stream.record_event(stop_ev)
    stream.synchronize()

    mu_ref = x.mean(1, dtype=fp32, keepdim=True)
    v = torch.square(x - mu_ref).mean(1, dtype=fp32, keepdim=True)
    rs_ref = torch.rsqrt(v + epsilon)
    y_ref = rs_ref * (x.to(fp32) - mu_ref)
    z_ref = (gamma.unsqueeze(0) * (y_ref).to(fp32) + beta.unsqueeze(0)).to(fp32)

    z.mean()

    print(start_ev.elapsed_time(stop_ev))
    print(z)
    print(z_ref)
