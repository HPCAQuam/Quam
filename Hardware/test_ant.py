import argparse
from accelerator import Accelerator 

model_list = ["facebook/opt-1.3b", "microsoft/phi-2", "01-ai/Yi-6B", "meta-llama/Llama-2-7b-hf", "meta-llama/Meta-Llama-3-8B"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--is_generation", type=bool, default=True, help="If enabled, then evaluate")
    args = parser.parse_args()
    is_generation = args.is_generation

    w_prec_list = {
        'facebook/opt-1.3b': 5, 
        'microsoft/phi-2': 5, 
        '01-ai/Yi-6B': 5.25, 
        'meta-llama/Llama-2-7b-hf': 5, 
        'meta-llama/Meta-Llama-3-8B': 4.5, 
    }

    batch_size = 64
    if(batch_size>=16):
        pe_array_dim = [16, 74]
    else:
        pe_array_dim = [batch_size, 74*16/batch_size]
    
    total_energy_list = [[0, 0] for _ in model_list]
    total_latency_list = [0 for _ in model_list]

    for idx, model_name in enumerate(model_list):
        if is_generation:
            w_prec = w_prec_list[model_name]
        else:
            w_prec = 5.5

        acc = Accelerator(
            model_name=model_name, 
            i_prec=16,
            w_prec=w_prec,
            is_bit_serial=False,
            pe_dp_size=batch_size,
            pe_energy=0.0235,
            pe_area=2328.564,
            pe_array_dim=pe_array_dim,
            context_length=256,
            is_generation=is_generation,
        )

        total_cycle    = acc.calc_cycle()
        compute_energy = acc.calc_compute_energy() / 1e6
        sram_rd_energy = acc.calc_sram_rd_energy() / 1e6
        sram_wr_energy = acc.calc_sram_wr_energy() / 1e6
        dram_energy    = acc.calc_dram_energy() / 1e6
        onchip_energy  = compute_energy + sram_rd_energy + sram_wr_energy
        total_energy   = compute_energy + sram_rd_energy + sram_wr_energy + dram_energy

        print(f'model: {model_name}')
        print(f'total cycle:        {total_cycle}')
        total_latency_list[idx] = total_cycle[1]

        print(f'pe array area:      {acc.pe_array_area / 1e6} mm2')
        print(f'weight buffer area: {acc.w_sram.area} mm2')
        print(f'input buffer area:  {acc.i_sram.area} mm2')
        # print(f'compute energy:     {compute_energy} uJ')
        # print(f'sram rd energy:     {sram_rd_energy} uJ')
        # print(f'sram wr energy:     {sram_wr_energy} uJ')
        print(f'dram energy:        {dram_energy} uJ')
        print(f'on-chip energy:     {onchip_energy} uJ')
        print(f'total energy:       {total_energy} uJ')
        total_energy_list[idx][0] = round(onchip_energy)
        total_energy_list[idx][1] = round(total_energy)
        print('\n')

    print(f'Latency: {total_latency_list}')
    print(f'Energy: {total_energy_list}')
    