## 
simulate_pixels.py \
--input_filename=/workspace/larnd_simulation/data/edep.LArBath_converted.h5 \
--detector_properties=/workspace/larnd_simulation/data/larndsim_meta/ndlar-module.yaml \
--pixel_layout=/workspace/larnd_simulation/data/larndsim_meta/multi_tile_layout-3.0.40.yaml \
--simulation_properties=/workspace/larnd_simulation/data/larndsim_meta/singles_sim.yaml \
--output_filename=/workspace/larnd_simulation/data/edep.LarBath.larndsim_converted.h5 \
--response_file=/workspace/larnd_simulation/larnd-sim/larndsim/bin/response_44.npy

#--response_file=/workspace/larnd_simulation/data/larndsim_meta/response_38.npy