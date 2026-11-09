# Prepare an isolated vision+audio design; Linux owns PS/DDR initialization.
set root [file dirname [file dirname [file normalize [info script]]]]
set build "$root/diagnostics/av_system_build"
set out "$root/output/av_system"
file mkdir $out
if {![file isfile "$build/vision_pl.xpr"]} {
    set ::env(VISION_FLOW_BUILD_DIR) $build
    set ::env(VISION_FLOW_EXPORT_DIR) $out
    set ::env(VISION_FLOW_LK_IP_DIR) "$root/hls/lk_pyramid/build_cached/solution1/impl/ip"
    puts [exec [info nameofexecutable] -mode batch -source "$root/scripts/build_vision_flow.tcl" -tclargs --prepare-only --without-roi-projection]
}
cd $build
set_param general.maxThreads 1
open_project "$build/vision_pl.xpr"
open_bd_design [get_files vision_pl.bd]
if {[llength [get_bd_cells -quiet music]]} {
    puts "AV_SYSTEM_ALREADY_PREPARED: use implement_av_system.tcl to build"
    close_project
    return
}
add_files -norecurse [glob "$root/rtl/*.v"]
add_files -norecurse "$root/rtl/hann4096.mem"
add_files -fileset constrs_1 -norecurse "$root/constraints/pynq_z2_audio_ps_clock.xdc"
create_ip -name xfft -vendor xilinx.com -library ip -version 9.1 -module_name audio_fft_4096
set_property -dict [list CONFIG.channels {1} CONFIG.transform_length {4096} \
    CONFIG.run_time_configurable_transform_length {false} CONFIG.data_format {fixed_point} \
    CONFIG.input_width {16} CONFIG.phase_factor_width {16} CONFIG.rounding_modes {convergent_rounding} \
    CONFIG.output_ordering {natural_order} CONFIG.throttle_scheme {nonrealtime} \
    CONFIG.xk_index {false} CONFIG.aresetn {true} CONFIG.aclken {false} \
    CONFIG.cyclic_prefix_insertion {false} CONFIG.super_sample_rates {1} \
    CONFIG.memory_options_data {block_ram} CONFIG.memory_options_phase_factors {block_ram} \
    CONFIG.memory_options_reorder {block_ram} CONFIG.complex_mult_type {use_mults_resources} \
    CONFIG.butterfly_type {use_luts} CONFIG.target_clock_frequency {125} \
    CONFIG.implementation_options {radix_2_burst_io} CONFIG.scaling_options {unscaled} \
    CONFIG.ovflo {false}] [get_ips audio_fft_4096]
generate_target all [get_ips audio_fft_4096]
set_property -dict [list CONFIG.PCW_EN_CLK1_PORT {1} CONFIG.PCW_EN_RST1_PORT {1} \
    CONFIG.PCW_FPGA1_PERIPHERAL_FREQMHZ {125} CONFIG.PCW_FCLK1_PERIPHERAL_CLKSRC {IO PLL}] [get_bd_cells ps7]
set core [create_bd_cell -type module -reference audio_music_pl music]
set reset [create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 audio_reset_control]
set_property CONFIG.C_AUX_RESET_HIGH {0} $reset
connect_bd_net [get_bd_pins ps7/FCLK_RESET1_N] [get_bd_pins $reset/ext_reset_in]
connect_bd_net [get_bd_pins logic_high/dout] [get_bd_pins $reset/dcm_locked]
connect_bd_net [get_bd_pins logic_high/dout] [get_bd_pins $reset/aux_reset_in]
connect_bd_net [get_bd_pins logic_low/dout] [get_bd_pins $reset/mb_debug_sys_rst]
set bus [get_bd_cells control_axi]
set index [get_property CONFIG.NUM_MI $bus]
set_property CONFIG.NUM_MI [expr {$index+1}] $bus
connect_bd_intf_net [get_bd_intf_pins [format "control_axi/M%02d_AXI" $index]] [get_bd_intf_pins music/S_AXI]
connect_bd_net [get_bd_pins ps7/FCLK_CLK1] [get_bd_pins music/clk] \
    [get_bd_pins $reset/slowest_sync_clk] [get_bd_pins [format "control_axi/M%02d_ACLK" $index]]
connect_bd_net [get_bd_pins $reset/peripheral_aresetn] [get_bd_pins music/rstn] \
    [get_bd_pins [format "control_axi/M%02d_ARESETN" $index]]
foreach name {rst_btn source_sw gain_sw codec_mclk audio_bclk audio_lrclk audio_sdata_in audio_sdata_out audio_i2c_scl audio_i2c_sda codec_addr led} {
    make_bd_pins_external [get_bd_pins music/$name]
    set_property name $name [get_bd_ports ${name}_0]
}
assign_bd_address -offset 0x43C20000 -range 0x00004000 \
    -target_address_space [get_bd_addr_spaces ps7/Data] [get_bd_addr_segs music/S_AXI/reg0] -force
validate_bd_design
save_bd_design
generate_target all [get_files vision_pl.bd]
add_files -norecurse [make_wrapper -files [get_files vision_pl.bd] -top]
set_property top vision_pl_wrapper [get_filesets sources_1]
update_compile_order -fileset sources_1
add_files -fileset utils_1 -norecurse "$root/scripts/vision_single_thread.tcl"
foreach run [get_runs -filter {IS_SYNTHESIS == 1}] {
    set_property STEPS.SYNTH_DESIGN.TCL.PRE "$root/scripts/vision_single_thread.tcl" $run
}
write_bd_tcl -force "$out/av_system_bd.tcl"
puts "AV_SYSTEM_PREPARED: vision100MHz audio125MHz DMA40400000 LK43c00000 AUDIO43c20000"
close_project
