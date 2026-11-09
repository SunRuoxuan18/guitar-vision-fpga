set root [file dirname [file dirname [file normalize [info script]]]]
set build_dir "$root/diagnostics/audio_music_build"
set export_dir "$root/output/audio_music"
set export_arg [lsearch -exact $argv "--export-dir"]
if {$export_arg >= 0} {
    if {$export_arg + 1 >= [llength $argv]} { error "Missing --export-dir path" }
    set export_dir [file normalize [lindex $argv [expr {$export_arg + 1}]]]
}
file mkdir $export_dir
set reuse [expr {[lsearch -exact $argv "--reuse"] >= 0}]
set implement_only [expr {[lsearch -exact $argv "--implement-only"] >= 0}]
if {$reuse} {
    open_project "$build_dir/audio_music.xpr"
    update_compile_order -fileset sources_1
    if {$implement_only} {
        reset_run impl_1
    } else {
        reset_run [get_runs audio_music_music_0_synth_1]
        reset_run synth_1
    }
} else {
create_project audio_music $build_dir -part xc7z020clg400-1 -force
set_property target_language Verilog [current_project]
set_property simulator_language Mixed [current_project]
add_files -norecurse [glob "$root/rtl/*.v"]
add_files -norecurse "$root/rtl/hann4096.mem"
add_files -fileset constrs_1 -norecurse "$root/constraints/pynq_z2_audio_ps_clock.xdc"
create_ip -name xfft -vendor xilinx.com -library ip -version 9.1 -module_name audio_fft_4096
set_property -dict [list \
    CONFIG.channels {1} CONFIG.transform_length {4096} \
    CONFIG.run_time_configurable_transform_length {false} \
    CONFIG.data_format {fixed_point} CONFIG.input_width {16} \
    CONFIG.phase_factor_width {16} CONFIG.rounding_modes {convergent_rounding} \
    CONFIG.output_ordering {natural_order} CONFIG.throttle_scheme {nonrealtime} \
    CONFIG.xk_index {false} CONFIG.aresetn {true} CONFIG.aclken {false} \
    CONFIG.cyclic_prefix_insertion {false} CONFIG.super_sample_rates {1} \
    CONFIG.memory_options_data {block_ram} CONFIG.memory_options_phase_factors {block_ram} \
    CONFIG.memory_options_reorder {block_ram} \
    CONFIG.complex_mult_type {use_mults_resources} CONFIG.butterfly_type {use_luts} \
    CONFIG.target_clock_frequency {125} \
    CONFIG.implementation_options {radix_2_burst_io} \
    CONFIG.scaling_options {unscaled} CONFIG.ovflo {false}] [get_ips audio_fft_4096]
generate_target all [get_ips audio_fft_4096]

create_bd_design audio_music
set ps [create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 ps7]
set_property -dict [list \
    CONFIG.PCW_CRYSTAL_PERIPHERAL_FREQMHZ {50} \
    CONFIG.PCW_UIPARAM_DDR_ENABLE {0} CONFIG.PCW_USE_M_AXI_GP0 {1} \
    CONFIG.PCW_EN_CLK0_PORT {1} CONFIG.PCW_EN_RST0_PORT {1} \
    CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ {125} \
    CONFIG.PCW_FCLK0_PERIPHERAL_CLKSRC {IO PLL}] $ps
make_bd_intf_pins_external [get_bd_intf_pins ps7/FIXED_IO]
set core [create_bd_cell -type module -reference audio_music_pl music]
set interconnect [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_bus]
set_property CONFIG.NUM_MI {1} $interconnect
set reset [create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 reset_control]
set_property CONFIG.C_EXT_RESET_HIGH {0} $reset
set high [create_bd_cell -type ip -vlnv xilinx.com:ip:xlconstant:1.1 high]
set_property CONFIG.CONST_VAL {1} $high
connect_bd_intf_net [get_bd_intf_pins ps7/M_AXI_GP0] [get_bd_intf_pins axi_bus/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_bus/M00_AXI] [get_bd_intf_pins music/S_AXI]
connect_bd_net [get_bd_pins ps7/FCLK_CLK0] \
    [get_bd_pins ps7/M_AXI_GP0_ACLK] [get_bd_pins music/clk] \
    [get_bd_pins axi_bus/ACLK] [get_bd_pins axi_bus/S00_ACLK] \
    [get_bd_pins axi_bus/M00_ACLK] [get_bd_pins reset_control/slowest_sync_clk]
connect_bd_net [get_bd_pins ps7/FCLK_RESET0_N] [get_bd_pins reset_control/ext_reset_in]
connect_bd_net [get_bd_pins high/dout] [get_bd_pins reset_control/dcm_locked]
connect_bd_net [get_bd_pins reset_control/interconnect_aresetn] [get_bd_pins axi_bus/ARESETN]
connect_bd_net [get_bd_pins reset_control/peripheral_aresetn] \
    [get_bd_pins music/rstn] [get_bd_pins axi_bus/S00_ARESETN] [get_bd_pins axi_bus/M00_ARESETN]
foreach name {rst_btn source_sw gain_sw codec_mclk audio_bclk audio_lrclk audio_sdata_in audio_sdata_out audio_i2c_scl audio_i2c_sda codec_addr led} {
    make_bd_pins_external [get_bd_pins music/$name]
    set_property name $name [get_bd_ports ${name}_0]
}
assign_bd_address -offset 0x43C00000 -range 0x00004000 \
    -target_address_space [get_bd_addr_spaces ps7/Data] [get_bd_addr_segs music/S_AXI/reg0] -force
validate_bd_design
save_bd_design
set bd_file [get_files audio_music.bd]
generate_target all $bd_file
add_files -norecurse [make_wrapper -files $bd_file -top]
set_property top audio_music_wrapper [get_filesets sources_1]
update_compile_order -fileset sources_1
}

if {[llength $argv] > 0 && [lindex $argv 0] eq "--prepare-only"} {
    puts "AUDIO_MUSIC_PREPARED"
    close_project
    exit
}
if {!$implement_only} {
    launch_runs synth_1 -jobs 4
    wait_on_run synth_1
}
if {[get_property STATUS [get_runs synth_1]] ne "synth_design Complete!"} { error "Synthesis failed" }
launch_runs impl_1 -to_step route_design -jobs 4
wait_on_run impl_1
if {![string match "route_design Complete*" [get_property STATUS [get_runs impl_1]]]} { error "Implementation failed" }
open_run impl_1
phys_opt_design -directive AggressiveExplore
set_property BITSTREAM.CONFIG.USERID 32'h41554434 [current_design]
foreach pad {audio_i2c_scl audio_i2c_sda} {
    set nets [get_nets -segments -of_objects [get_ports $pad]]
    set cells [get_cells -of_objects [get_pins -leaf -of_objects $nets]]
    set buffers [filter $cells {REF_NAME == OBUFT || REF_NAME == IOBUF}]
    if {[llength $buffers] != 1} { error "Invalid open-drain buffer: $pad" }
    set enable [get_nets -segments -of_objects [get_pins [lindex $buffers 0]/T]]
    set drivers [get_cells -of_objects [get_pins -leaf -of_objects $enable -filter {DIRECTION == OUT}]]
    if {[llength $drivers] == 0 || [llength [filter $drivers {REF_NAME == GND || REF_NAME == VCC}]]} {
        error "Constant I2C output enable: $pad"
    }
    puts "I2C_PAD_VERIFIED: $pad"
}
report_timing_summary -file "$export_dir/audio_music_timing.rpt"
report_utilization -file "$export_dir/audio_music_utilization.rpt"
report_drc -file "$export_dir/audio_music_drc.rpt"
report_cdc -details -file "$export_dir/audio_music_cdc.rpt"
report_clock_interaction -file "$export_dir/audio_music_clock_interaction.rpt"
if {[get_property SLACK [lindex [get_timing_paths -delay_type max -max_paths 1] 0]] < 0} { error "Setup timing failed" }
if {[get_property SLACK [lindex [get_timing_paths -delay_type min -max_paths 1] 0]] < 0} { error "Hold timing failed" }
set banks [get_cells -hier -filter {NAME =~ *capture_inst/bank*_reg* && REF_NAME =~ RAMB*}]
if {[llength $banks] != 4} { error "Expected four capture RAMB36 primitives" }
write_checkpoint -force "$export_dir/audio_music_routed.dcp"
write_bitstream -force "$export_dir/audio_music.bit"
file copy -force "$build_dir/audio_music.gen/sources_1/bd/audio_music/hw_handoff/audio_music.hwh" "$export_dir/audio_music.hwh"
puts "AUDIO_MUSIC_BUILD_PASS"
close_project
