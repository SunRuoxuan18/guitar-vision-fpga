set root "D:/FPGA/ZynqAudioLab"
set build_dir "$root/diagnostics/ps_clock_test"
if {[llength $argv] > 0 && [lindex $argv 0] eq "--verify-export"} {
    open_project "$build_dir/audio_ps_clock.xpr"
} else {
create_project audio_ps_clock $build_dir -part xc7z020clg400-1 -force
add_files -norecurse [glob "$root/rtl/*.v"]
add_files -fileset constrs_1 -norecurse "$root/constraints/pynq_z2_audio_ps_clock.xdc"

create_bd_design audio_ps_clock
set ps [create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 ps7]
set_property -dict [list \
    CONFIG.PCW_CRYSTAL_PERIPHERAL_FREQMHZ {50} \
    CONFIG.PCW_UIPARAM_DDR_ENABLE {0} \
    CONFIG.PCW_USE_M_AXI_GP0 {0} \
    CONFIG.PCW_EN_CLK0_PORT {1} \
    CONFIG.PCW_EN_RST0_PORT {0} \
    CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ {125} \
    CONFIG.PCW_FCLK0_PERIPHERAL_CLKSRC {IO PLL}] $ps
make_bd_intf_pins_external [get_bd_intf_pins ps7/FIXED_IO]
set core [create_bd_cell -type module -reference pynq_audio_board_top audio_core]
connect_bd_net [get_bd_pins ps7/FCLK_CLK0] [get_bd_pins audio_core/sys_clk]
foreach name {rst_btn source_sw gain_sw codec_mclk audio_bclk audio_lrclk audio_sdata_in audio_sdata_out audio_i2c_scl audio_i2c_sda codec_addr led} {
    make_bd_pins_external [get_bd_pins audio_core/$name]
    set_property name $name [get_bd_ports ${name}_0]
}
validate_bd_design
save_bd_design
set bd_file [get_files audio_ps_clock.bd]
generate_target all $bd_file
add_files -norecurse [make_wrapper -files $bd_file -top]
set_property top audio_ps_clock_wrapper [get_filesets sources_1]
update_compile_order -fileset sources_1
puts "PS_CLOCK_ACTUAL_MHZ: [get_property CONFIG.PCW_ACT_FPGA0_PERIPHERAL_FREQMHZ $ps]"

launch_runs synth_1 -jobs 4
wait_on_run synth_1
if {[get_property STATUS [get_runs synth_1]] ne "synth_design Complete!"} {
    error "PS-clock synthesis failed."
}
launch_runs impl_1 -to_step write_bitstream -jobs 4
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] ne "100%"} {
    error "PS-clock bitstream failed."
}
}
open_run impl_1
foreach pad_name {audio_i2c_scl audio_i2c_sda} {
    set pad_net [get_nets -segments -of_objects [get_ports $pad_name]]
    set pad_cells [get_cells -of_objects [get_pins -leaf -of_objects $pad_net]]
    set iobufs [filter $pad_cells {REF_NAME == IOBUF}]
    set output_buffers [filter $pad_cells {REF_NAME == OBUFT}]
    set input_buffers [filter $pad_cells {REF_NAME == IBUF}]
    if {[llength $iobufs] == 1} {
        set output_buffer [lindex $iobufs 0]
        set input_buffer $output_buffer
    } elseif {[llength $output_buffers] == 1 &&
              ($pad_name eq "audio_i2c_scl" || [llength $input_buffers] == 1)} {
        set output_buffer [lindex $output_buffers 0]
        set input_buffer [lindex $input_buffers 0]
    } else {
        error "Open-drain I2C pad $pad_name has no bidirectional pad buffer."
    }
    set enable_net [get_nets -segments -of_objects [get_pins $output_buffer/T]]
    set enable_drivers [get_cells -of_objects [get_pins -leaf -of_objects $enable_net -filter {DIRECTION == OUT}]]
    if {[llength $enable_drivers] == 0 ||
        [llength [filter $enable_drivers {REF_NAME == GND || REF_NAME == VCC}]] != 0} {
        error "Open-drain I2C pad $pad_name has a constant output enable."
    }
    set data_net [get_nets -segments -of_objects [get_pins $output_buffer/I]]
    set data_drivers [get_cells -of_objects [get_pins -leaf -of_objects $data_net -filter {DIRECTION == OUT}]]
    if {[llength [filter $data_drivers {REF_NAME == GND}]] != 1} {
        error "Open-drain I2C pad $pad_name does not drive a constant zero."
    }
    if {$pad_name eq "audio_i2c_sda"} {
        set feedback_net [get_nets -segments -of_objects [get_pins $input_buffer/O]]
        if {[llength [get_pins -leaf -of_objects $feedback_net -filter {DIRECTION == IN}]] == 0} {
            error "I2C SDA input feedback is disconnected."
        }
    }
    puts "I2C_PAD_VERIFIED: $pad_name $output_buffer"
}
report_timing_summary -file "$root/output/audio_ps_clock_timing.rpt"
if {[get_property SLACK [lindex [get_timing_paths -delay_type max -max_paths 1] 0]] < 0} {
    error "PS-clock setup timing failed."
}
if {[get_property SLACK [lindex [get_timing_paths -delay_type min -max_paths 1] 0]] < 0} {
    error "PS-clock hold timing failed."
}
file copy -force "$build_dir/audio_ps_clock.runs/impl_1/audio_ps_clock_wrapper.bit" "$root/output/audio_ps_clock_wrapper.bit"
set init_files [glob -nocomplain "$build_dir/audio_ps_clock.gen/sources_1/bd/audio_ps_clock/ip/audio_ps_clock_ps7_0/ps7_init.tcl"]
if {[llength $init_files] != 1} {
    error "Cannot locate PS7 initialization script."
}
file copy -force [lindex $init_files 0] "$root/output/audio_ps_clock_init.tcl"
puts "PS_CLOCK_BUILD_SUCCESS"
close_project
