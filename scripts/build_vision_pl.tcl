# Run from an isolated working directory to keep Vivado logs away from audio:
# vivado -mode batch -source D:/FPGA/ZynqAudioLab/scripts/build_vision_pl.tcl
# Append -tclargs --prepare-only to generate/validate, or --reuse to rebuild.
# VISION_PL_BUILD_DIR / VISION_PL_EXPORT_DIR select isolated comparison paths.
# This overlay consumes DDR initialized by Linux. Never execute ps7_init.
# References: Xilinx/PYNQ v2.7.0 pynq/overlay.py and pl_server/hwh_parser.py;
# Xilinx/embeddedsw drivers/axidma/examples/xaxidma_example_simple_poll.c.
set root [file dirname [file dirname [file normalize [info script]]]]
set build_dir "$root/diagnostics/vision_pl_build"
set export_dir "$root/output/vision_pl"
if {[info exists ::env(VISION_PL_BUILD_DIR)]} {
    set build_dir [file normalize $::env(VISION_PL_BUILD_DIR)]
}
if {[info exists ::env(VISION_PL_EXPORT_DIR)]} {
    set export_dir [file normalize $::env(VISION_PL_EXPORT_DIR)]
}
set prepare_only 0
set reuse 0
set export_only 0
foreach argument $argv {
    switch -exact -- $argument {
        --prepare-only { set prepare_only 1 }
        --reuse { set reuse 1 }
        --export-only { set reuse 1; set export_only 1 }
        default { error "Unknown argument: $argument (use --prepare-only and/or --reuse)" }
    }
}
foreach source {vision_rgb_gray.v vision_sobel3x3.v vision_dma_stream.v} {
    if {![file isfile "$root/rtl/$source"]} {
        error "Vision RTL dependency missing: $root/rtl/$source"
    }
}
file mkdir $build_dir
file mkdir $export_dir
cd $build_dir
set_param general.maxThreads 4

if {$reuse} {
    if {![file isfile "$build_dir/vision_pl.xpr"]} {
        error "No isolated project to reuse; run without --reuse first"
    }
    open_project "$build_dir/vision_pl.xpr"
    if {[get_property PART [current_project]] ne "xc7z020clg400-1"} {
        error "Unexpected part in reused vision project"
    }
    open_bd_design [get_files vision_pl.bd]
    update_compile_order -fileset sources_1
    update_module_reference [get_bd_cells vision_stream]
} else {
    create_project vision_pl $build_dir -part xc7z020clg400-1 -force
    set_property target_language Verilog [current_project]
    set_property simulator_language Mixed [current_project]
    foreach source {vision_rgb_gray.v vision_sobel3x3.v vision_dma_stream.v} {
        add_files -norecurse "$root/rtl/$source"
    }
    create_bd_design vision_pl
    set ps [create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 ps7]
    # DDR must be enabled in the address model for HP0_DDR_LOWOCM. The hard
    # PS DDR controller is already running: no DDR ports or PS init are exported.
    set_property -dict [list \
        CONFIG.PCW_CRYSTAL_PERIPHERAL_FREQMHZ {50} \
        CONFIG.PCW_UIPARAM_DDR_ENABLE {1} \
        CONFIG.PCW_DDR_RAM_HIGHADDR {0x1FFFFFFF} \
        CONFIG.PCW_USE_M_AXI_GP0 {1} \
        CONFIG.PCW_USE_S_AXI_HP0 {1} CONFIG.PCW_S_AXI_HP0_DATA_WIDTH {64} \
        CONFIG.PCW_EN_CLK0_PORT {1} CONFIG.PCW_EN_RST0_PORT {1} \
        CONFIG.PCW_EN_CLK1_PORT {0} CONFIG.PCW_EN_CLK2_PORT {0} \
        CONFIG.PCW_EN_CLK3_PORT {0} \
        CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ {100} \
        CONFIG.PCW_FCLK0_PERIPHERAL_CLKSRC {IO PLL}] $ps
    make_bd_intf_pins_external [get_bd_intf_pins ps7/FIXED_IO]

    set dma [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 vision_dma]
    set_property -dict [list \
        CONFIG.c_include_sg {0} CONFIG.c_sg_length_width {23} \
        CONFIG.c_include_mm2s {1} CONFIG.c_include_s2mm {1} \
        CONFIG.c_addr_width {32} \
        CONFIG.c_m_axi_mm2s_data_width {32} CONFIG.c_m_axi_s2mm_data_width {32} \
        CONFIG.c_m_axis_mm2s_tdata_width {32} CONFIG.c_s_axis_s2mm_tdata_width {32} \
        CONFIG.c_mm2s_burst_size {16} CONFIG.c_s2mm_burst_size {16} \
        CONFIG.c_include_mm2s_dre {0} CONFIG.c_include_s2mm_dre {0} \
        CONFIG.c_sg_include_stscntrl_strm {0}] $dma
    set stream [create_bd_cell -type module -reference vision_dma_stream vision_stream]
    set_property -dict [list CONFIG.FRAME_WIDTH {640} CONFIG.FRAME_HEIGHT {480}] $stream
    set threshold [create_bd_cell -type ip -vlnv xilinx.com:ip:xlconstant:1.1 edge_threshold]
    set_property -dict [list CONFIG.CONST_WIDTH {8} CONFIG.CONST_VAL {64}] $threshold
    connect_bd_net [get_bd_pins edge_threshold/dout] [get_bd_pins vision_stream/threshold]

    set control [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 control_axi]
    set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {1}] $control
    set memory [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 memory_axi]
    # AXI Interconnect converts the DMA AXI4 masters to the PS HP0 AXI3 slave.
    set_property -dict [list CONFIG.NUM_SI {2} CONFIG.NUM_MI {1}] $memory
    connect_bd_intf_net [get_bd_intf_pins ps7/M_AXI_GP0] [get_bd_intf_pins control_axi/S00_AXI]
    connect_bd_intf_net [get_bd_intf_pins control_axi/M00_AXI] [get_bd_intf_pins vision_dma/S_AXI_LITE]
    connect_bd_intf_net [get_bd_intf_pins vision_dma/M_AXI_MM2S] [get_bd_intf_pins memory_axi/S00_AXI]
    connect_bd_intf_net [get_bd_intf_pins vision_dma/M_AXI_S2MM] [get_bd_intf_pins memory_axi/S01_AXI]
    connect_bd_intf_net [get_bd_intf_pins memory_axi/M00_AXI] [get_bd_intf_pins ps7/S_AXI_HP0]
    connect_bd_intf_net [get_bd_intf_pins vision_dma/M_AXIS_MM2S] [get_bd_intf_pins vision_stream/S_AXIS]
    connect_bd_intf_net [get_bd_intf_pins vision_stream/M_AXIS] [get_bd_intf_pins vision_dma/S_AXIS_S2MM]

    set reset [create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 reset_control]
    set high [create_bd_cell -type ip -vlnv xilinx.com:ip:xlconstant:1.1 logic_high]
    set_property CONFIG.CONST_VAL {1} $high
    set low [create_bd_cell -type ip -vlnv xilinx.com:ip:xlconstant:1.1 logic_low]
    set_property CONFIG.CONST_VAL {0} $low
    connect_bd_net [get_bd_pins ps7/FCLK_RESET0_N] [get_bd_pins reset_control/ext_reset_in]
    connect_bd_net [get_bd_pins logic_high/dout] \
        [get_bd_pins reset_control/dcm_locked] [get_bd_pins reset_control/aux_reset_in]
    connect_bd_net [get_bd_pins logic_low/dout] [get_bd_pins reset_control/mb_debug_sys_rst]
    connect_bd_net [get_bd_pins ps7/FCLK_CLK0] \
        [get_bd_pins ps7/M_AXI_GP0_ACLK] [get_bd_pins ps7/S_AXI_HP0_ACLK] \
        [get_bd_pins vision_dma/s_axi_lite_aclk] \
        [get_bd_pins vision_dma/m_axi_mm2s_aclk] [get_bd_pins vision_dma/m_axi_s2mm_aclk] \
        [get_bd_pins vision_stream/aclk] [get_bd_pins reset_control/slowest_sync_clk] \
        [get_bd_pins control_axi/ACLK] [get_bd_pins control_axi/S00_ACLK] \
        [get_bd_pins control_axi/M00_ACLK] [get_bd_pins memory_axi/ACLK] \
        [get_bd_pins memory_axi/S00_ACLK] [get_bd_pins memory_axi/S01_ACLK] \
        [get_bd_pins memory_axi/M00_ACLK]
    connect_bd_net [get_bd_pins reset_control/interconnect_aresetn] \
        [get_bd_pins control_axi/ARESETN] [get_bd_pins memory_axi/ARESETN]
    connect_bd_net [get_bd_pins reset_control/peripheral_aresetn] \
        [get_bd_pins vision_dma/axi_resetn] [get_bd_pins vision_stream/aresetn] \
        [get_bd_pins control_axi/S00_ARESETN] [get_bd_pins control_axi/M00_ARESETN] \
        [get_bd_pins memory_axi/S00_ARESETN] [get_bd_pins memory_axi/S01_ARESETN] \
        [get_bd_pins memory_axi/M00_ARESETN]

    assign_bd_address -offset 0x40400000 -range 0x00010000 \
        -target_address_space [get_bd_addr_spaces ps7/Data] \
        [get_bd_addr_segs vision_dma/S_AXI_LITE/Reg] -force
    foreach channel {Data_MM2S Data_S2MM} {
        assign_bd_address -offset 0x00000000 -range 0x20000000 \
            -target_address_space [get_bd_addr_spaces vision_dma/$channel] \
            [get_bd_addr_segs ps7/S_AXI_HP0/HP0_DDR_LOWOCM] -force
    }
}

validate_bd_design
save_bd_design
set ps [get_bd_cells ps7]
if {[get_property CONFIG.PCW_EN_DDR $ps] != 1 ||
    [get_property CONFIG.PCW_S_AXI_HP0_DATA_WIDTH $ps] != 64} {
    error "PS7 must retain DDR address decoding and 64-bit HP0"
}
if {abs([get_property CONFIG.PCW_ACT_FPGA0_PERIPHERAL_FREQMHZ $ps] - 100.0) > 0.01} {
    error "PS7 FCLK0 is not 100 MHz"
}
if {[llength [get_bd_intf_ports -quiet -filter {VLNV =~ *:ddrx:*}]] != 0 ||
    [llength [get_bd_intf_ports -quiet]] != 1} {
    error "Vision overlay must expose only FIXED_IO, with no PL or DDR pins"
}
set bd_file [get_files vision_pl.bd]
generate_target all $bd_file
if {!$reuse} {
    add_files -norecurse [make_wrapper -files $bd_file -top]
}
set_property top vision_pl_wrapper [get_filesets sources_1]
update_compile_order -fileset sources_1
write_bd_tcl -force "$build_dir/vision_pl_bd.tcl"
puts "VISION_PL_CONFIG: FCLK0=100MHz(IO_PLL) HP0=64 DMA_memory=32 DMA_stream=32 DMA_length=23 DMA_burst=16 DRE=0 SG=0"
puts "VISION_PL_FRAME: 640x480 bytes=1228800 B,G,R,0 -> gray,magnitude,edge255,0 threshold=64"
puts "VISION_PL_LINUX_DDR: existing PS DDR initialization preserved; never run ps7_init"
if {$prepare_only} {
    puts "VISION_PL_PREPARED: $build_dir/vision_pl.xpr"
    close_project
    exit 0
}

if {$reuse && !$export_only} {
    reset_run impl_1
    foreach run [get_runs -filter {IS_SYNTHESIS == 1}] {
        if {[get_property NAME $run] ne "synth_1"} { reset_run $run }
    }
    reset_run synth_1
}
if {!$export_only} {
    launch_runs synth_1 -jobs 4
    wait_on_run synth_1
}
if {[get_property STATUS [get_runs synth_1]] ne "synth_design Complete!"} {
    error "Vision synthesis failed: [get_property STATUS [get_runs synth_1]]"
}
if {!$export_only} {
    launch_runs impl_1 -to_step route_design -jobs 4
    wait_on_run impl_1
}
if {![string match "route_design Complete*" [get_property STATUS [get_runs impl_1]]]} {
    error "Vision implementation failed: [get_property STATUS [get_runs impl_1]]"
}
open_run impl_1
set_property BITSTREAM.CONFIG.USERID 32'h56495331 [current_design]
report_timing_summary -delay_type min_max -report_unconstrained -check_timing_verbose \
    -file "$export_dir/vision_pl_timing.rpt"
report_utilization -file "$export_dir/vision_pl_utilization.rpt"
report_drc -file "$export_dir/vision_pl_drc.rpt"
report_route_status -file "$export_dir/vision_pl_route.rpt"
report_methodology -file "$export_dir/vision_pl_methodology.rpt"
set setup_paths [get_timing_paths -delay_type max -max_paths 1]
set hold_paths [get_timing_paths -delay_type min -max_paths 1]
if {[llength $setup_paths] == 0 || [llength $hold_paths] == 0} {
    error "Vision timing is missing constrained setup/hold paths"
}
set setup_slack [get_property SLACK [lindex $setup_paths 0]]
set hold_slack [get_property SLACK [lindex $hold_paths 0]]
if {$setup_slack < 0 || $hold_slack < 0} {
    error "Vision timing failed: setup=$setup_slack ns hold=$hold_slack ns"
}
set drc_failures [get_drc_violations -quiet -filter {SEVERITY == Error || SEVERITY == {Critical Warning}}]
if {[llength $drc_failures] != 0} { error "Vision DRC failed: $drc_failures" }
if {[get_msg_config -count -severity {CRITICAL WARNING}] != 0 ||
    [get_msg_config -count -severity ERROR] != 0} {
    error "Vivado reported critical warnings/errors; inspect the isolated build log"
}
write_checkpoint -force "$export_dir/vision_pl_routed.dcp"
write_bitstream -force "$export_dir/vision_pl.bit"
set hwh "$build_dir/vision_pl.gen/sources_1/bd/vision_pl/hw_handoff/vision_pl.hwh"
if {![file isfile $hwh]} { error "Missing generated vision hardware handoff: $hwh" }
file copy -force $hwh "$export_dir/vision_pl.hwh"
puts "VISION_PL_BUILD_PASS: setup=$setup_slack ns hold=$hold_slack ns USERID=0x56495331"
puts "VISION_PL_OUTPUT: $export_dir/vision_pl.bit $export_dir/vision_pl.hwh"
close_project
