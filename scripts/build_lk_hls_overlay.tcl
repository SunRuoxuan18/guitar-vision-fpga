# Isolated optical-flow experiment; never initializes the running PS/DDR.
set root [file dirname [file dirname [file normalize [info script]]]]
set build "$root/diagnostics/lk_hls_overlay"
set output "$root/output/lk_hls"
file mkdir $build
file mkdir $output
cd $build
set_param general.maxThreads 4
create_project lk_hls $build -part xc7z020clg400-1 -force
set_property ip_repo_paths [list "$root/hls/lk_sparse/build/solution1/impl/ip"] [current_project]
update_ip_catalog
set cores [get_ipdefs -all -filter {NAME == lk_sparse}]
if {[llength $cores] != 1} { error "Build/export lk_sparse HLS IP before building overlay" }
create_bd_design lk_hls
set ps [create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 ps7]
set_property -dict [list CONFIG.PCW_CRYSTAL_PERIPHERAL_FREQMHZ {50} \
    CONFIG.PCW_UIPARAM_DDR_ENABLE {1} CONFIG.PCW_DDR_RAM_HIGHADDR {0x1FFFFFFF} \
    CONFIG.PCW_USE_M_AXI_GP0 {1} CONFIG.PCW_USE_S_AXI_HP0 {1} \
    CONFIG.PCW_S_AXI_HP0_DATA_WIDTH {64} CONFIG.PCW_EN_CLK0_PORT {1} \
    CONFIG.PCW_EN_RST0_PORT {1} CONFIG.PCW_EN_CLK1_PORT {0} \
    CONFIG.PCW_EN_CLK2_PORT {0} CONFIG.PCW_EN_CLK3_PORT {0} \
    CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ {100} \
    CONFIG.PCW_FCLK0_PERIPHERAL_CLKSRC {IO PLL}] $ps
make_bd_intf_pins_external [get_bd_intf_pins ps7/FIXED_IO]
set core [create_bd_cell -type ip -vlnv [lindex $cores 0] lk_sparse]
set masters [get_bd_intf_pins -of_objects $core -filter {MODE == Master && VLNV =~ *:aximm*:*}]
set controls [get_bd_intf_pins -of_objects $core -filter {MODE == Slave && VLNV =~ *:aximm*:*}]
if {[llength $controls] != 1 || [llength $masters] < 1} { error "Unexpected HLS interface topology" }
set control [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 control_axi]
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {1}] $control
set memory [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 memory_axi]
set_property -dict [list CONFIG.NUM_SI [llength $masters] CONFIG.NUM_MI {1}] $memory
connect_bd_intf_net [get_bd_intf_pins ps7/M_AXI_GP0] [get_bd_intf_pins control_axi/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins control_axi/M00_AXI] $controls
set index 0
foreach master $masters {
    connect_bd_intf_net $master [get_bd_intf_pins [format "memory_axi/S%02d_AXI" $index]]
    incr index
}
connect_bd_intf_net [get_bd_intf_pins memory_axi/M00_AXI] [get_bd_intf_pins ps7/S_AXI_HP0]
create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 reset_control
create_bd_cell -type ip -vlnv xilinx.com:ip:xlconstant:1.1 clock_locked
set_property -dict [list CONFIG.CONST_WIDTH {1} CONFIG.CONST_VAL {1}] [get_bd_cells clock_locked]
connect_bd_net [get_bd_pins clock_locked/dout] [get_bd_pins reset_control/dcm_locked] [get_bd_pins reset_control/aux_reset_in]
create_bd_cell -type ip -vlnv xilinx.com:ip:xlconstant:1.1 reset_inactive
set_property CONFIG.CONST_VAL {0} [get_bd_cells reset_inactive]
connect_bd_net [get_bd_pins reset_inactive/dout] [get_bd_pins reset_control/mb_debug_sys_rst]
connect_bd_net [get_bd_pins ps7/FCLK_RESET0_N] [get_bd_pins reset_control/ext_reset_in]
set clocks [concat [get_bd_pins ps7/M_AXI_GP0_ACLK] [get_bd_pins ps7/S_AXI_HP0_ACLK] \
    [get_bd_pins lk_sparse/ap_clk] [get_bd_pins reset_control/slowest_sync_clk] \
    [get_bd_pins -of_objects $control -filter {TYPE == clk}] \
    [get_bd_pins -of_objects $memory -filter {TYPE == clk}]]
connect_bd_net [get_bd_pins ps7/FCLK_CLK0] {*}$clocks
set resets [concat [get_bd_pins lk_sparse/ap_rst_n] \
    [get_bd_pins -of_objects $control -filter {TYPE == rst}] \
    [get_bd_pins -of_objects $memory -filter {TYPE == rst}]]
connect_bd_net [get_bd_pins reset_control/peripheral_aresetn] {*}$resets
assign_bd_address
validate_bd_design
save_bd_design
set bd [get_files lk_hls.bd]
generate_target all $bd
add_files -norecurse [make_wrapper -files $bd -top]
set_property top lk_hls_wrapper [get_filesets sources_1]
update_compile_order -fileset sources_1
write_bd_tcl -force "$build/lk_hls_bd.tcl"
if {[lsearch -exact $argv --prepare-only] >= 0} { close_project; exit 0 }
launch_runs synth_1 -jobs 4
wait_on_run synth_1
if {[get_property STATUS [get_runs synth_1]] ne "synth_design Complete!"} { error "Synthesis failed" }
launch_runs impl_1 -to_step route_design -jobs 4
wait_on_run impl_1
if {![string match "route_design Complete*" [get_property STATUS [get_runs impl_1]]]} { error "Routing failed" }
open_run impl_1
report_timing_summary -file "$output/timing.rpt"
report_utilization -file "$output/utilization.rpt"
report_drc -file "$output/drc.rpt"
foreach type {min max} {
    set paths [get_timing_paths -delay_type $type -max_paths 1]
    if {[llength $paths] == 0 || [get_property SLACK [lindex $paths 0]] < 0} { error "Timing failed: $type" }
}
if {[llength [get_drc_violations -quiet -filter {SEVERITY == Error}]] > 0} { error "DRC errors" }
set_property BITSTREAM.CONFIG.USERID 32'h4C4B4831 [current_design]
write_bitstream -force "$output/lk_hls.bit"
file copy -force "$build/lk_hls.gen/sources_1/bd/lk_hls/hw_handoff/lk_hls.hwh" "$output/lk_hls.hwh"
puts "LK_HLS_OVERLAY_PASS: $output"
close_project
