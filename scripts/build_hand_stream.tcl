# Isolated candidate only. The routed dw_wide project is never modified.
set root [file dirname [file dirname [file normalize [info script]]]]
set build "$root/diagnostics/hand_stream_overlay_build"
set out "$root/output/hand_stream"
set repository "$root/hls/hand_dw_pw_stream/build/solution1/impl/ip"
if {![file isfile "$repository/component.xml"]} { error "Stream IP not exported" }
if {[file exists "$build/vision_pl.xpr"]} { error "Use existing isolated stream project; refusing overwrite" }
file mkdir $out
open_project "$root/diagnostics/hand_neural_dw_wide_build/vision_pl.xpr"
save_project_as -include_local_ip_cache vision_pl $build
close_project
open_project "$build/vision_pl.xpr"
cd $build
set_param general.maxThreads 2
set repos [get_property ip_repo_paths [current_project]]
lappend repos $repository
set_property ip_repo_paths $repos [current_project]
update_ip_catalog
open_bd_design [get_files vision_pl.bd]
if {[string first $build [file normalize [get_files vision_pl.bd]]] != 0} { error "BD is not isolated" }
if {[info exists ::env(VIVADO_IPCFG_TCL)]} {
    source -notrace $::env(VIVADO_IPCFG_TCL)
}
if {![info exists ::hand_stream_native_log]} { error "Missing native session log path" }
set ::ipcfg::native_log_path $::hand_stream_native_log
set python_status [ipcfg::python_status]
puts $python_status
if {![string match {PYTHON:OK *} $python_status]} { error $python_status }
puts [ipcfg::python_exec -c {import argparse, base64, hashlib, json, pathlib, re, sys, typing; print("PYTHON_IMPORTS:OK")}]
set definitions [get_ipdefs -all -filter {NAME == hand_dw_pw_stream}]
if {[llength $definitions] != 1} { error "Stream IP definition must be unique" }
puts [ipcfg::create_cell [lindex $definitions 0] hand_dw_pw_stream]
set cell [get_bd_cells hand_dw_pw_stream]
set masters [get_bd_intf_pins -of_objects $cell -filter {MODE == Master}]
if {[llength $masters] != 3} { error "Expected operand, pointweights and results AXI masters" }
set ci [get_property CONFIG.NUM_MI [get_bd_cells control_axi]]
set mi [get_property CONFIG.NUM_SI [get_bd_cells memory_axi]]
set cn [expr {$ci+1}]
set mn [expr {$mi+3}]
foreach {name property count reason} [list control_axi CONFIG.NUM_MI $cn {Stream control master} memory_axi CONFIG.NUM_SI $mn {Stream memory masters}] {
    set receipt [ipcfg::capture "$out/$name.[clock clicks].tcldict" \
        [list ipcfg::require $name [dict create $property $count] $reason] {REQ:pinned+applied}]
    puts $receipt
    if {[dict get $receipt status] ne "OK"} { error "Stream interconnect configuration failed" }
    if {[get_property $property [get_bd_cells $name]] != $count} { error "Interconnect readback failed" }
}
set clk [get_bd_pins ps7/FCLK_CLK0]
set reset [get_bd_pins reset_control/peripheral_aresetn]
connect_bd_intf_net [get_bd_intf_pins [format "control_axi/M%02d_AXI" $ci]] [get_bd_intf_pins hand_dw_pw_stream/s_axi_control]
foreach pin [get_bd_pins -of_objects $cell -filter {TYPE == clk}] { connect_bd_net $clk $pin }
foreach pin [get_bd_pins -of_objects $cell -filter {TYPE == rst}] { connect_bd_net $reset $pin }
connect_bd_net $clk [get_bd_pins [format "control_axi/M%02d_ACLK" $ci]]
connect_bd_net $reset [get_bd_pins [format "control_axi/M%02d_ARESETN" $ci]]
assign_bd_address -offset 0x43C50000 -range 0x00010000 -target_address_space [get_bd_addr_spaces ps7/Data] \
    [get_bd_addr_segs hand_dw_pw_stream/s_axi_control/Reg] -force
foreach master $masters {
    set name [get_property NAME $master]
    connect_bd_intf_net $master [get_bd_intf_pins [format "memory_axi/S%02d_AXI" $mi]]
    connect_bd_net $clk [get_bd_pins [format "memory_axi/S%02d_ACLK" $mi]]
    connect_bd_net $reset [get_bd_pins [format "memory_axi/S%02d_ARESETN" $mi]]
    assign_bd_address -offset 0 -range 0x20000000 -target_address_space [get_bd_addr_spaces "hand_dw_pw_stream/Data_$name"] \
        [get_bd_addr_segs ps7/S_AXI_HP0/HP0_DDR_LOWOCM] -force
    incr mi
}
validate_bd_design
save_bd_design
generate_target all [get_files vision_pl.bd]
write_bd_tcl -force "$out/hand_stream_bd.tcl"
puts "HAND_STREAM_PREPARED control=0x43C50000 masters=3"
close_project
