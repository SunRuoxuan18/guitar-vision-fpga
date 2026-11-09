# Isolated extension of the verified dw_wide overlay. Never edits its source.
set root [file dirname [file dirname [file normalize [info script]]]]
set build "$root/diagnostics/hand_requant_overlay_build"
set out "$root/output/hand_requant"
set source_project "$root/diagnostics/hand_neural_dw_wide_build/vision_pl.xpr"
set repository "$root/hls/hand_requant/build/solution1/impl/ip"
if {![file isfile "$repository/component.xml"]} { error "Requant IP not exported" }
if {[file exists "$build/vision_pl.xpr"]} { error "Refusing to overwrite prepared build $build" }
file mkdir $out
open_project $source_project
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
set bd_path [file normalize [get_files vision_pl.bd]]
if {[string first $build $bd_path] != 0} { error "BD not isolated: $bd_path" }
if {[info exists ::env(VIVADO_IPCFG_TCL)]} {
    source -notrace $::env(VIVADO_IPCFG_TCL)
}
if {![info exists ::hand_requant_native_log]} { error "Missing Vivado session native log receipt" }
set ::ipcfg::native_log_path $::hand_requant_native_log
puts [ipcfg::python_status]
puts [ipcfg::python_exec -c {import argparse, base64, hashlib, json, pathlib, re, sys, typing; print("PYTHON_IMPORTS:OK")}]
set definitions [get_ipdefs -all -filter {NAME == hand_requant}]
if {[llength $definitions] != 1} { error "Requant IP not unique: $definitions" }
set created [ipcfg::create_cell [lindex $definitions 0] hand_requant]
puts $created
if {[llength [get_bd_cells -quiet hand_requant]] != 1} { error "Requant creation failed" }
set cell [get_bd_cells hand_requant]
set masters [get_bd_intf_pins -of_objects $cell -filter {MODE == Master}]
set control_index [get_property CONFIG.NUM_MI [get_bd_cells control_axi]]
set memory_index [get_property CONFIG.NUM_SI [get_bd_cells memory_axi]]
set control_count [expr {$control_index+1}]
set memory_count [expr {$memory_index+[llength $masters]}]
set stamp [clock clicks]
set receipt [ipcfg::capture "$out/control.$stamp.tcldict" \
    [list ipcfg::require control_axi [dict create CONFIG.NUM_MI $control_count] {Requant AXI-Lite control port}] {REQ:pinned+applied}]
puts $receipt
if {[dict get $receipt status] ne "OK"} { error "Control configuration failed" }
set receipt [ipcfg::capture "$out/memory.$stamp.tcldict" \
    [list ipcfg::require memory_axi [dict create CONFIG.NUM_SI $memory_count] {Requant DDR master ports}] {REQ:pinned+applied}]
puts $receipt
if {[dict get $receipt status] ne "OK"} { error "Memory configuration failed" }
if {[get_property CONFIG.NUM_MI [get_bd_cells control_axi]] != $control_count ||
    [get_property CONFIG.NUM_SI [get_bd_cells memory_axi]] != $memory_count} { error "Interconnect readback failed" }
connect_bd_intf_net [get_bd_intf_pins [format "control_axi/M%02d_AXI" $control_index]] [get_bd_intf_pins hand_requant/s_axi_control]
set clk [get_bd_pins ps7/FCLK_CLK0]
set reset [get_bd_pins reset_control/peripheral_aresetn]
foreach pin [get_bd_pins -of_objects $cell -filter {TYPE == clk}] { connect_bd_net $clk $pin }
foreach pin [get_bd_pins -of_objects $cell -filter {TYPE == rst}] { connect_bd_net $reset $pin }
connect_bd_net $clk [get_bd_pins [format "control_axi/M%02d_ACLK" $control_index]]
connect_bd_net $reset [get_bd_pins [format "control_axi/M%02d_ARESETN" $control_index]]
assign_bd_address -offset 0x43C40000 -range 0x00010000 \
    -target_address_space [get_bd_addr_spaces ps7/Data] [get_bd_addr_segs hand_requant/s_axi_control/Reg] -force
foreach master $masters {
    set name [get_property NAME $master]
    connect_bd_intf_net $master [get_bd_intf_pins [format "memory_axi/S%02d_AXI" $memory_index]]
    connect_bd_net $clk [get_bd_pins [format "memory_axi/S%02d_ACLK" $memory_index]]
    connect_bd_net $reset [get_bd_pins [format "memory_axi/S%02d_ARESETN" $memory_index]]
    set space [get_bd_addr_spaces "hand_requant/Data_$name"]
    assign_bd_address -offset 0 -range 0x20000000 -target_address_space $space \
        [get_bd_addr_segs ps7/S_AXI_HP0/HP0_DDR_LOWOCM] -force
    incr memory_index
}
validate_bd_design
save_bd_design
generate_target all [get_files vision_pl.bd]
write_bd_tcl -force "$out/hand_requant_bd.tcl"
puts "HAND_REQUANT_PREPARED control=0x43C40000 masters=[llength $masters]"
close_project
