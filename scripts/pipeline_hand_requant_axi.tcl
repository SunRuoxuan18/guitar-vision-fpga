set root [file dirname [file dirname [file normalize [info script]]]]
set build "$root/diagnostics/hand_requant_overlay_build"
set out "$root/output/hand_requant"
open_project "$build/vision_pl.xpr"
cd $build
set_param general.maxThreads 2
open_bd_design [get_files vision_pl.bd]
if {[info exists ::env(VIVADO_IPCFG_TCL)]} {
    source -notrace $::env(VIVADO_IPCFG_TCL)
}
set ::ipcfg::native_log_path "$build/pipelined_selective.log"
puts [ipcfg::python_status]
puts [ipcfg::python_exec -c {import argparse, base64, hashlib, json, pathlib, re, sys, typing; print("PYTHON_IMPORTS:OK")}]
set settings [dict create CONFIG.M00_HAS_REGSLICE 1]
set count [get_property CONFIG.NUM_SI [get_bd_cells memory_axi]]
for {set index 0} {$index < $count} {incr index} {
    dict set settings [format "CONFIG.S%02d_HAS_REGSLICE" $index] [expr {$index >= $count-2 ? 1 : 0}]
}
set receipt [ipcfg::capture "$out/pipelined.[clock clicks].tcldict" \
    [list ipcfg::require memory_axi $settings {Break measured DDR write-ready path only at HP output and requant inputs, limiting control-set usage}] {REQ:pinned+applied}]
puts $receipt
if {[dict get $receipt status] ne "OK"} { error "AXI pipeline configuration failed" }
dict for {key value} $settings {
    if {[get_property $key [get_bd_cells memory_axi]] != $value} { error "AXI register slice readback failed $key" }
}
validate_bd_design
save_bd_design
generate_target all [get_files vision_pl.bd]
write_bd_tcl -force "$out/hand_requant_bd.tcl"
close_project
source "$root/scripts/implement_hand_requant.tcl"
