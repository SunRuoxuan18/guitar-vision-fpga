set here [file dirname [file normalize [info script]]]
set variant $::env(FRETLENS_PREOPT_VARIANT)
if {$variant ni {q16 q8 q8_fabric w8a16 w8a16_spatial}} { error "Unknown preopt variant" }
set probe_root [file normalize "$here/../../diagnostics/hand_conv_preopt_20261028"]
file mkdir $probe_root
set build_path "$probe_root/$variant"
if {[info exists ::env(FRETLENS_PREOPT_REV)]} {
    set revision $::env(FRETLENS_PREOPT_REV)
    if {![regexp {^[a-zA-Z0-9_]+$} $revision]} { error "Invalid probe revision" }
    append build_path "_$revision"
}
if {[info exists ::env(FRETLENS_PREOPT_COSIM)]} {
    open_project $build_path
    open_solution solution1
    cosim_design -rtl verilog -tool xsim -argv "smoke"
    exit
}
if {[file exists "$build_path/hls.app"]} { error "Probe already exists: $build_path" }
set flags "-std=c++14 -DHAND_POINTWISE8 -DHAND_BURST -DHAND_COMPACT -DHAND_BANKED -DHAND_DW_CACHE -DHAND_PW_FUSED -DHAND_DW_WIDE -DHAND_PW_FIRST_FOLD"
if {$variant eq "q16"} { append flags " -DPREOPT_VALUE_BITS=16" }
if {$variant eq "q8_fabric"} { append flags " -DPREOPT_FABRIC_MUL" }
if {$variant eq "w8a16_spatial"} { append flags " -DPREOPT_SPATIAL_FABRIC" }
open_project $build_path
set_top hand_conv_preopt
set suffix ""
if {$variant in {w8a16 w8a16_spatial}} { set suffix "_w8a16" }
add_files "$here/hand_conv_preopt$suffix.cpp" -cflags $flags
add_files -tb "$here/test_hand_conv_preopt$suffix.cpp" -cflags $flags
open_solution solution1 -flow_target vivado
set_part {xc7z020clg400-1}
create_clock -period 10 -name default
csim_design
csynth_design
exit
