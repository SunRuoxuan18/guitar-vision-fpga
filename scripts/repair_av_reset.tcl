set root [file dirname [file dirname [file normalize [info script]]]]
set build "$root/diagnostics/av_system_build"
cd $build
set_param general.maxThreads 1
open_project "$build/vision_pl.xpr"
add_files -norecurse [glob "$root/rtl/*.v"]
open_bd_design [get_files vision_pl.bd]
set pin [get_bd_pins audio_reset_control/aux_reset_in]
disconnect_bd_net [get_bd_nets -of_objects $pin] $pin
set_property CONFIG.C_AUX_RESET_HIGH {0} [get_bd_cells audio_reset_control]
connect_bd_net [get_bd_pins logic_high/dout] $pin
validate_bd_design
save_bd_design
generate_target all [get_files vision_pl.bd]
update_compile_order -fileset sources_1
reset_run synth_1
reset_run vision_pl_music_0_synth_1
foreach run [get_runs -filter {IS_SYNTHESIS == 1}] {
    if {![string match "*Complete*" [get_property STATUS $run]]} {
        reset_run $run
    }
}
write_bd_tcl -force "$root/output/av_system/av_system_bd.tcl"
close_project
puts "AV_RESET_REPAIRED"
