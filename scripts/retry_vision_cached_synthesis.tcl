set root [file dirname [file dirname [file normalize [info script]]]]
set_param general.maxThreads 1
open_project "$root/diagnostics/vision_cached_build/vision_pl.xpr"
set run [get_runs vision_pl_lk_pyramid_0_synth_1]
reset_run $run
reset_run synth_1
set_property STEPS.SYNTH_DESIGN.TCL.PRE "$root/scripts/vision_single_thread.tcl" $run
launch_runs synth_1 -jobs 1
wait_on_run synth_1
if {[get_property STATUS [get_runs synth_1]] ne "synth_design Complete!"} {
    error "Retry synthesis failed"
}
close_project
