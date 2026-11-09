set root [file dirname [file dirname [file normalize [info script]]]]
set out "$root/output/hand_requant"
set build "$root/diagnostics/hand_requant_overlay_build"
open_checkpoint "$build/vision_pl.runs/impl_1/vision_pl_wrapper_routed.dcp"
set_param general.maxThreads 2
report_methodology -json "$out/methodology_before_recovery.json"
write_xdc -force -exclude_physical "$out/constraints.xdc"
report_timing -delay_type max -max_paths 20 -file "$out/critical_paths_before_recovery.rpt"
phys_opt_design -directive AggressiveExplore
route_design -directive MoreGlobalIterations
report_timing_summary -delay_type min_max -file "$out/timing.rpt"
report_timing -delay_type max -max_paths 20 -file "$out/critical_paths.rpt"
report_methodology -json "$out/methodology.json"
report_utilization -hierarchical -hierarchical_depth 2 -file "$out/utilization.rpt"
report_drc -file "$out/drc.rpt"
write_checkpoint -force "$out/recovery_routed.dcp"
set setup [get_property SLACK [lindex [get_timing_paths -delay_type max -max_paths 1] 0]]
set hold [get_property SLACK [lindex [get_timing_paths -delay_type min -max_paths 1] 0]]
if {$setup < 0 || $hold < 0} { error "Requant timing recovery failed setup=$setup hold=$hold" }
set violations [get_drc_violations -quiet -filter {SEVERITY == Error || SEVERITY == {Critical Warning}}]
if {[llength $violations]} { error "Requant DRC failed: $violations" }
set_property BITSTREAM.CONFIG.USERID 32'h41565331 [current_design]
write_bitstream -force "$out/av_system.bit"
file copy -force "$build/vision_pl.gen/sources_1/bd/vision_pl/hw_handoff/vision_pl.hwh" "$out/av_system.hwh"
puts "HAND_REQUANT_RECOVERY_PASS setup=$setup hold=$hold"
