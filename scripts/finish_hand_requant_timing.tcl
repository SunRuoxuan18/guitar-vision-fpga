set root [file dirname [file dirname [file normalize [info script]]]]
set out "$root/output/hand_requant"
set build "$root/diagnostics/hand_requant_overlay_build"
open_checkpoint "$out/recovery_routed.dcp"
set_param general.maxThreads 2
phys_opt_design -directive AggressiveFanoutOpt
phys_opt_design -directive Explore
route_design -directive Quick
report_timing_summary -delay_type min_max -file "$out/timing_final.rpt"
report_timing -delay_type max -max_paths 20 -file "$out/critical_paths_final.rpt"
report_drc -file "$out/drc_final.rpt"
write_checkpoint -force "$out/final_routed.dcp"
set setup [get_property SLACK [lindex [get_timing_paths -delay_type max -max_paths 1] 0]]
set hold [get_property SLACK [lindex [get_timing_paths -delay_type min -max_paths 1] 0]]
if {$setup < 0 || $hold < 0} { error "Requant final timing failed setup=$setup hold=$hold" }
set violations [get_drc_violations -quiet -filter {SEVERITY == Error || SEVERITY == {Critical Warning}}]
if {[llength $violations]} { error "Requant DRC failed: $violations" }
set_property BITSTREAM.CONFIG.USERID 32'h41565331 [current_design]
write_bitstream -force "$out/av_system.bit"
file copy -force "$build/vision_pl.gen/sources_1/bd/vision_pl/hw_handoff/vision_pl.hwh" "$out/av_system.hwh"
puts "HAND_REQUANT_FINAL_PASS setup=$setup hold=$hold"
