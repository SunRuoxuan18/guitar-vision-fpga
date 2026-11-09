set root [file dirname [file dirname [file normalize [info script]]]]
set candidate_build "$root/diagnostics/hand_first_fold_20261026_build"
set candidate_out "$root/output/hand_first_fold_20261026"
set candidate_ip "$root/hls/hand_conv_q16/build_ross_pw_first_fold/solution1/impl/ip"
foreach path [list $candidate_build $candidate_out] {
    if {[file exists $path]} { error "Refusing to overwrite existing candidate path: $path" }
}
if {![file isfile "$candidate_ip/component.xml"]} { error "Missing verified candidate IP" }
if {![file isfile "$root/output/hand_neural_dw_wide/av_system.bit"]} { error "Missing production reference" }
foreach key [array names ::env HAND_NEURAL_*] { unset ::env($key) }
set ::env(HAND_NEURAL_DW_WIDE) 1
set prepare_file [open "$root/scripts/build_hand_neural.tcl" r]
set prepare_script [read $prepare_file]
close $prepare_file
set replacements [list \
    {diagnostics/hand_neural_dw_wide_build} {diagnostics/hand_first_fold_20261026_build} \
    {output/hand_neural_dw_wide} {output/hand_first_fold_20261026} \
    {hls/hand_conv_q16/build_dw_wide/solution1/impl/ip} {hls/hand_conv_q16/build_ross_pw_first_fold/solution1/impl/ip}]
eval [string map $replacements $prepare_script]
set build $candidate_build
set out $candidate_out
if {![file isfile "$build/vision_pl.xpr"]} { error "Candidate preparation failed" }
set implement_file [open "$root/scripts/implement_hand_neural.tcl" r]
set implement_script [read $implement_file]
close $implement_file
eval [string map $replacements $implement_script]
puts "FIRST_FOLD_COMPLETE: $candidate_out"
