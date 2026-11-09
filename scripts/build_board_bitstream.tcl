set project_file "D:/FPGA/ZynqAudioLab/ZynqAudioLab.xpr"
open_project $project_file

set_property top pynq_audio_board_top [get_filesets sources_1]
update_compile_order -fileset sources_1

reset_run synth_1
launch_runs synth_1 -jobs 4
wait_on_run synth_1
if {[get_property STATUS [get_runs synth_1]] ne "synth_design Complete!"} {
    error "Synthesis did not complete successfully."
}

launch_runs impl_1 -to_step write_bitstream -jobs 4
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] ne "100%"} {
    error "Implementation/bitstream did not complete successfully."
}

file mkdir "D:/FPGA/ZynqAudioLab/output"
file copy -force \
    "D:/FPGA/ZynqAudioLab/ZynqAudioLab.runs/impl_1/pynq_audio_board_top.bit" \
    "D:/FPGA/ZynqAudioLab/output/pynq_audio_board_top.bit"

open_run impl_1
report_utilization -file "D:/FPGA/ZynqAudioLab/output/board_utilization.rpt"
report_timing_summary -file "D:/FPGA/ZynqAudioLab/output/board_timing.rpt"
close_project
