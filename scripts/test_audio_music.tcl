set root [file dirname [file dirname [file normalize [info script]]]]
open_project "$root/diagnostics/audio_music_build/audio_music.xpr"
add_files -fileset sim_1 -norecurse [glob "$root/sim/*.v"]
add_files -fileset sim_1 -norecurse [glob "$root/sim/vectors/*.mem"]
set_property file_type SystemVerilog [get_files "$root/sim/tb_audio_fft_engine.v"]
set_property top tb_audio_fft_engine [get_filesets sim_1]
update_compile_order -fileset sim_1
launch_simulation -simset sim_1 -mode behavioral
run all
close_sim
close_project
