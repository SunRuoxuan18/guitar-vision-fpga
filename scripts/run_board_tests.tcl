set project_file "D:/FPGA/ZynqAudioLab/ZynqAudioLab.xpr"
open_project $project_file

set_property top tb_adau1761_init [get_filesets sim_1]
update_compile_order -fileset sim_1
launch_simulation -simset sim_1 -mode behavioral
run all
close_sim

set_property top tb_pynq_audio_stream_top [get_filesets sim_1]
update_compile_order -fileset sim_1
launch_simulation -simset sim_1 -mode behavioral
run all
close_sim

set_property top tb_pynq_audio_stream_top [get_filesets sim_1]
update_compile_order -fileset sim_1
close_project
