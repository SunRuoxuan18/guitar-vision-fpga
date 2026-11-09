set project_file "D:/FPGA/ZynqAudioLab/ZynqAudioLab.xpr"
open_project $project_file

set rtl_file "D:/FPGA/ZynqAudioLab/rtl/audio_status_axi_lite.v"
set tb_file  "D:/FPGA/ZynqAudioLab/sim/tb_audio_status_axi_lite.v"

if {[llength [get_files -quiet $rtl_file]] == 0} {
    add_files -norecurse $rtl_file
}

if {[llength [get_files -quiet $tb_file]] == 0} {
    add_files -fileset sim_1 -norecurse $tb_file
}

set previous_top [get_property top [get_filesets sim_1]]
set_property top tb_audio_status_axi_lite [get_filesets sim_1]
update_compile_order -fileset sources_1
update_compile_order -fileset sim_1

set simulation_failed [catch {
    launch_simulation -simset sim_1 -mode behavioral
    run all
} simulation_message]

catch {close_sim}
set_property top $previous_top [get_filesets sim_1]
update_compile_order -fileset sim_1
close_project

if {$simulation_failed} {
    error $simulation_message
}
