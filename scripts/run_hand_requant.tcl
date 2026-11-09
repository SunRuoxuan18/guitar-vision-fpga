set root [file dirname [file dirname [file normalize [info script]]]]
# This path is also explicitly supplied with Vivado's -log argument.
set ::hand_requant_native_log "$root/diagnostics/hand_requant_overlay_build/requant_native.log"
source -notrace "$root/scripts/build_hand_requant.tcl"
source -notrace "$root/scripts/implement_hand_requant.tcl"
