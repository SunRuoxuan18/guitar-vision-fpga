# Recover an unresponsive Linux boot through the board's hardware reset.
connect -url tcp:127.0.0.1:3121
targets -set -filter {name =~ "APU"}
rst -system
puts "PS_SYSTEM_RESET_REQUESTED"
disconnect
