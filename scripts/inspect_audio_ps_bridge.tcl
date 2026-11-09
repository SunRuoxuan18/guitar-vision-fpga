connect -url tcp:127.0.0.1:3121
puts "TARGETS: [targets]"
targets -set -filter {name =~ "APU"}
foreach address {0xF800012C 0xF8000170 0xF8000178 0xF8000240 0xF8000900 0xF8007000 0xF800700C 0xF8007080} {
    puts [mrd $address]
}
disconnect
