#define main full_reference_main
#include "test_hand_conv_q16.cpp"
#undef main

int main(int argc, char **) {
    bool ok = true;
    if (argc == 1) {
        ok &= full_reference_main() == 0;
        ok &= check(5,35,16,64,1,1,0,0,0,0,1);
        ok &= check(5,33,24,17,1,2,1,1,0,0,1);
    }
    // All co-simulation operands and results fit the 4096-beat interfaces.
    ok &= check(1,1,256,32,1,1,0,0,0,0,1);
    ok &= check(7,7,64,64,1,1,0,0,0,0,1);
    ok &= check(7,7,16,64,1,1,0,0,0,0,1);
    ok &= check(3,17,13,9,1,1,0,0,0,0,1);
    ok &= check(1,1,3,9,1,2,1,1,1,1,1);
    return ok ? 0 : 1;
}
