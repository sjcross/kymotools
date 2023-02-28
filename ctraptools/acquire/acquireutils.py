import bluelake as bl
import time

def identify_mirror(mirror_num, test_disp=2, speed=1, is_quad=False):
    if mirror_num == 1:
        mirror = bl.mirror1
    elif mirror_num == 2:
        mirror = bl.mirror2
    elif mirror_num == 3:
        mirror = bl.mirror3
    elif mirror_num == 4:
        mirror = bl.mirror4

    # Include special consideration if using mirror 1, as this needs to have space within its range to move
    n_beads = 4 if is_quad else 2

    # Store bead positions before moving mirror
    x1 = []
    for bead in range(1, n_beads+1):
        x1.append(bl.timeline["Bead position"]["Bead " + str(bead) + " X"].latest_value)

    # Moving mirror by a know amount
    mirror.move_by(dx=test_disp, dy=0, speed=speed)

    # Retesting bead positions and taking one with largest displacement as target.  Bead must have moved by at least 50%
    # the applied mirror displacement
    this_bead = -1
    max_disp = 0
    for bead in range(1, n_beads+1):
        disp = bl.timeline["Bead position"]["Bead " + str(bead) + " X"].latest_value - x1[bead-1]

        if disp > max_disp and disp > test_disp*0.5:
            max_disp = disp
            this_bead = bead

    # Resetting mirror to original location
    mirror.move_by(dx=-test_disp, dy=0, speed=speed)

    if this_bead == -1:
        print("Mirror ", mirror_num, " didn't find a bead (disp = ", max_disp, ")")
    else:
        print("Mirror ", mirror_num, " is bead ", this_bead, " (disp = ", max_disp, ")")

    return this_bead

def set_pressure(pressure):
    print()
    if bl.fluidics.pressure < pressure:
        # If we've started too low, increase the pressure till we're at or above the target
        while bl.fluidics.pressure < pressure:
            bl.fluidics.increase_pressure()
    else:
        # If we've started too high, decrease the pressure till we're at or below the target
        while bl.fluidics.pressure > pressure:
            bl.fluidics.decrease_pressure()




## DEMO TO SHOW CATCHING DNA
# Parameters
beads_ch = 'beads'
flow_pressure = 0.2
match_thresh = 99
bead_clear_thresh_s = 30 # If both beads haven't been caught yet, clear the traps

# Go to beads channel
bl.microstage.move_to(beads_ch)

# Turn on flow
set_pressure(0.2)
bl.fluidics.open(1,2,3,4,5,6)

# Empty traps of existing debris
bl.shutters.clear(1,2,3,4)

start_t = time.time()
while bl.timeline["Tracking Match Score"]["Bead 1"].latest_value < match_thresh or bl.timeline["Tracking Match Score"]["Bead 2"].latest_value < match_thresh:
    # If we've still not caught a bead, clear the traps and retry
    if (time.time() - start_t) > bead_clear_thresh_s:
        bl.shutters.clear(1, 2, 3, 4)
        start_t = time.time()

    time.sleep(0.1)

mirror_1 = identify_mirror(1)
mirror_2 = identify_mirror(2)

# Go to DNA channel

# Put traps far enough apart (i.e. check they don't overlap)
