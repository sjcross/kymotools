def get_average_timestep(timestamps):
    s = 0
    
    for (t1,t2) in zip(timestamps[:-1],timestamps[1:]):
        s = s + (t2-t1)

    return s/(len(timestamps)-1)
