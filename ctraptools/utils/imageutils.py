def apply_normalisation(image, output_range):
    if output_range is not None:
        for c in range(min(image.shape[2],len(output_range))):
            image[:,:,c] = (image[:,:,c]-output_range[c][0])/(output_range[c][1]-output_range[c][0])
    
    return image

def get_output_range_string(output_range):
    if output_range is None:
        return "Default"
    
    or_str = ""
    for c in range(len(output_range)):
        if c > 0:
            or_str = or_str + ", "
            
        or_str = ("%sC%i:%i-%i" % (or_str,c,output_range[c][0],output_range[c][1]))
            
    return or_str