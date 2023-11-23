def strip_ext(file_name):
    file_name_split = file_name.split(".h5")
    file_name_no_ext = file_name_split[0]
    return(file_name_no_ext)