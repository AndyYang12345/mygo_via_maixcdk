################# Add include #################
list(APPEND ADD_INCLUDE "include")
###############################################

############## Add source files ###############
append_srcs_dir(ADD_SRCS "src")
###############################################

###### Add required/dependent components ######
list(APPEND ADD_REQUIREMENTS basic opencv)
if(PLATFORM_MAIXCAM OR PLATFORM_MAIXCAM2)
    list(APPEND ADD_REQUIREMENTS vision peripheral)
    list(APPEND ADD_DEFINITIONS -DMYGO_TARGETTRACKING_USE_MAIX=1)
endif()
###############################################

# register component
register_component()
