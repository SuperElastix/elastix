
# This project is intended to be built outside the Insight source tree
project(xout)

set(xoutcfiles xoutmain.cxx xouttest.cxx)

set(xoutcxxfiles
  xoutbase.cxx
  xoutmain.cxx
  xoutsimple.cxx
  xoutrow.cxx
  xoutcell.cxx)

set(xouthfiles
  xoutbase.h
  xoutmain.h
  xoutsimple.h
  xoutrow.h
  xoutcell.h)

# a lib defining the global variable xout.
add_library(xoutlib STATIC ${xoutcxxfiles} ${xouthfiles})
if(NOT ELASTIX_NO_INSTALL_DEVELOPMENT)
  install(TARGETS xoutlib
    ARCHIVE DESTINATION ${ELASTIX_ARCHIVE_DIR}
    LIBRARY DESTINATION ${ELASTIX_LIBRARY_DIR}
    RUNTIME DESTINATION ${ELASTIX_RUNTIME_DIR}
    COMPONENT Development)
endif()

# Group in IDE's like Visual Studio
set_property(TARGET xoutlib PROPERTY FOLDER "libraries")
