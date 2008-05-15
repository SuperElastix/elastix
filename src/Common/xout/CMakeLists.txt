CMAKE_MINIMUM_REQUIRED(VERSION 1.6)
# This project is intended to be built outside the Insight source tree
PROJECT(xout)

SET(xoutcfiles xoutmain.cxx xouttest.cxx)

SET(xouthxxfiles
xoutbase.hxx
xoutsimple.hxx
xoutrow.hxx
xoutcell.hxx )


SET(xouthfiles
xoutbase.h
xoutmain.h
xoutsimple.h
xoutrow.h
xoutcell.h)


IF(CMAKE_COMPILER_IS_GNUCXX)
  #Add -fPIC to the gcc-options. 
  #This is needed for compilation on a 64-bit machine 
  #SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
  #SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
ENDIF(CMAKE_COMPILER_IS_GNUCXX)

# a lib defining the global variable xout.
ADD_LIBRARY(xoutlib xoutmain.cxx ${xouthxxfiles} ${xouthfiles})

# a test program
# ADD_EXECUTABLE(xouttest xouttest.cxx)

# a second test program, which tests the xoutlib
# ADD_EXECUTABLE(xouttest2 xouttest2.cxx)
# TARGET_LINK_LIBRARIES(xouttest2 xoutlib)
