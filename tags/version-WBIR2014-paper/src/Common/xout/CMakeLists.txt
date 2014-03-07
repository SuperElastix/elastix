CMAKE_MINIMUM_REQUIRED( VERSION 1.6 )

# This project is intended to be built outside the Insight source tree
PROJECT( xout )

SET( xoutcfiles xoutmain.cxx xouttest.cxx )

SET( xouthxxfiles
  xoutbase.hxx
  xoutsimple.hxx
  xoutrow.hxx
  xoutcell.hxx )

SET( xouthfiles
  xoutbase.h
  xoutmain.h
  xoutsimple.h
  xoutrow.h
  xoutcell.h )

# a lib defining the global variable xout.
ADD_LIBRARY( xoutlib xoutmain.cxx ${xouthxxfiles} ${xouthfiles} )

# Group in IDE's like Visual Studio
SET_PROPERTY( TARGET xoutlib PROPERTY FOLDER "libraries" )
