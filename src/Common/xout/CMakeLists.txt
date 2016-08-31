
# This project is intended to be built outside the Insight source tree
project( xout )

set( xoutcfiles xoutmain.cxx xouttest.cxx )

set( xouthxxfiles
  xoutbase.hxx
  xoutsimple.hxx
  xoutrow.hxx
  xoutcell.hxx )

set( xouthfiles
  xoutbase.h
  xoutmain.h
  xoutsimple.h
  xoutrow.h
  xoutcell.h )

# a lib defining the global variable xout.
add_library( xoutlib STATIC xoutmain.cxx ${xouthxxfiles} ${xouthfiles} )

# Group in IDE's like Visual Studio
set_property( TARGET xoutlib PROPERTY FOLDER "libraries" )
