#
# To use elastix-code in your own program, add the following
# cmake code to your CMakeLists file:
#
# set( Elastix_DIR "path/to/elastix/binary/directory" )
# find_package( Elastix REQUIRED )
# include( ${ELASTIX_USE_FILE} )
# 

# Add include dirs
include_directories( ${ELASTIX_INCLUDE_DIRS} )

# Add library dirs
link_directories( ${ELASTIX_LIBRARY_DIRS} )

# If Elastix_FOUND is set, this file is included via find_package() which provides
# ELASTIX_CONFIG_TARGETS_FILE and elxLIBRARY_DEPENDS_FILE. Guarding the following
# include statements allow users to include this file directly for backwards 
# compatibility.
if( Elastix_FOUND )
  # This file was found by find_package
  include( ${ELASTIX_CONFIG_TARGETS_FILE} )
else()
  # Include linking dependency info for backwards compatibility
  include( ${elxLIBRARY_DEPENDS_FILE} )
endif()
