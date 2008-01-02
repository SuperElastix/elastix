#---------------------------------------------------------------------
# Sub project name

PROJECT(elxComponents)

#---------------------------------------------------------------------
# Set the libraries to be linked.

# Linux gives an error, if param is linked.
IF(UNIX)
  IF (NOT APPLE)
    SET(elxLinkLibs
      xoutlib
      elxCommon
      elxCore
      ITKBasicFilters
      ITKNumerics
      ITKIO
      ITKCommon
      ITKStatistics 
      KNNlib
      ANNlib )
	ENDIF(NOT APPLE)
ENDIF(UNIX)

# Interestingly, visual c++ and MAC-GCC do need param
IF(WIN32 OR APPLE)
  SET(elxLinkLibs
    param
    xoutlib
    elxCommon
    elxCore
    ITKBasicFilters
    ITKNumerics
    ITKIO
    ITKCommon
    ITKStatistics
    KNNlib
    ANNlib )
ENDIF(WIN32 OR APPLE)

#---------------------------------------------------------------------
# Macro that simplifies the addition of elx-components
# 
# Usage: 
# ADD_ELXCOMPONENT( <name_of_dll> <list_of_source_and_header_files> )
#

MACRO(ADD_ELXCOMPONENT name)
  MARK_AS_ADVANCED( USE_${name} )
  OPTION( USE_${name} "Compile this component." ON )
  IF( USE_${name} )
    PROJECT(${name})
    ADD_LIBRARY(${name} MODULE ${ARGN} ${ELX_GDCMHACK} )
    TARGET_LINK_LIBRARIES( ${name} ${elxLinkLibs} )
    INSTALL_TARGETS(/. ${name})
  ENDIF( USE_${name} )
ENDMACRO(ADD_ELXCOMPONENT)

#---------------------------------------------------------------------
# Search for all components

FILE(GLOB_RECURSE ListOfComponents "*/CMakeLists.txt")

FOREACH(comp ${ListOfComponents} )
  GET_FILENAME_COMPONENT(path_to_comp ${comp} PATH)
  STRING(REGEX REPLACE "${elxComponents_SOURCE_DIR}/" "" relative_path_to_comp ${path_to_comp} )
  SUBDIRS( ${relative_path_to_comp} )
ENDFOREACH(comp)

