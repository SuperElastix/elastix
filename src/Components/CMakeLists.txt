#---------------------------------------------------------------------
# Sub project name

PROJECT(elxComponents)

#---------------------------------------------------------------------
# Set the libraries to be linked.

# linux gives an error, if param is linked.
IF(CMAKE_COMPILER_IS_GNUCXX)
  SET(elxLinkLibs
    xoutlib
    elxCommon
    elxCore
    ITKBasicFilters
    ITKNumerics
    ITKIO
    ITKCommon )
ENDIF(CMAKE_COMPILER_IS_GNUCXX)

# interestingly, visual c++ does need param
IF(WIN32)
  SET(elxLinkLibs
    param
    xoutlib
    elxCommon
    elxCore
    ITKBasicFilters
    ITKNumerics
    ITKIO
    ITKCommon )
ENDIF(WIN32)

#---------------------------------------------------------------------
# Macro that simplifies the addition of elx-components
# 
# Usage: 
# ADD_ELXCOMPONENT( <name_of_dll> <list_of_source_and_header_files> )
#

MACRO(ADD_ELXCOMPONENT name)
  PROJECT(${name})
  ADD_LIBRARY(${name} MODULE ${ARGN} )
  TARGET_LINK_LIBRARIES( ${name} ${elxLinkLibs} )
  INSTALL_TARGETS(/. ${name})
ENDMACRO(ADD_ELXCOMPONENT)

#---------------------------------------------------------------------
# Search for all components

FILE(GLOB_RECURSE ListOfComponents "*/CMakeLists.txt")

FOREACH(comp ${ListOfComponents} )
  GET_FILENAME_COMPONENT(path_to_comp ${comp} PATH)
  STRING(REGEX REPLACE "${elxComponents_SOURCE_DIR}/" "" relative_path_to_comp ${path_to_comp} )
  SUBDIRS( ${relative_path_to_comp} )
ENDFOREACH(comp)

