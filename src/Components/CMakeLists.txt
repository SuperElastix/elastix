
PROJECT(elxComponents)

SET(elxLinkLibs
elxCommon
elxCore
param
xoutlib
ITKBasicFilters
ITKNumerics
ITKIO
ITKCommon )

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

FILE(GLOB_RECURSE ListOfComponents "*/CMakeLists.txt")


FOREACH(comp ${ListOfComponents} )
  GET_FILENAME_COMPONENT(path_to_comp ${comp} PATH)
  STRING(REGEX REPLACE "${elxComponents_SOURCE_DIR}/" "" relative_path_to_comp ${path_to_comp} )
  SUBDIRS( ${relative_path_to_comp} )
ENDFOREACH(comp)

