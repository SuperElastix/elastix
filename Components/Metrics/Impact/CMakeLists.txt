#---------------------------------------------------------------------
# Macros that simplify the addition of elx-components with a dependency on Torch
#
# Usage:
# ADD_ELXCOMPONENT_TORCH( <name_of_lib> <ON|OFF> <list_of_source_and_header_files>)
#
# This macro behaves like ADD_ELXCOMPONENT but explicitly links the component
# with LibTorch. It is intended for components that require Torch models.
#
# The name_of_lib should be equal to the name of the elx class,
# which also should be the name used in the elxInstallMacro, in the
# the .cxx file of the component.
#
macro(ADD_ELXCOMPONENT_TORCH name)

  # Compile this component or not
  set(defaultValue ON)
  if(${ARGV1} STREQUAL "OFF")
    set(defaultValue OFF)
  endif()
  mark_as_advanced(USE_${name})
  set(USE_${name} ${defaultValue} CACHE BOOL "Compile this component")

  # If USE_ALL_COMPONENTS is turned ON, we make a backup of the
  # current value, and force the use_var to ON. If USE_ALL_COMPONENTS
  # is OFF, and there is a backup, set the use_var to the backed-up
  # value and remove the backup; otherwise, leave the use_var to its
  # current (default or user-specified) value.
  if(USE_ALL_COMPONENTS)
    # make backup
    if(NOT DEFINED USE_${name}_BACKUP)
      if(USE_${name})
        set(USE_${name}_BACKUP ON CACHE INTERNAL "Backup of USE_var" FORCE)
      else()
        set(USE_${name}_BACKUP OFF CACHE INTERNAL "Backup of USE_var" FORCE)
      endif()
    endif()
    # turn use_var to on;
    set(USE_${name} ON CACHE BOOL "Compile this component" FORCE)
  else()
    # if there is a backup, set the use_var to the backed-up value
    # and remove the backup.
    if(DEFINED USE_${name}_BACKUP)
      # this unset command is important; it makes sure the use_var is reset
      unset(USE_${name} CACHE)
      if(USE_${name}_BACKUP)
        set(USE_${name} ON CACHE BOOL "Compile this component")
      else()
        set(USE_${name} OFF CACHE BOOL "Compile this component")
      endif()
      unset(USE_${name}_BACKUP CACHE)
    endif()
  endif()

  if(USE_${name})
    # Create the list of files which create the library
    set(filelist ${ARGN})
    list(REMOVE_ITEM filelist "ON" "OFF")

    # Create project and static library
    project(${name})
    add_library(${name} STATIC ${filelist})

    # Custom handling (different from ADD_ELXCOMPONENT): manually find Torch and link it to the component
    find_package(Torch REQUIRED)
    target_link_libraries(${name} ${elxLinkLibs} torch)


    if(NOT ELASTIX_NO_INSTALL_DEVELOPMENT)
      install(TARGETS ${name}
        ARCHIVE DESTINATION ${ELASTIX_ARCHIVE_DIR}
        LIBRARY DESTINATION ${ELASTIX_LIBRARY_DIR}
        RUNTIME DESTINATION ${ELASTIX_RUNTIME_DIR}
        COMPONENT Development)
    endif()

    # Group in IDE's like Visual Studio
    string(REGEX MATCH "[a-zA-Z]+" comp_group ${relative_path_to_comp})
    set_property(TARGET ${name} PROPERTY FOLDER ${comp_group})

    # Update AllComponentLibs, while avoiding duplicates:
    if(DEFINED AllComponentLibs)
      set(dummy ${AllComponentLibs})
      if(NOT "${dummy}" STREQUAL "")
        list(REMOVE_ITEM dummy ${name})
      endif()
      set(dummy ${dummy} ${name})
    else()
      set(dummy ${name})
    endif()
    set(AllComponentLibs ${dummy} CACHE INTERNAL "All component libraries for elastix/transformix")

    # Write lines to two files
    file(APPEND ${InstallFunctionDeclarationFile}
      "elxInstallComponentFunctionDeclarationMacro(" ${name} ");\n\n")
    file(APPEND ${InstallFunctionCallFile}
      "elxInstallComponentFunctionCallMacro(" ${name} ");\n\n")

  else(USE_${name})
    # Remove from link list
    REMOVE_ELXCOMPONENT( ${name})
  endif()

endmacro(ADD_ELXCOMPONENT_TORCH)

ADD_ELXCOMPONENT_TORCH(ImpactMetric OFF
 elxImpactMetric.cxx
 elxImpactMetric.h
 elxImpactMetric.hxx
 ImpactLoss.hxx
 ImpactTensorUtils.h
 ImpactTensorUtils.hxx
 itkBSplineInterpolateVectorImageFunction.h
 itkBSplineInterpolateVectorImageFunction.hxx
 itkImpactImageToImageMetric.h
 itkImpactImageToImageMetric.hxx
 )
