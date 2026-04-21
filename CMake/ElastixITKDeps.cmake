#
# ElastixITKDeps.cmake
#
# Provides a helper macro and convenience variables for linking ITK dependencies
# using modern CMake interface library targets (ITK 6+) or the legacy
# ${ITK_LIBRARIES} variable (ITK 5.x), maintaining backward compatibility.
#
# Must be included AFTER find_package(ITK ...) and, for ITK 6+, after
# itk_generate_factory_registration() has been called so that factory
# meta-module target properties are properly configured.
#
# Variables set by this file
# --------------------------
#
#   ELASTIX_ITK_EXECUTABLE_LIBRARIES
#     For executables that already carry ITK processing module dependencies
#     transitively via elxCommon (e.g. elastix_exe, transformix_exe, GTest targets).
#     ITK 6+: IO factory meta-module targets (ITK::ITKImageIO, etc.) that provide
#             the factory registration include directories and compile definitions.
#     ITK 5.x: ${ITK_LIBRARIES} (factory registration handled globally by UseITK.cmake).
#
#   ELASTIX_ITK_ALL_LIBRARIES
#     For standalone executables that do NOT link elxCommon/elxCore and therefore
#     need explicit ITK processing module dependencies as well as IO factories
#     (e.g. elxComputeOverlap, elxImageCompare, external project examples).
#     ITK 6+: ${ITK_INTERFACE_LIBRARIES} (all requested module interface libs)
#             plus IO factory meta-module targets.
#     ITK 5.x: ${ITK_LIBRARIES}.
#
# Macro provided
# --------------
#
#   elastix_link_itk(<target> [PUBLIC|PRIVATE|INTERFACE] MODULES <mod1> [<mod2> ...])
#
#   Links <target> to specific ITK modules using:
#     ITK 6+: ITK::<mod>Module interface library targets
#     ITK 5.x: ${ITK_LIBRARIES} (all loaded modules)
#
#   The optional scope keyword (PUBLIC / PRIVATE / INTERFACE) controls CMake
#   link visibility and may be omitted for old-style plain linkage.
#
#   Example (library that exposes ITK types in its public headers):
#     elastix_link_itk(elxCommon PUBLIC MODULES
#       ITKCommon ITKTransform ITKOptimizers ITKRegistrationCommon)
#
#   Example (internal library with no public ITK API):
#     elastix_link_itk(param PRIVATE MODULES ITKCommon)
#

if(ITK_VERSION_MAJOR GREATER_EQUAL 6)
  # Collect IO factory meta-module targets configured by itk_generate_factory_registration().
  # These carry INTERFACE_INCLUDE_DIRECTORIES pointing to the generated factory
  # registration headers and INTERFACE_COMPILE_DEFINITIONS enabling the managers.
  set(ELASTIX_ITK_EXECUTABLE_LIBRARIES "")
  foreach(_elx_meta ITKImageIO ITKTransformIO ITKMeshIO)
    if(TARGET ITK::${_elx_meta})
      list(APPEND ELASTIX_ITK_EXECUTABLE_LIBRARIES ITK::${_elx_meta})
    endif()
  endforeach()
  unset(_elx_meta)

  # ITKTestKernel provides test utilities (itkTestingComparisonImageFilter.h, etc.).
  # Linked only to test targets; not added to production libraries.
  if(TARGET ITK::ITKTestKernelModule)
    set(ELASTIX_ITK_TEST_LIBRARIES ITK::ITKTestKernelModule)
  else()
    set(ELASTIX_ITK_TEST_LIBRARIES "")
  endif()

  # Full ITK: all requested module interface libraries + IO factory meta-modules.
  # ITK_INTERFACE_LIBRARIES is set by itk_module_config() in ITKConfig.cmake
  # and contains one ITK::<mod>Module target per loaded ITK module.
  set(ELASTIX_ITK_ALL_LIBRARIES ${ITK_INTERFACE_LIBRARIES} ${ELASTIX_ITK_EXECUTABLE_LIBRARIES})

else()
  # ITK 5.x: the single ${ITK_LIBRARIES} variable covers all requested modules
  # and IO factories (factory registration is done globally by UseITK.cmake).
  set(ELASTIX_ITK_EXECUTABLE_LIBRARIES ${ITK_LIBRARIES})
  set(ELASTIX_ITK_TEST_LIBRARIES "")
  set(ELASTIX_ITK_ALL_LIBRARIES ${ITK_LIBRARIES})
endif()

#
# elastix_link_itk(<target> [PUBLIC|PRIVATE|INTERFACE] MODULES <mod1> [<mod2> ...])
#
function(elastix_link_itk _elt_target)
  cmake_parse_arguments(PARSE_ARGV 1 _elt "PUBLIC;PRIVATE;INTERFACE" "" "MODULES")
  if(_elt_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR "elastix_link_itk: unrecognized arguments: ${_elt_UNPARSED_ARGUMENTS}")
  endif()
  if(NOT _elt_MODULES)
    message(FATAL_ERROR "elastix_link_itk: MODULES keyword with at least one module name is required")
  endif()
  set(_elt_scope "")
  foreach(_elt_kw IN ITEMS PUBLIC PRIVATE INTERFACE)
    if(_elt_${_elt_kw})
      set(_elt_scope ${_elt_kw})
    endif()
  endforeach()
  if(ITK_VERSION_MAJOR GREATER_EQUAL 6)
    set(_elt_itk_link_targets "")
    foreach(_elt_mod IN LISTS _elt_MODULES)
      list(APPEND _elt_itk_link_targets ITK::${_elt_mod}Module)
    endforeach()
    target_link_libraries(${_elt_target} ${_elt_scope} ${_elt_itk_link_targets})
  else()
    target_link_libraries(${_elt_target} ${_elt_scope} ${ITK_LIBRARIES})
  endif()
endfunction()
