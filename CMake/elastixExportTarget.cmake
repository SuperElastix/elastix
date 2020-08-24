function(elastix_export_target tgt)
  # Remove the build tree's ElastixTargets file if this is the first call:
  get_property(first_time GLOBAL PROPERTY ELASTIX_FIRST_EXPORTED_TARGET)
  if(NOT first_time)
    file(REMOVE ${elastix_BINARY_DIR}/ElastixTargets.cmake)
    set_property(GLOBAL PROPERTY ELASTIX_FIRST_EXPORTED_TARGET 1)
  endif()

  get_target_property( type ${tgt} TYPE )
  if (type STREQUAL "STATIC_LIBRARY" OR
      type STREQUAL "MODULE_LIBRARY" OR
      type STREQUAL "SHARED_LIBRARY")
    set_property(TARGET ${tgt} PROPERTY VERSION 1)
    set_property(TARGET ${tgt} PROPERTY SOVERSION 1)

    if ("${tgt}" STREQUAL "elastix_lib")
      set_property(TARGET ${tgt} PROPERTY
        OUTPUT_NAME elastix-${ELASTIX_VERSION_MAJOR}.${ELASTIX_VERSION_MINOR})
    elseif ("${tgt}" STREQUAL "transformix_lib")
      set_property(TARGET ${tgt} PROPERTY
        OUTPUT_NAME transformix-${ELASTIX_VERSION_MAJOR}.${ELASTIX_VERSION_MINOR})
    else()
      set_property(TARGET ${tgt} PROPERTY
        OUTPUT_NAME ${tgt}-${ELASTIX_VERSION_MAJOR}.${ELASTIX_VERSION_MINOR})
    endif()
  endif()

  export(TARGETS ${tgt}
    APPEND FILE "${elastix_BINARY_DIR}/ElastixTargets.cmake"
  )
endfunction()
