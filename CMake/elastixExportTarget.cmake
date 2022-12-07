function(elastix_export_target tgt)
  # Remove the build tree's ElastixTargets file if this is the first call:
  get_property(first_time GLOBAL PROPERTY ELASTIX_FIRST_EXPORTED_TARGET)
  if(NOT first_time)
    file(REMOVE ${elastix_BINARY_DIR}/ElastixTargets.cmake)
    set_property(GLOBAL PROPERTY ELASTIX_FIRST_EXPORTED_TARGET 1)
  endif()

  get_target_property(type ${tgt} TYPE)
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

    if(type STREQUAL "STATIC_LIBRARY")
      if(NOT ELASTIX_NO_INSTALL_DEVELOPMENT)
        install(TARGETS ${tgt}
          EXPORT ElastixTargets
          RUNTIME DESTINATION ${ELASTIX_INSTALL_RUNTIME_DIR}
          LIBRARY DESTINATION ${ELASTIX_INSTALL_LIBRARY_DIR}
          ARCHIVE DESTINATION ${ELASTIX_INSTALL_ARCHIVE_DIR}
          COMPONENT Development
          )
      endif()
    else()
      if(NOT ELASTIX_NO_INSTALL_RUNTIME_LIBRARIES)
        install(TARGETS ${tgt}
          EXPORT ElastixTargets
          RUNTIME DESTINATION ${ELASTIX_INSTALL_RUNTIME_DIR}
          LIBRARY DESTINATION ${ELASTIX_INSTALL_LIBRARY_DIR}
          ARCHIVE DESTINATION ${ELASTIX_INSTALL_ARCHIVE_DIR}
          COMPONENT RuntimeLibraries
          )
      endif()
    endif()
  elseif(type STREQUAL "EXECUTABLE")
    if(NOT ELASTIX_NO_INSTALL_RUNTIME_LIBRARIES)
      install(TARGETS ${tgt}
        EXPORT ElastixTargets
        RUNTIME DESTINATION ${ELASTIX_INSTALL_RUNTIME_DIR}
        LIBRARY DESTINATION ${ELASTIX_INSTALL_LIBRARY_DIR}
        ARCHIVE DESTINATION ${ELASTIX_INSTALL_ARCHIVE_DIR}
        COMPONENT Executables
        )
    endif()
  else()
    if(NOT ELASTIX_NO_INSTALL_DEVELOPMENT)
      install(TARGETS ${tgt}
        EXPORT ElastixTargets
        RUNTIME DESTINATION ${ELASTIX_INSTALL_RUNTIME_DIR}
        LIBRARY DESTINATION ${ELASTIX_INSTALL_LIBRARY_DIR}
        ARCHIVE DESTINATION ${ELASTIX_INSTALL_ARCHIVE_DIR}
        COMPONENT Development
        )
    endif()
  endif()

  export(TARGETS ${tgt}
    APPEND FILE "${elastix_BINARY_DIR}/ElastixTargets.cmake"
    )
endfunction()
