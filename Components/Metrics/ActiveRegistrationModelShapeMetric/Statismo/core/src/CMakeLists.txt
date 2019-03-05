add_library( statismo_core STATIC ModelInfo.cxx)
target_link_libraries( statismo_core ${HDF5_LIBRARIES} ${Boost_LIBRARIES} )

set_target_properties( statismo_core PROPERTIES
  DEBUG_POSTFIX "-d"
  VERSION "${statismo_LIB_VERSION}"
  SOVERSION "${statismo_LIB_SOVERSION}"
)
