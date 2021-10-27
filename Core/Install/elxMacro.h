/*=========================================================================
 *
 *  Copyright UMC Utrecht and contributors
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/
#ifndef elxMacro_h
#define elxMacro_h

// Avoid creation of multiple instances of `itk::ImageIOFactoryRegisterManager`, `itk::MeshIOFactoryRegisterManager`,
// and `itk::TransformIOFactoryRegisterManager`.
#undef ITK_IO_FACTORY_REGISTER_MANAGER

/** This include is only used to get rid of a MSVS compiler warning
 * when using std::copy. The warning is like:
 *   'conversion' conversion from 'type1' to 'type2', possible loss of data
 * To solve it ITK recommends to include itkMacro.h before <algorithm> or
 * any other stl header. In elastix we try to make sure that elxIncludes.h
 * is included before anything else, which includes elxMacro.h, which now
 * includes itkWin32Header.h.
 */
#include "itkWin32Header.h"
#include "itkMacro.h"

/**
 * Macro for installing support new components
 * (like a new metric, interpolator, or transform).
 * Must be invoked in the .cxx file of the component,
 * after declaration of the class.
 *
 * Example of usage:
 *
 * \code
 * // elxMyMetric.h //
 * #include "elxMetricBase.h"
 * #include "itkMattesMutualInformationImageToImageMetric.h"
 *
 * namespace elastix {
 *
 *   template <class TElastix>
 *     class ITK_TEMPLATE_EXPORT MyMetric : public MetricBase<TElastix>,
 *      public itk::MattesMutualInformationImageToImageMetric
 *           < MetricBase<TElastix>::FixedImageType
 *             MetricBase<TElastix>::MovingImageType >
 *   {
 *     typedef MyMetric Self;
 *     itkNewMacro( Self ); //needed for the elxInstallMacro
 *     elxClassNameMacro("MattesMutualInformation"); //also needed
 *     .......
 *   };
 *
 * } // end namespace elastix
 * \endcode
 *
 * \code
 * // elxMyMetric.hxx //
 * // Definitions of the methods of the MyMetric class template
 * \endcode
 *
 * \code
 * // elxMyMetric.cxx //
 * #include elxMyMetric.h
 * elxInstallMacro(MyMetric);
 *
 * // CMakeLists.txt //
 * ADD_ELXCOMPONENT( MyMetric
 *   elxMyMetric.h
 *   elxMyMetric.hxx
 *   elxMyMetric.cxx
 *   [<any other source files needed>] )
 * \endcode
 *
 * The class to be installed should inherit from the appropriate base class.
 * (elx::MetricBase, elx::TransformBase etc,) and from a specific itk object.
 *
 * IMPORTANT: only one template argument <class TElastix> is allowed. Not more,
 * not less.
 *
 * Details: a function "int _classname##InstallComponent( _cdb )" is defined.
 * In this function a template is defined, _classname\#\#_install<VIndex>.
 * It contains the ElastixTypedef<VIndex>, and recursive function DO(cdb).
 * DO installs the component for all defined ElastixTypedefs (so for all
 * supported image types).
 *
 */
#define elxInstallMacro(_classname)                                                                                    \
  template <unsigned VIndex>                                                                                           \
  class ITK_TEMPLATE_EXPORT _classname##_install                                                                       \
  {                                                                                                                    \
  public:                                                                                                              \
    static int                                                                                                         \
    DO(::elastix::ComponentDatabase * cdb)                                                                             \
    {                                                                                                                  \
      typedef typename ::elastix::ElastixTypedef<VIndex>::ElastixType ElastixType;                                     \
      const auto name = ::elastix::_classname<ElastixType>::elxGetClassNameStatic();                                   \
      const int  dummy =                                                                                               \
        ::elastix::InstallFunctions<::elastix::_classname<ElastixType>>::InstallComponent(name, VIndex, cdb);          \
      if (::elastix::ElastixTypedef<VIndex + 1>::IsDefined)                                                            \
      {                                                                                                                \
        return _classname##_install<VIndex + 1>::DO(cdb);                                                              \
      }                                                                                                                \
      return dummy;                                                                                                    \
    }                                                                                                                  \
  };                                                                                                                   \
  template <>                                                                                                          \
  class _classname##_install<::elastix::NrOfSupportedImageTypes + 1>                                                   \
  {                                                                                                                    \
  public:                                                                                                              \
    static int                                                                                                         \
    DO(::elastix::ComponentDatabase * /** cdb */)                                                                      \
    {                                                                                                                  \
      return 0;                                                                                                        \
    }                                                                                                                  \
  };                                                                                                                   \
  extern "C" int _classname##InstallComponent(::elastix::ComponentDatabase * _cdb)                                     \
  {                                                                                                                    \
    int _InstallDummy##_classname = _classname##_install<1>::DO(_cdb);                                                 \
    return _InstallDummy##_classname;                                                                                  \
  } // ignore semicolon

/**
 * elxInstallComponentFunctionDeclarationMacro
 *
 * Usage example:
 *   elxInstallComponentFunctionDeclarationMacro( BSplineTransform );
 *
 * This macro declares the function implemented by
 * the elxInstallMacro. This macro is used by the
 * CMake generated file
 * elxInstallComponentFunctionDeclarations.h
 * only.
 *
 * Details: the declaration of InstallComponent function defined
 * by elxInstallMacro is simply repeated.
 *
 * See also elxInstallAllComponents.h.
 */
#define elxInstallComponentFunctionDeclarationMacro(_classname)                                                        \
  extern "C" int _classname##InstallComponent(::elastix::ComponentDatabase * _cdb)

/**
 * elxInstallComponentFunctionCallMacro
 *
 * Usage example:
 *   elxInstallComponentFunctionCallMacro( BSplineTransform );
 *
 * This macro calls the function implemented by
 * the elxInstallMacro. This macro is used by the
 * CMake generated file
 * elxInstallComponentFunctionCalls.h
 * only.
 *
 * Details: the InstallComponent function defined
 * by elxInstallMacro is called.
 *
 * See also elxInstallAllComponents.h.
 */
#define elxInstallComponentFunctionCallMacro(_classname) ret |= _classname##InstallComponent(_cdb)


/**
 * elxClassNameMacro(_name)
 *
 * Example of usage:
 *
 * class MyMetric
 * {
 * public:
 *   elxClassNameMacro("MyFirstMetric");
 * }
 *
 * This macro defines two functions.
 *
 * static const char * elxGetClassNameStatic(void){return _name;}
 * const char * elxGetClassName( void ) const override { return _name; }
 *
 * Use this macro in every component that will be installed in the
 * ComponentDatabase, using "elxInstallMacro".
 */
#define elxClassNameMacro(_name)                                                                                       \
  static const char * elxGetClassNameStatic(void) { return _name; }                                                    \
  const char *        elxGetClassName(void) const override { return _name; }

/** Declares a pair of pure virtual member functions (overloaded for const
 * and non-const) to get a reference to itself, of the specified type.
 */
#define elxDeclarePureVirtualGetSelfMacro(type)                                                                        \
  virtual const type & GetSelf(void) const override = 0;                                                               \
  virtual type &       GetSelf(void) override = 0

/** Defines a pair of overrides of GetSelf(void) (overloaded for const and
 * non-const), which return a reference to itself. Declares a deleted static
 * member function overload, just to allow macro calls to end with a semicolon.
 */
#define elxOverrideGetSelfMacro                                                                                        \
  auto GetSelf(void) const->decltype(*this) override { return *this; }                                                 \
  auto GetSelf(void)->decltype(*this) override { return *this; }                                                       \
  static void                         GetSelf(const void *) = delete


/**
 *  elxout
 *
 *  This macro replaces 'elxout' by 'xl::xout["standard"]'.
 *  This simplifies writing messages to screen and logfile.
 *
 *  NB: for error and warning messages, for writing to the
 *  transformparameterfile etc. do not use elxout, but
 *  xl::xout["{error, warning, etc}"].
 *
 */
#define elxout ::xl::xout["standard"]

/********************************************************************************
 *                    *
 *      Dll export    *
 *                    *
 ********************************************************************************/

#if (defined(_WIN32) || defined(WIN32))
#  define ELASTIXLIB_API
#else
#  if (__GNUC__ >= 4 || defined(__clang__))
#    define ELASTIXLIB_API __attribute__((visibility("default")))
#  else
#    define ELASTIXLIB_API
#  endif
#endif

#endif // end #ifndef elxMacro_h
