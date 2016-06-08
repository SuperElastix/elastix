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
#ifndef __elxMacro_h
#define __elxMacro_h

/** This include is only used to get rid of a MSVS compiler warning
 * when using std::copy. The warning is like:
 *   'conversion' conversion from 'type1' to 'type2', possible loss of data
 * To solve it ITK recommends to include itkMacro.h before <algorithm> or
 * any other stl header. In elastix we try to make sure that elxIncludes.h
 * is included before anything else, which includes elxMacro.h, which now
 * includes itkWin32Header.h.
 */
#include "itkWin32Header.h"

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
 *     class MyMetric : public MetricBase<TElastix>,
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
 * In this function a template is defined, _classname##_install<VIndex>.
 * It contains the ElastixTypedef<VIndex>, and recursive function DO(cdb).
 * DO installs the component for all defined ElastixTypedefs (so for all
 * supported image types).
 *
 */
#define elxInstallMacro( _classname ) \
  template< ::elx::ComponentDatabase::IndexType VIndex > \
  class _classname##_install \
  { \
public: \
    typedef typename::elx::ElastixTypedef< VIndex >::ElastixType ElastixType; \
    typedef::elx::ComponentDatabase::ComponentDescriptionType    ComponentDescriptionType; \
    static int DO( ::elx::ComponentDatabase * cdb ) \
    { \
      ComponentDescriptionType name  = ::elx::_classname< ElastixType >::elxGetClassNameStatic(); \
      int                      dummy = ::elx::InstallFunctions< ::elx::_classname< ElastixType > >::InstallComponent( name, VIndex, cdb ); \
      if( ::elx::ElastixTypedef< VIndex + 1 >::Defined() ) \
      { return _classname##_install< VIndex + 1 >::DO( cdb ); } \
      return dummy;  \
    } \
  }; \
  template< > \
  class _classname##_install< ::elx::NrOfSupportedImageTypes + 1 > \
  { \
public: \
    typedef::elx::ComponentDatabase::ComponentDescriptionType ComponentDescriptionType; \
    static int DO( ::elx::ComponentDatabase * /** cdb */ ) \
    { return 0; } \
  }; \
  extern "C" int _classname##InstallComponent( \
  ::elx::ComponentDatabase * _cdb ) \
  { \
    int _InstallDummy##_classname = _classname##_install< 1 >::DO( _cdb  ); \
    return _InstallDummy##_classname; \
  } //ignore semicolon

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
#define elxInstallComponentFunctionDeclarationMacro( _classname ) \
  extern "C" int _classname##InstallComponent( \
  ::elx::ComponentDatabase * _cdb )

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
#define elxInstallComponentFunctionCallMacro( _classname ) \
  ret |= _classname##InstallComponent( _cdb )

/**
 * elxPrepareImageTypeSupportMacro
 * To be called once, before the elxSupportImageTypeMacro()
 *
 * IMPORTANT: the macro must be invoked in namespace elastix!
 */

#define elxPrepareImageTypeSupportMacro() \
  template< ::elx::ComponentDatabase::IndexType VIndex > \
  /**unsigned int*/ \
  class ElastixTypedef \
  { \
public: \
    /** In the specialisations of this template class */ \
    /** this typedef will make sense */ \
    typedef::itk::Object                                      ElastixType;  \
    typedef::elx::ComponentDatabase::PixelTypeDescriptionType PixelTypeString; \
    static PixelTypeString fPixelTypeAsString( void ) \
    { return PixelTypeString( "" ); } \
    static PixelTypeString mPixelTypeAsString( void ) \
    { return PixelTypeString( "" ); } \
    static unsigned int fDim( void ) \
    { return 0; } \
    static unsigned int mDim( void ) \
    { return 0; } \
    /** In the specialisations of this template class*/ \
    /** this function will return 'true' */ \
    static bool Defined( void ) \
    { return false; } \
  }

/**
 * Macro for installing support for new ImageTypes.
 * Used in elxSupportedImageTypes.cxx .
 *
 * Example of usage:
 *
 * namespace elastix {
 * elxSupportedImageTypeMacro(unsigned short, 2, float, 3, 1);
 * elxSupportedImageTypeMacro(unsigned short, 3, float, 3, 2);
 * etc.
 * } //end namespace elastix
 *
 * The first line adds support for the following combination of ImageTypes:
 * fixedImage: 2D unsigned short
 * movingImage 3D float
 * The Index (last argument) of this combination of ImageTypes is 1.
 *
 * The Index should not be 0. This value is reserved for errormessages.
 * Besides, duplicate indices are not allowed.
 *
 * IMPORTANT: the macro must be invoked in namespace elastix!
 *
 * Details: the macro adds a class template specialization for the class
 * ElastixTypedef<VIndex>.
 *
 */

#define elxSupportedImageTypeMacro( _fPixelType, _fDim, _mPixelType, _mDim, _VIndex ) \
  template< > \
  class ElastixTypedef< _VIndex > \
  { \
public: \
    typedef::itk::Image< _fPixelType, _fDim >                        FixedImageType; \
    typedef::itk::Image< _mPixelType, _mDim >                        MovingImageType; \
    typedef::elx::ElastixTemplate< FixedImageType, MovingImageType > ElastixType; \
    typedef::elx::ComponentDatabase::PixelTypeDescriptionType        PixelTypeString; \
    static PixelTypeString fPixelTypeAsString( void ) \
    { return PixelTypeString( #_fPixelType ); } \
    static PixelTypeString mPixelTypeAsString( void ) \
    { return PixelTypeString( #_mPixelType ); } \
    static unsigned int fDim( void ) \
    { return _fDim; } \
    static unsigned int mDim( void ) \
    { return _mDim; } \
    static bool Defined( void ) \
    { return true; }  \
  }

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
 * virtual const char * elxGetClassName(void){return _name;}
 *
 * Use this macro in every component that will be installed in the
 * ComponentDatabase, using "elxInstallMacro".
 */
#define elxClassNameMacro( _name ) \
  static const char * elxGetClassNameStatic( void ) { return _name; } \
  virtual const char * elxGetClassName( void ) const { return _name; }

/**
 *  elxout
 *
 *  This macro replaces 'elxout' by 'xl::xout["standard"]'.
 *  This simplifies writing messages to screen and logfile.
 *
 *  NB: for error and warning messages, for writing to the
 *  transformparameterfile etc. do not use elxout, but
 *  xout["{error, warning, etc}"].
 *
 */
#define elxout ::xl::xout[ "standard" ]

/********************************************************************************
 *                    *
 *      Dll export    *
 *                    *
 ********************************************************************************/
#if ( defined( _WIN32 ) || defined( WIN32 ) )
#  ifdef _ELASTIX_BUILD_LIBRARY
#    ifdef _ELASTIX_BUILD_SHARED_LIBRARY
#      define ELASTIXLIB_API __declspec( dllexport )
#    else
#      define ELASTIXLIB_API __declspec( dllimport )
#    endif
#  else
#    define ELASTIXLIB_API __declspec( dllexport )
#  endif
#else
#  if __GNUC__ >= 4
#    define ELASTIXLIB_API __attribute__ ( ( visibility( "default" ) ) )
#  else
#    define ELASTIXLIB_API
#  endif
#endif

#endif // end #ifndef __elxMacro_h
