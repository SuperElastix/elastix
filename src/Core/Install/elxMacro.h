#ifndef __elxMacro_h
#define __elxMacro_h

/** Some header files that may be needed for the macros to function correctly */

//#include "elxMacro.h"
//#include "elxInstallFunctions.h"
//#include "elxComponentDatabase.h"
//#include "elxBaseComponent.h"
//#include "elxElastixTemplate.h"
//#include "itkImage.h"
//#include "elxSupportedImageTypes.h"

/** 
 * Macro for creating function in .DLL's
 * 
 * Windows needs __declspec stuff. Unix does not need anything.
 *
 * usage: see elxInstallMacro for an example.
 */
#ifdef _WIN32
#	define __ELX_DLLEXPORT __declspec(dllexport)
#else
#	define __ELX_DLLEXPORT
#endif


/**
* Macro for installing support new components (like a new metric, interpolator,
* or transform).
* Must be invoked in the .cxx file of the component, after declaration of the 
* class.
*
* Example of usage:
*
* // elxMyMetric.h //
* #include "elxMetricBase.h"
* #include "itkMattesMutualInformationImageToImageMetric.h"
*
* namespace elastix {
*
*		template <class TElastix>
*			class MyMetric : public MetricBase<TElastix>,
*			 public itk::MattesMutualInformationImageToImageMetric
*						< MetricBase<TElastix>::FixedImageType
*							MetricBase<TElastix>::MovingImageType > 
*		{
*			typedef MyMetric Self;
*			itkNewMacro( Self ); //needed for the elxInstallMacro
*			elxClassNameMacro("MattesMutualInformation"); //also needed
*			.......
*		};
*
* } // end namespace elastix
*
* // elxMyMetric.hxx //
* definitions of the methods of the MyMetric class template
*
* // elxMyMetric.cxx //
* #include elxMyMetric.h
*	elxInstallMacro(MyMetric);
*	
* 
* The class to be installed should inherit from the appropriate base class.
* (elx::MetricBase, elx::TransformBase etc,) and from a specific itk object.
* 
* IMPORTANT: only one template argument <class TElastix> is allowed. Not more, 
* not less.
*
* Details: a function "int InstallComponent(_cdb, _xout)" is defined.
* In this function a template is defined, _classname##_install<VIndex>.
* It contains the ElastixTypedef<VIndex>, and recursive function DO(cdb).
* DO installs the component for all defined ElastixTypedefs (so for all
* supported image types).
* Additionally xout is prepared for use by calling set_xout(_xout).
*
*/
#define elxInstallMacro(_classname) \
	template < ::elx::ComponentDatabase::IndexType VIndex> \
		class _classname##_install \
	{ \
	public: \
		typedef typename ::elx::ElastixTypedef<VIndex>::ElastixType ElastixType; \
		typedef ::elx::ComponentDatabase::ComponentDescriptionType ComponentDescriptionType; \
		static int DO(::elx::ComponentDatabase * cdb) \
		{ \
		ComponentDescriptionType name = ::elx:: _classname <ElastixType>::elxGetClassNameStatic(); \
		int dummy = ::elx::InstallFunctions< ::elx:: _classname <ElastixType> >::InstallComponent(name, VIndex, cdb); \
			if ( ::elx::ElastixTypedef<VIndex+1>::Defined() ) \
			{	return _classname##_install<VIndex+1>::DO( cdb ); } \
			return dummy;  \
		} \
	}; \
	template <> \
		class _classname##_install < ::elx::NrOfSupportedImageTypes+1 > \
	{ \
	public: \
		typedef ::elx::ComponentDatabase::ComponentDescriptionType ComponentDescriptionType; \
		static int DO(::elx::ComponentDatabase * cdb) \
		{ return 0; } \
	}; \
	extern "C" __ELX_DLLEXPORT int InstallComponent( \
		::elx::ComponentDatabase * _cdb, \
		::xl::xoutbase_type * _xout ) \
	{ \
		int _InstallDummy##_classname = _classname##_install<1>::DO( _cdb  ); \
		::xl::set_xout( _xout ); \
		return _InstallDummy##_classname ; \
	}//ignore semicolon



/**
* elxPrepareImageTypeSupportMacro
* To be called once, before the elxSupportImageTypeMacro()
*
* IMPORTANT: the macro must be invoked in namespace elastix!
*/

#define elxPrepareImageTypeSupportMacro() \
	template < ::elx::ComponentDatabase::IndexType VIndex >  /**unsigned int*/ \
		class ElastixTypedef \
	{ \
	public: \
		/** In the specialisations of this template class */ \
		/** this typedef will make sense */ \
		typedef ::itk::Object ElastixType;  \
		typedef ::elx::ComponentDatabase::PixelTypeDescriptionType PixelTypeString; \
		static PixelTypeString fPixelTypeAsString(void) \
			{ return PixelTypeString("");} \
		static PixelTypeString mPixelTypeAsString(void) \
			{ return PixelTypeString("");} \
		static unsigned int fDim(void) \
			{ return 0;} \
		static unsigned int mDim(void) \
			{ return 0;} \
		/** In the specialisations of this template class*/ \
		/** this function will return 'true' */ \
		static bool Defined(void) \
			{ return false;} \
	} 
	
		
/**
* Macro for installing support for new ImageTypes.
* Used in elxSupportedImageTypes.cxx .
*
* Example of usage:
*
* namespace elastix {
* elxSupportedImageTypeMacro(unsigned short, 2, float, 3, 1);
*	elxSupportedImageTypeMacro(unsigned short, 3, float, 3, 2);
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

#define elxSupportedImageTypeMacro(_fPixelType,_fDim,_mPixelType,_mDim,_VIndex) \
	template<> \
		class ElastixTypedef < _VIndex > \
	{ \
	public: \
		typedef ::itk::Image< _fPixelType , _fDim > FixedImageType; \
		typedef ::itk::Image< _mPixelType , _mDim > MovingImageType; \
		typedef ::elx::ElastixTemplate< FixedImageType, MovingImageType > ElastixType; \
		typedef ::elx::ComponentDatabase::PixelTypeDescriptionType PixelTypeString; \
		static PixelTypeString fPixelTypeAsString(void) \
			{ return PixelTypeString( #_fPixelType );} \
		static PixelTypeString mPixelTypeAsString(void) \
			{ return PixelTypeString( #_mPixelType );} \
		static unsigned int fDim(void) \
			{ return _fDim ;} \
		static unsigned int mDim(void) \
			{ return _mDim ;} \
		static bool Defined(void) \
		{	return true; }  \
	} 





/**
 * elxClassNameMacro(_name)
 *
 * Example of usage:
 * 
 * class MyMetric
 * {
 * public:
 *	 elxClassNameMacro("MyFirstMetric");
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
#define elxClassNameMacro(_name) \
static const char * elxGetClassNameStatic(void) {return _name ;} \
virtual const char * elxGetClassName(void) {return _name ;}


/**
 *	elxout
 *
 *	This macro replaces 'elxout' by '::xl::xout["standard"]'. 
 *	This simplifies writing messages to screen and logfile.
 *  
 *	NB: for error and warning messages, for writing to the 
 *	transformparameterfile etc. do not use elxout, but 
 *	xout["{error, warning, etc}"].
 *  
 */
#define elxout ::xl::xout["standard"]


#endif // end #ifndef __elxMacro_h

