#ifndef __elxElastixMain_h
#define __elxElastixMain_h

#include "elxComponentDatabase.h"
#include "elxComponentLoader.h"
//#include "elxSupportedImageTypes.h"

#include "elxBaseComponent.h"
#include "elxElastixBase.h"
#include "itkObject.h"

#include <iostream>
#include <fstream>



namespace elastix 
{
	using namespace itk;

	/// 
	/// ********************** Global Functions **********************
	/// 
	/// NB: not part of the ElastixMain class.
	///

	/** Configure the xl::xout variable, which has to be used for 
	 * for writing messages. The function adds some default fields,
	 * such as "warning", "error", "standard", and it sets the outputs
	 * to std::cout and a logfile.
	 * 
	 * The method takes a logfile name as its input argument.
	 * It returns 0 if everything went ok. 1 otherwise.
	 */
	extern int xoutSetup(const char * logfilename);

	/**
	 * \class ElastixMain
	 * \brief ???
	 *
	 * The ElastixMain class ....
	 *
	 * \ingroup Kernel
	 */

	class ElastixMain : public Object
	{
	public:
		
		/** Standard itk.*/
		typedef ElastixMain								Self;
		typedef Object										Superclass;
		typedef SmartPointer<Self>				Pointer;
		typedef SmartPointer<const Self>  ConstPointer;
		
		/** Method for creation through the object factory.*/
		itkNewMacro( Self );

		/** Run-time type information (and related methods).*/
		itkTypeMacro( ElastixMain, Object );

		/** Typedef's.*/

		/** itk base objects.*/
		typedef Object									ObjectType;
		typedef ObjectType::Pointer			ObjectPointer;
		typedef DataObject							DataObjectType;
		typedef DataObjectType::Pointer	DataObjectPointer;

		/** Elastix components.*/
		typedef ElastixBase																			ElastixBaseType;
		typedef ElastixBase::ConfigurationType									ConfigurationType;
		typedef ConfigurationType::ArgumentMapType							ArgumentMapType;
		typedef ConfigurationType::Pointer											ConfigurationPointer;
		typedef ComponentDatabase																ComponentDatabaseType;
		typedef ComponentDatabaseType::Pointer									ComponentDatabasePointer;
		typedef ComponentDatabaseType::PtrToCreator							PtrToCreator;
		typedef ComponentDatabaseType::ComponentDescriptionType	ComponentDescriptionType;
		typedef ComponentDatabaseType::PixelTypeDescriptionType	PixelTypeDescriptionType;
		typedef ComponentDatabaseType::ImageDimensionType				ImageDimensionType;
		typedef ComponentDatabaseType::IndexType								DBIndexType;

		typedef ComponentLoader																	ComponentLoaderType;
		typedef ComponentLoaderType::Pointer										ComponentLoaderPointer;
		
		/** Set/Get functions for the description of the imagetype.*/
		itkSetMacro( FixedImagePixelType,		PixelTypeDescriptionType );
		itkSetMacro( MovingImagePixelType,	PixelTypeDescriptionType );
		itkSetMacro( FixedImageDimension,		ImageDimensionType );
		itkSetMacro( MovingImageDimension,	ImageDimensionType );
		itkGetMacro( FixedImagePixelType,		PixelTypeDescriptionType );
		itkGetMacro( MovingImagePixelType,	PixelTypeDescriptionType );
		itkGetMacro( FixedImageDimension,		ImageDimensionType );
		itkGetMacro( MovingImageDimension,	ImageDimensionType );

		/**
		 * Set/Get functions for the fixed and moving images
		 * (if these are not used, elastix tries to read them from disk,
		 * according to the commandline parameters).
		 */
		itkSetObjectMacro( FixedImage,	DataObjectType );
		itkSetObjectMacro( MovingImage,	DataObjectType );
		itkGetObjectMacro( FixedImage,	DataObjectType );
		itkGetObjectMacro( MovingImage,	DataObjectType );

		itkSetObjectMacro( FixedInternalImage,	DataObjectType );
		itkSetObjectMacro( MovingInternalImage,	DataObjectType );
		itkGetObjectMacro( FixedInternalImage,	DataObjectType );
		itkGetObjectMacro( MovingInternalImage,	DataObjectType );

		/**
		 * Functions to get pointers to the elastix components. 
		 * The components are returned as Object::Pointer.
		 * Before calling this functions, call run().
		 */
		itkGetObjectMacro( Elastix,								ObjectType );
		itkGetObjectMacro( FixedImagePyramid,			ObjectType );
		itkGetObjectMacro( MovingImagePyramid,		ObjectType );
		itkGetObjectMacro( Interpolator,					ObjectType );
		itkGetObjectMacro( Metric,								ObjectType );
		itkGetObjectMacro( Optimizer,							ObjectType );
		itkGetObjectMacro( Registration,					ObjectType );
		itkGetObjectMacro( Resampler,							ObjectType );
		itkGetObjectMacro( ResampleInterpolator,	ObjectType );
		itkGetObjectMacro( Transform,							ObjectType );

		/** Set/Get the configuration object.*/
		itkSetObjectMacro( Configuration, ConfigurationType );
		itkGetObjectMacro( Configuration, ConfigurationType );

		/** Set/Get the initial transform
		 *
		 * the type is ObjectType, but the pointer should actually point 
		 * to an itk::Transform type (or inherited from that one).
		 */
		itkSetObjectMacro( InitialTransform, ObjectType );
		itkGetObjectMacro( InitialTransform, ObjectType );

		/** Get and Set the elastix-level.*/
		void SetElastixLevel( unsigned int level );
		unsigned int GetElastixLevel(void);

		/** Returns the Index that is used in elx::ComponentDatabase.*/
		itkGetConstMacro( DBIndex, DBIndexType );

		
		/** Enter the command line parameters, which were given by the user,
		 * if elastix.exe is used to do a registration.	
		 * The Configuration object will be initialized in this way.
		 */
		virtual void EnterCommandLineArguments( ArgumentMapType & argmap );

		/** Start the registration
		 * run() without commandline parameters; it assumes that 
		 * EnterCommandLineParameters has been invoked already, or that
		 * m_Configuration is initialised in a different way.
		 */
		virtual int Run(void);

		/** Start the registration
		 * this version of 'run' first calls this->EnterCommandLineParameters(argc,argv)
		 * and then calls run().
		 */
		virtual int Run( ArgumentMapType & argmap );

		/** 
		 * Functions to get/set the ComponentDatabase
		 */
		static ComponentDatabase * GetComponentDatabase(void)
		{
			return s_CDB.GetPointer();
		}

		static void SetComponentDatabase(ComponentDatabase * arg)
		{
			if ( s_CDB != arg )
			{
				s_CDB = arg;
			}
		}

		static void UnloadComponents(void);

	protected:

		ElastixMain();
		virtual ~ElastixMain();

		/** A pointer to elastix as an itk::object. In run() this
		 * pointer will be assigned to an ElastixTemplate<>.
		 */
		ObjectPointer m_Elastix;

		/** The same pointer, but casted to an ElastixBaseType
		 * (from which all ElastixTemplates should inherit).
		 */
		ElastixBaseType *			m_elx_Elastix;
		ConfigurationPointer	m_Configuration;

		/** Description of the ImageTypes.*/
		PixelTypeDescriptionType		m_FixedImagePixelType;
		ImageDimensionType					m_FixedImageDimension;
		PixelTypeDescriptionType		m_MovingImagePixelType;
		ImageDimensionType					m_MovingImageDimension;

		DBIndexType									m_DBIndex;

		DataObjectPointer						m_FixedImage;
		DataObjectPointer						m_MovingImage;
		DataObjectPointer						m_FixedInternalImage;
		DataObjectPointer						m_MovingInternalImage;

		ObjectPointer	m_FixedImagePyramid;
		ObjectPointer m_MovingImagePyramid;
		ObjectPointer m_Interpolator;
		ObjectPointer m_Metric;
		ObjectPointer m_Optimizer;
		ObjectPointer m_Registration;
		ObjectPointer m_Resampler;
		ObjectPointer m_ResampleInterpolator;
		ObjectPointer m_Transform;

		/** The initial transform.*/
		ObjectPointer m_InitialTransform;

		static ComponentDatabasePointer s_CDB;
		static ComponentLoaderPointer s_ComponentLoader;
		virtual int LoadComponents(void);
		

		/**
		 * InitDBIndex sets m_DBIndex to the value obtained from the
		 * ComponentDatabase.
		 */
    virtual int InitDBIndex(void);

	private:

		ElastixMain( const Self& );			// purposely not implemented
		void operator=( const Self& );	// purposely not implemented

	}; // end class ElastixMain
	
	
	
} // end namespace elastix


#endif // end #ifndef __elxElastixMain_h

