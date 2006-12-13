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
	 * \brief A class with all functionality to configure elastix.
	 *
	 * The ElastixMain initializes the MyConfiguration class with the
	 * parameters and commandline arguments. After this, the class loads
	 * and creates all components and sets them in ElastixTemplate.
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
    typedef ElastixBase::ObjectContainerType                ObjectContainerType;
    typedef ElastixBase::DataObjectContainerType            DataObjectContainerType;
    typedef ElastixBase::ObjectContainerPointer             ObjectContainerPointer;
    typedef ElastixBase::DataObjectContainerPointer         DataObjectContainerPointer;

    /** Typedefs for the database that holds pointers to ::New() functions.
     * Those functions are used to instantiate components, such as the metric etc. */
		typedef ComponentDatabase																ComponentDatabaseType;
		typedef ComponentDatabaseType::Pointer									ComponentDatabasePointer;
		typedef ComponentDatabaseType::PtrToCreator							PtrToCreator;
		typedef ComponentDatabaseType::ComponentDescriptionType	ComponentDescriptionType;
		typedef ComponentDatabaseType::PixelTypeDescriptionType	PixelTypeDescriptionType;
		typedef ComponentDatabaseType::ImageDimensionType				ImageDimensionType;
		typedef ComponentDatabaseType::IndexType								DBIndexType;
		
    /** Typedef for class that populates a ComponentDatabase */
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

		/** Set/Get functions for the fixed and moving images
		 * (if these are not used, elastix tries to read them from disk,
		 * according to the commandline parameters). */
		itkSetObjectMacro( FixedImageContainer,	 DataObjectContainerType );
		itkSetObjectMacro( MovingImageContainer, DataObjectContainerType );
		itkGetObjectMacro( FixedImageContainer,	 DataObjectContainerType );
		itkGetObjectMacro( MovingImageContainer, DataObjectContainerType );

    /** Set/Get functions for the fixed and moving masks
		 * (if these are not used, elastix tries to read them from disk,
		 * according to the commandline parameters). */
		itkSetObjectMacro( FixedMaskContainer, DataObjectContainerType );
		itkSetObjectMacro( MovingMaskContainer, DataObjectContainerType );
		itkGetObjectMacro( FixedMaskContainer, DataObjectContainerType );
		itkGetObjectMacro( MovingMaskContainer, DataObjectContainerType );

    /** Set/Get the configuration object.*/
		itkSetObjectMacro( Configuration, ConfigurationType );
		itkGetObjectMacro( Configuration, ConfigurationType );

		/** Functions to get pointers to the elastix components. 
		 * The components are returned as Object::Pointer.
		 * Before calling this functions, call run(). */
		itkGetObjectMacro( Elastix,	ObjectType );

    /** Convenience function that returns the Elastix component as
     * a pointer to an ElastixBaseType. Use only after having called run()!  */
    virtual ElastixBaseType * GetElastixBase(void) const;

    /** Get the final transform (the result of running elastix).
     * You may pass this as an InitialTransform in an other instantiation
     * of ElastixMain.
     * Only valid after calling Run()!  */
		itkGetObjectMacro( FinalTransform, ObjectType );
		
		/** Set/Get the initial transform
		 * the type is ObjectType, but the pointer should actually point 
		 * to an itk::Transform type (or inherited from that one). */
		itkSetObjectMacro( InitialTransform, ObjectType );
		itkGetObjectMacro( InitialTransform, ObjectType );

		/** Get and Set the elastix-level.*/
		void SetElastixLevel( unsigned int level );
		unsigned int GetElastixLevel(void);

		/** Returns the Index that is used in elx::ComponentDatabase.*/
		itkGetConstMacro( DBIndex, DBIndexType );
		
		/** Enter the command line parameters, which were given by the user,
		 * if elastix.exe is used to do a registration.	
		 * The Configuration object will be initialized in this way. */
		virtual void EnterCommandLineArguments( ArgumentMapType & argmap );

		/** Start the registration
		 * run() without commandline parameters; it assumes that 
		 * EnterCommandLineParameters has been invoked already, or that
		 * m_Configuration is initialised in a different way. */
		virtual int Run(void);

		/** Start the registration
		 * this version of 'run' first calls this->EnterCommandLineParameters(argc,argv)
		 * and then calls run(). */
		virtual int Run( ArgumentMapType & argmap );

		/** Functions to get/set the ComponentDatabase */
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
		 * pointer will be assigned to an ElastixTemplate<>. */
		ObjectPointer m_Elastix;

    /** The configuratio object, containing the parameters and command-line arguments */
		ConfigurationPointer	m_Configuration;

		/** Description of the ImageTypes.*/
		PixelTypeDescriptionType		m_FixedImagePixelType;
		ImageDimensionType					m_FixedImageDimension;
		PixelTypeDescriptionType		m_MovingImagePixelType;
		ImageDimensionType					m_MovingImageDimension;

		DBIndexType									m_DBIndex;

    /** The images and masks */
		DataObjectContainerPointer	m_FixedImageContainer;
		DataObjectContainerPointer	m_MovingImageContainer;
    DataObjectContainerPointer	m_FixedMaskContainer;
		DataObjectContainerPointer	m_MovingMaskContainer;

    /** A transform that is the result of registration. */
		ObjectPointer m_FinalTransform;

		/** The initial transform.*/
		ObjectPointer m_InitialTransform;

		static ComponentDatabasePointer s_CDB;
		static ComponentLoaderPointer s_ComponentLoader;
		virtual int LoadComponents(void);
		
		/** InitDBIndex sets m_DBIndex by asking the ImageTypes
     * from the Configuration object and obtaining the corresponding
     * DB index from the ComponentDatabase.	*/
    virtual int InitDBIndex(void);

    /** Create a component. Make sure InitDBIndex has been called before.
     * The input is a string, like MattesMutualInformation */
    virtual ObjectPointer CreateComponent( const ComponentDescriptionType & name );

    /** Create components. Reads from the configuration object (using the provided key)
     * the names of the components to create and store their instantations in the 
     * provided ObjectContainer.
     * The errorcode remains what it was if no error occured. Otherwise it's set to 1.
     * The 'key' is the entry inspected in the parameter file
     * A component named 'defaultComponentName' is used when the key is not found
     * in the parameter file.  */
    virtual ObjectContainerPointer CreateComponents(
      const ComponentDescriptionType & key,
      const ComponentDescriptionType & defaultComponentName,
      int & errorcode );

	private:

		ElastixMain( const Self& );			// purposely not implemented
		void operator=( const Self& );	// purposely not implemented

	}; // end class ElastixMain
	
	
	
} // end namespace elastix


#endif // end #ifndef __elxElastixMain_h

