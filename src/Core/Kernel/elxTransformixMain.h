#ifndef __elxTransformixMain_H_
#define __elxTransformixMain_H_

#include "elxElastixMain.h"

namespace elastix 
{
	using namespace itk;
	
	/**
	 * \class TransformixMain
	 * \brief ???
	 *
	 * The TransformixMain class inherits from ElastixMain. We overwrite the Run()
	 * -function. In the new Run() the Run()-function from the
	 * ElastixTemplate-class is not called (as in elxElastixMain.cxx),
	 * because this time we don't want to start a registration, but
	 * just apply a transformation to an input image.
	 *
	 * \ingroup Kernel
	 */
	
	class TransformixMain : public ElastixMain
	{
	public:
		
		/** Standard itk.*/
		typedef TransformixMain						Self;
		typedef ElastixMain								Superclass;
		typedef SmartPointer<Self>				Pointer;
		typedef SmartPointer<const Self>  ConstPointer;
		
		/** Method for creation through the object factory.*/
		itkNewMacro( Self);
		
		/** Run-time type information (and related methods).*/
		itkTypeMacro( TransformixMain, ElastixMain );
		
		/** Typedef's from Superclass.*/
		
		/** typedef's from itk base Object.*/
		typedef Superclass::ObjectType					ObjectType;
		typedef Superclass::ObjectPointer				ObjectPointer;
		typedef Superclass::DataObjectType			DataObjectType;
		typedef Superclass::DataObjectPointer		DataObjectPointer;
		
		/** Typedef's from Elastix components.*/
		typedef Superclass::ElastixBaseType							ElastixBaseType;
		typedef Superclass::ConfigurationType						ConfigurationType;
		typedef Superclass::ArgumentMapType							ArgumentMapType;
		typedef Superclass::ConfigurationPointer				ConfigurationPointer;
		typedef Superclass::ComponentDatabaseType				ComponentDatabaseType;
		typedef Superclass::PtrToCreator								PtrToCreator;
		typedef Superclass::ComponentDescriptionType		ComponentDescriptionType;
		typedef Superclass::PixelTypeDescriptionType		PixelTypeDescriptionType;
		typedef Superclass::ImageDimensionType					ImageDimensionType;
		typedef Superclass::DBIndexType									DBIndexType;
		
		typedef Superclass::ComponentLoaderType					ComponentLoaderType;
		typedef Superclass::ComponentLoaderPointer			ComponentLoaderPointer;
		
		
		/** Overwrite Run() from base-class.*/
		virtual int Run(void);
		
		/** Overwrite Run( argmap ) from base-class.*/
		virtual int Run( ArgumentMapType & argmap );
		
		/** Get and Set input- and outputImage.*/
		virtual void SetInputImage( DataObjectType * inputImage );
		
	protected:
		
		TransformixMain();
		virtual ~TransformixMain();
		
		/** InitDBIndex sets m_DBIndex to the value obtained
		 * from the ComponentDatabase.
		 */
		virtual int InitDBIndex(void);
		
	private:
		
		TransformixMain( const Self& );	// purposely not implemented
		void operator=( const Self& );	// purposely not implemented
		
	}; // end class TransformixMain
	
	
} // end namespace elastix


#endif // end #ifndef __elxTransformixMain_h

