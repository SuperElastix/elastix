/**
 * This file contains the declaration of the elx::ElastixBase class. 
 * elx::ElastixTemplate<> inherits from this class. It is an abstract class,
 * since it contains pure virtual functions (which must be implemented
 * in ElastixTemplate<>).
 * 
 * The Configuration object is stored in this class.
 */

#ifndef __elxElastixBase_h
#define __elxElastixBase_h

#include "elxBaseComponent.h"
#include "elxComponentDatabase.h"
#include "elxConfigurationToUse.h"
#include "itkObject.h"
#include "itkDataObject.h"
#include "elxMacro.h"
#include "xoutmain.h"

#include <fstream>
#include <iomanip>


namespace elastix
{
  using namespace itk;	
	
	/**
	 * \class ElastixBase
	 * \brief ???
	 *
	 * The ElastixBase class ....
	 *
	 * \ingroup Kernel
	 */

	class ElastixBase : public BaseComponent
	{
	public:

		/** Standard typedefs etc. */
		typedef ElastixBase				Self;
		typedef BaseComponent			Superclass;
	
		/** Typedefs used in this class */
		typedef MyConfiguration							ConfigurationType;
		typedef ConfigurationType::Pointer	ConfigurationPointer;
		typedef itk::Object									ObjectType; //for the components
		typedef itk::DataObject							DataObjectType; //for the images

		/** Other typedef's.*/
		typedef ComponentDatabase																ComponentDatabaseType;
		typedef ComponentDatabaseType::Pointer									ComponentDatabasePointer;
		typedef ComponentDatabaseType::IndexType								DBIndexType;

		/** Set/Get the Configuration Object.
		 *
		 * The Set-functions cannot be defined with the itkSetObjectMacro,
		 * since this class does not derive from itk::Object and 
		 * thus does not have a ::Modified() method.
		 *
		 * This method checks if this instance of the class can be casted
		 * (dynamically) to an itk::Object. If yes, it calls ::Modified().
		 */
		virtual void SetConfiguration( ConfigurationType * _arg );
		virtual ConfigurationType * GetConfiguration(void)
		{
			return this->m_Configuration.GetPointer();
		}

		/** Set the database index of the instantiated elastix object */
		virtual void SetDBIndex( DBIndexType _arg );
		virtual DBIndexType GetDBIndex(void)
		{
			return this->m_DBIndex;
		}

		/** 
		 * Functions to get/set the ComponentDatabase
		 */
		virtual ComponentDatabase * GetComponentDatabase(void)
		{
			return this->m_CDB.GetPointer();
		}

		virtual void SetComponentDatabase(ComponentDatabase * arg)
		{
			if ( this->m_CDB != arg )
			{
				this->m_CDB = arg;
			}
		}


		/**
		 * Pure virtual functions for setting/getting the components
		 * of a registration method. Declaring these functions here
		 * ensures that they can be used in ElastixMain. Implementation
		 * must be done in ElastixTemplate<>.
		 */
		virtual void SetRegistration( ObjectType * _arg ) = 0;
		virtual void SetFixedImagePyramid( ObjectType * _arg ) = 0;
		virtual void SetMovingImagePyramid( ObjectType * _arg ) = 0;
		virtual void SetInterpolator( ObjectType * _arg ) = 0;
		virtual void SetMetric( ObjectType * _arg ) = 0;
		virtual void SetOptimizer( ObjectType * _arg ) = 0;
		virtual void SetResampler( ObjectType * _arg ) = 0;
		virtual void SetResampleInterpolator( ObjectType * _arg ) = 0;
		virtual void SetTransform( ObjectType * _arg ) = 0;
		
		virtual void SetFixedImage( DataObjectType * _arg ) = 0;
		virtual void SetMovingImage( DataObjectType * _arg ) = 0;
		virtual void SetFixedInternalImage( DataObjectType * _arg ) = 0;
		virtual void SetMovingInternalImage( DataObjectType * _arg ) = 0;

		virtual ObjectType * GetRegistration(void) = 0;
		virtual ObjectType * GetFixedImagePyramid(void) = 0;
		virtual ObjectType * GetMovingImagePyramid(void) = 0;
		virtual ObjectType * GetInterpolator(void) = 0;
		virtual ObjectType * GetMetric(void) = 0;
		virtual ObjectType * GetOptimizer(void) = 0;
		virtual ObjectType * GetResampler(void) = 0;
		virtual ObjectType * GetResampleInterpolator(void) = 0;
		virtual ObjectType * GetTransform(void) = 0;

		virtual DataObjectType * GetFixedImage(void) = 0;
		virtual DataObjectType * GetMovingImage(void) = 0;
		virtual DataObjectType * GetFixedInternalImage(void) = 0;
		virtual DataObjectType * GetMovingInternalImage(void) = 0;

		virtual ObjectType * GetInitialTransform(void) = 0;
		virtual void SetInitialTransform( ObjectType * _arg ) = 0;

		/** Empty Run()-function to be overridden.*/
		virtual int Run(void) = 0;

		/** Empty ApplyTransform()-function to be overridden.*/
		virtual int ApplyTransform(void) = 0;


		/** Function that is called at the very beginning of ElastixTemplate::Run().
		 * It checks the command line input arguments
		 */
		virtual int BeforeAllBase(void);
		virtual void BeforeRegistrationBase(void);
		virtual void AfterRegistrationBase(void);

		/** Get the default precision of xout.
		 * (The value assumed when no DefaultOutputPrecision is given in the 
		 * parameter file 
		 */
		virtual int GetDefaultOutputPrecision(void)
		{
			return this->m_DefaultOutputPrecision;
		}
		
		
	protected:
		
		ElastixBase();
		virtual ~ElastixBase() {};
		
		ConfigurationPointer	m_Configuration;
		DBIndexType						m_DBIndex;
		ComponentDatabasePointer m_CDB;

	private:

		ElastixBase( const Self& );			// purposely not implemented
		void operator=( const Self& );	// purposely not implemented

		xl::xoutrow_type			m_IterationInfo;	
		
		int m_DefaultOutputPrecision;

	};  // end class ElastixBase


} // end namespace elastix


#endif // end #ifndef __elxElastixBase_h

