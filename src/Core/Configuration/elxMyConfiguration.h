/**
 * This header file declares the MyConfiguration class
 *	
 *	In elxConfigurationToUse.h this file can be included if this version
 * of elx::MyConfiguration is desired.
 */

#ifndef	__elxMyConfiguration_H__
#define __elxMyConfiguration_H__

#include "itkObject.h"
#include "itkObjectFactory.h"

#include "elxConfigurationBase.h"
#include "xoutmain.h"

#include <fstream>
#include <iostream>

#include "param.h"
#include <map>


namespace elastix
{
	using namespace itk;

	/**
	 * \class MyConfiguration
	 * \brief ???
	 *
	 * The MyConfiguration class ....
	 *
	 * \ingroup Configuration
	 */

	class MyConfiguration : public Object, public ConfigurationBase
	{
	public:

		/** Standard itk.*/
		typedef MyConfiguration						Self;
		typedef Object										Superclass1;
		typedef ConfigurationBase					Superclass2;
		typedef SmartPointer<Self>				Pointer;
		typedef SmartPointer<const Self>  ConstPointer;

		/** Typedef's for map's.*/
		typedef std::map<std::string, std::string>	ArgumentMapType;
		typedef ArgumentMapType::value_type					EntryType;
		
		/** Method for creation through the object factory. */
		itkNewMacro( Self );
		
		/** Standard part of all itk objects. */
		itkTypeMacro( MyConfiguration, Object );
		
		/** Typedef's.*/
		typedef VPF::ParameterFile		ParameterFileType;
		
		/** .*/
		virtual int Initialize(ArgumentMapType & _arg);

		virtual bool Initialized(void); //to elxconfigurationbase

		/** Get and Set CommandLine arguments into the argument map.*/
		const char * GetCommandLineArgument( const char * key ) const;
		void SetCommandLineArgument( const char * key, const char * value );

//		int ReadParameter(double param, const char * name_field, const unsigned int entry_nr);
		/**
		* Use this function to read a parameter from the parameter
		* file. 
		* - 'param' is the variable in which you want to store the 
		* parameter. Give it a default value, in case the desired
		* field in the ParameterFile does not exist.
		* - 'name_field' is the name of the field in the parameter
		* file.
		* - 'entry_nr' indicates which entry to take in the desired
		* field (start with 0).
		*
		* The function returns 0 if everything went well. In case
		* of an error: 1.
		* In case of errors, param still contains the value as during
		* the calling stage of this function.
		*/
		template <class T>
		int ReadParameter( T & param, const char * name_field, const unsigned int entry_nr, bool silent = false )
		{			
			/** \todo make a standard parameter file with the default values. */
			
			/** Very basic error-checking.*/
			if (VPF::set(param, m_ParameterFile[name_field][entry_nr])
				== VPF::INVALID)
			{
				if (!silent)
				{
					xl::xout["warning"] << "WARNING: Cannot find entry " << entry_nr <<
						" in the field " << name_field << "." << std::endl;
					xl::xout["warning"] << "         Default value " << param <<
						" is assumed." << std::endl;
				}
				
				/* param now still contains its original value!*/
				return 1;
			}

			/* param now contains the value from the parameterfile!*/
			
			return 0;

		} // end ReadParameter; this function must be defined here, otherwise the compiler cannot find it (because it's a template)
		
		/** Provide 'support' for doubles, by converting them to float */
		int ReadParameter(double & param, const char * name_field, const unsigned int entry_nr, bool silent)
		{
			float floatparam = static_cast<float>( param );
			int dummy =  this->ReadParameter( floatparam, name_field, entry_nr, silent );
			param = static_cast<double>( floatparam );
			return dummy;

		} // end ReadParameter

		int ReadParameter(double & param, const char * name_field, const unsigned int entry_nr)
		{
		  return ReadParameter(param, name_field, entry_nr, false);
		}


		/** Get/Set the name of the parameterFileName.*/
		itkGetStringMacro( ParameterFileName );
		itkSetStringMacro( ParameterFileName );

		/** Get and Set the elastix-level.*/
		itkSetMacro( ElastixLevel, unsigned int );
		itkGetMacro( ElastixLevel, unsigned int );

		/** Methods that have to be present everywhere.*/
		virtual int BeforeAll(void);

	protected:

		MyConfiguration(); 
		virtual ~MyConfiguration() {}; 
		
		/** Member variables.*/
		ParameterFileType					m_ParameterFile;
		mutable ArgumentMapType		m_ArgumentMap;
		std::string								m_ParameterFileName;
		bool											m_Initialized;
		unsigned int							m_ElastixLevel;
		
	private:

		MyConfiguration( const Self& );	// purposely not implemented
		void operator=( const Self& );	// purposely not implemented
		
	}; // end class MyConfiguration
	
		
} // end namespace elastix


#endif // end #ifndef	__elxMyConfiguration_H__

