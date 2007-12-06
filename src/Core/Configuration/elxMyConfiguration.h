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
	 * \brief A class that deals with user given parameters and command line arguments.
	 *
	 * The MyConfiguration class provides the functions 
	 * ReadParameter (to read parameters from the parameter file) and
	 * ReadCommandLineArgument, and provides an easy way to get the
	 * current elastix level.
	 *
	 * In elxConfigurationToUse.h this file can be included if this version
	 * of elx::MyConfiguration is desired. Currently there are no other choices.
   *
   * \parameter Silent: defines if warnings should be printed to screen, when 
   * a parameter cannot be found and the default is used.
   * example: <tt>(Silent "true")</tt>\n
   * Default: "false"
	 *
 	 * \sa ConfigurationBase
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
		
		/** Typedef's. */
		typedef VPF::ParameterFile		ParameterFileType;
		
		/** Pass the command line arguments as a map. 
		 * It should contain -p \<parfile\> or -tp \<parfile\>.
		 * The specified parameter file is read in memory.
     */
		virtual int Initialize( ArgumentMapType & _arg );

		/** True, if Initialize was succesfully called. */
		virtual bool Initialized( void ) const; //to elxconfigurationbase

		/** Get and Set CommandLine arguments into the argument map.*/
		const char * GetCommandLineArgument( const char * key ) const;
		void SetCommandLineArgument( const char * key, const char * value );

    /** Get/Set the name of the parameterFileName. */
		itkGetStringMacro( ParameterFileName );
		itkSetStringMacro( ParameterFileName );

		/** Get and Set the elastix-level.*/
		itkSetMacro( ElastixLevel, unsigned int );
		itkGetConstMacro( ElastixLevel, unsigned int );

    /** Set/Get whether warnings are allowed to be printed, when reading a parameter */
    itkSetMacro( Silent, bool );
    itkGetConstMacro( Silent, bool );

		/** Methods that is called at the very beginning of elastixTemplate::Run.
     * \li Prints the parameter file  */
		virtual int BeforeAll( void );

    /** Methods that is called at the very beginning of elastixTemplate::ApplyTransform.
     * \li Prints the parameter file  */
    virtual int BeforeAllTransformix( void );

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
    * - 'silent' (optional): if true, warnings and errors are not printed to screen.
    * if false, errors are always printed, and warnings are only printed when the m_Silent
    * is also false;
		*
		* The function returns 0 if everything went well. In case
		* of an error: 1.
		* In case of errors, param still contains the value as during
		* the calling stage of this function.
    *
    * \todo make a standard parameter file with the default values, based on the
    * the calls to this function
		*/
		template <class T>
		int ReadParameter( T & param, const char * name_field,
      const unsigned int entry_nr, bool silent )
    {
      VPF::ReturnStatusType ret = VPF::INVALID;
      try
      {
        ret = VPF::set( param, m_ParameterFile[ name_field ][ entry_nr ] );
      }
      catch ( itk::ExceptionObject & excp )
      {
        if ( !silent )
        {
          xl::xout["error"] << "ERROR: Unexpected error while reading parameter file.\n"
            << "Parameter file reader reports:\n"
            << excp
            << "\nDefault value will be assumed."
            << std::endl;
        }
        return 1;
      }

      /** Very basic error-checking. */
      if ( ret == VPF::INVALID )
      {
        if ( !silent && !this->GetSilent() )
        {
          xl::xout["warning"] << "WARNING: Cannot find entry " << entry_nr
            << " in the field " << name_field << "." << std::endl;
          xl::xout["warning"] << "         Default value " << param
            << " is assumed." << std::endl;
        }

        /* param now still contains its original value. */
        return 1;
      }

      /* param now contains the value from the parameterfile. */
      return 0;

    } // end ReadParameter()

    /** ReadParameter: with three inputs. */
    template <class T>
		int ReadParameter( T & param, const char * name_field,
      const unsigned int entry_nr )
    {
		  return ReadParameter( param, name_field, entry_nr, false );
		}
		
		/** Provide 'support' for doubles. */
		int ReadParameter( double & param, const char * name_field,
      const unsigned int entry_nr, bool silent );

		int ReadParameter( double & param, const char * name_field,
      const unsigned int entry_nr )
		{
		  return ReadParameter( param, name_field, entry_nr, false );
		}

    /** Provide 'support' for bools. */
		int ReadParameter( bool & param, const char * name_field,
      const unsigned int entry_nr, bool silent );
		
    int ReadParameter( bool & param, const char * name_field,
      const unsigned int entry_nr )
		{
		  return ReadParameter( param, name_field, entry_nr, false );
		}

    /** Convenience function to read parameters while specifying some more defaults.
     * This method adds two arguments: the prefix and the default_entry_nr.
     * prefix: try first to read \<prefix\>\<name_field\> from the parameter file.
     * If that fails, try \<name_field\>. 
     * default_entry_nr: if set to a value <0, it is ignored. if >=0, it indicates
     * the entry_nr used as a default when the entry_nr cannot be found. 
     */     
    template <class T>
		int ReadParameter( T & param, const char * name_field,
      const char * prefix, const unsigned int entry_nr,
      int default_entry_nr, bool silent )
    {
      std::string fullname( prefix );
      fullname += name_field;
      int ret = 1;

      /** Silently try to read the parameter. */
      if ( default_entry_nr >= 0 )
      {
        /** Try the default_entry_nr if the entry_nr is not found. */
        unsigned int uintdefault = static_cast<unsigned int>( default_entry_nr );
        ret &= this->ReadParameter( param, name_field, uintdefault, true );
        ret &= this->ReadParameter( param, name_field, entry_nr, true );
        ret &= this->ReadParameter( param, fullname.c_str(), uintdefault, true );
        ret &= this->ReadParameter( param, fullname.c_str(), entry_nr, true );
      }
      else
      {
        /** Just try the entry_nr. */
        ret &= this->ReadParameter( param, name_field, entry_nr, true );
        ret &= this->ReadParameter( param, fullname.c_str(), entry_nr, true );
      }

      /** If we haven't found anything, give a warning that the default value
      * provided by the caller is used.
      */
      if ( ret && !silent )
      {
        return this->ReadParameter( param, name_field, entry_nr, false );
      }

      return ret;

    } // end ReadParameter()

    /** ReadParameter: with five inputs. */
    template <class T>
		int ReadParameter( T & param, const char * name_field,
      const char * prefix, const unsigned int entry_nr,
      int default_entry_nr )
    {
      return ReadParameter( param, name_field, prefix, entry_nr, default_entry_nr, false );
    }

    /** Get the number of user supplied parameters. */
    template <class T>
		unsigned int CountNumberOfParameters( T & param, const char * name_field )
    {
      int ret = 0;
      unsigned int count = 0;
      while ( ret == 0 )
      {
        ret = this->ReadParameter( param, name_field, count, true );
        count++;
      }

      return count - 1;

    } // end CountNumberOfParameters()
   
	protected:

		MyConfiguration(); 
		virtual ~MyConfiguration() {}; 
		
		/** Member variables.*/
		ParameterFileType					m_ParameterFile;
		mutable ArgumentMapType		m_ArgumentMap;
		std::string								m_ParameterFileName;
		bool											m_Initialized;
		unsigned int							m_ElastixLevel;
    bool                      m_Silent;

    /** Print the parameter file to the logfile. Called by BeforeAll().
     * This function is not really generic. It's just added because it needs to be
     * called by both BeforeAll and BeforeAllTransformix.
     */
    virtual int PrintParameterFile( void );

	private:

		MyConfiguration( const Self& );	// purposely not implemented
		void operator=( const Self& );	// purposely not implemented

    std::string m_EmptyString;
		
	}; // end class MyConfiguration
	
		
} // end namespace elastix


#endif // end #ifndef	__elxMyConfiguration_H__

