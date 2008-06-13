/*======================================================================

This file is part of the elastix software.

Copyright (c) University Medical Center Utrecht. All rights reserved.
See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
details.

This software is distributed WITHOUT ANY WARRANTY; without even 
the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef __elxMyConfiguration_CXX__
#define __elxMyConfiguration_CXX__

#include "elxMyConfiguration.h"

namespace elastix
{
  using namespace itk;


  /**
  * ********************* Constructor ****************************
  */

  MyConfiguration::MyConfiguration()
  {
    /** Initialize stuff. */
    this->m_ParameterFileName = "";
    this->m_Initialized = false;
    this->m_ElastixLevel = 0;
    this->m_Silent = false;

  } // end Constructor()


  /**
  * ******************** PrintParameterFile ***************************
  * This function prints the ParameterFile to the log-file.
  */

  int MyConfiguration::PrintParameterFile(void)
  {
    /** Open the ParameterFile. */
    std::ifstream parfile( this->GetParameterFileName() );
    if ( parfile.is_open() )
    {
      /** Separate clearly in log-file. */
      xl::xout["logonly"] << std::endl << "=============== start of ParameterFile: "
        << GetParameterFileName() << " ===============" << std::endl;

      /** Read and write. */
      char inout;
      while ( !parfile.eof() )
      {
        parfile.get( inout );
        xl::xout["logonly"] << inout;
      }

      /** Separate clearly in log-file. */
      xl::xout["logonly"] << std::endl << "=============== end of ParameterFile: "
        << GetParameterFileName() << " ===============" << std::endl << std::endl;
    }
    else
    {
      xl::xout["error"] << "ERROR: the file \"" << GetParameterFileName() <<
        "\" could not be opened!" << std::endl;
      return 1;
    }

    /** Return a value. */
    return 0;

  } // end PrintParameterFile()


  /**
   * ************************ BeforeAll ***************************
   *
   * This function prints the ParameterFile to the log-file.
   */

  int MyConfiguration::BeforeAll(void)
  {
    return this->PrintParameterFile();
  } // end BeforeAll()


  /**
   * ************************ BeforeAllTransformix ***************************
   *
   * This function prints the ParameterFile to the log-file.
   */

  int MyConfiguration::BeforeAllTransformix(void)
  {
    return this->PrintParameterFile();
  } // end BeforeAllTransformix()


  /**
   * ********************** Initialize ****************************
   */

  int MyConfiguration::Initialize( ArgumentMapType & _arg )
  {
    this->m_ArgumentMap = _arg;

    /** This function can either be called by elastix or transformix.
     * If called by elastix the command line argument "-p" has to be
     * specified. If called by transformix the command line argument
     * "-tp" has to be specified.
     * NOTE: this implies that one can not use "-tp" for elastix and
     * "-p" for transformix.
     */

    std::string p = this->GetCommandLineArgument( "-p" );
    std::string tp = this->GetCommandLineArgument( "-tp" );

    if ( p != "" && tp == "" )
    {
      /** elastix called Initialize(). */
      this->SetParameterFileName( p.c_str() );
    }
    else if ( p == "" && tp != "" )
    {
      /** transformix called Initialize(). */
      this->SetParameterFileName( tp.c_str() );
    }
    else if ( p == "" && tp == "" )
    {
      xl::xout["error"] << "ERROR: No (Transform-)Parameter file has been entered" << std::endl;
      xl::xout["error"] << "for elastix: command line option \"-p\"" << std::endl;
      xl::xout["error"] << "for transformix: command line option \"-tp\"" << std::endl;
      return 1;
    }
    else
    {
      /** Both "p" and "tp" are used, which is prohibited. */
      xl::xout["error"] << "ERROR: Both \"-p\" and \"-tp\" are used, which is prohibited."
        << std::endl;
      return 1;
    }

    /** Open the ParameterFile. */
    try 
    {
      this->m_ParameterFile.Initialize( this->m_ParameterFileName.c_str() );
    }
    catch ( ... )
    {
      xl::xout["error"] << "ERROR: Reading the parameter file failed: "
        << this->m_ParameterFileName << std::endl;
      return 1;
    }

    this->m_Initialized = true;

    /** Check in the parameter file if silence is desired (less warnings). */
    bool silence = false;
    this->ReadParameter( silence, "Silent", 0, true );
    this->SetSilent( silence );

    /** Return a value.*/
    return 0;

  } // end Initialize()


  /**
   * ********************** Initialized ***************************
   *
   * Check if Initialized.
   */

  bool MyConfiguration::Initialized( void ) const
  {
    return this->m_Initialized;

  } // end Initialized()


  /**
   * ****************** GetCommandLineArgument ********************
   */

  const char * MyConfiguration::GetCommandLineArgument( const char * key ) const
  {
    /** Check if the argument was given. If yes return it. If no return "".*/
    if ( this->m_ArgumentMap.count( key ) == 0 )
    {
      return this->m_EmptyString.c_str();
    }
    else
    {
      return this->m_ArgumentMap[ key ].c_str();
    }

  } // end GetCommandLineArgument()


  /**
   * ****************** SetCommandLineArgument ********************
   */

  void MyConfiguration::SetCommandLineArgument( const char * key, const char * value )
  {
    /** Remove all (!) entries with key 'key' and
    * insert one entry ( key, value ).
    */
    this->m_ArgumentMap.erase( key );
    this->m_ArgumentMap.insert( EntryType( key, value ) );

  } // end SetCommandLineArgument()


  /**
   * ****************** ReadParameter ********************
   * Provide 'support' for doubles, by converting them to float.
   */

  int MyConfiguration::ReadParameter( double & param,
    const char * name_field,
    const unsigned int entry_nr,
    bool silent )
  {
    float floatparam = static_cast<float>( param );
    int dummy =  this->ReadParameter( floatparam, name_field, entry_nr, silent );
    param = static_cast<double>( floatparam );

    return dummy;

  } // end ReadParameter()


  /**
   * ****************** ReadParameter ********************
   * Provide 'support' for bools, by using strings and checking for "true" and "false".
   */

  int MyConfiguration::ReadParameter( bool & param,
    const char * name_field,
    const unsigned int entry_nr,
    bool silent )
  {
    std::string stringparam;
    if ( param ) stringparam = "true";
    else stringparam = "false";

    int dummy =  this->ReadParameter( stringparam, name_field, entry_nr, silent );

    if ( stringparam == "true" ) param = true;
    else param = false;

    return dummy;

  } // end ReadParameter()


} // end namespace elastix

#endif // end #ifndef __elxMyConfiguration_CXX__

